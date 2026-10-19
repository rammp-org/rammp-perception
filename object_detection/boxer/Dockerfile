# syntax=docker/dockerfile:1
# BoxerNet 3D boxes for any detector's ObjectDetections (rammp_boxer). Jetson Orin only: FROM rammp-cuda, run with
# --runtime nvidia. Build from the repo root:
#   docker build -f object_detection/boxer/Dockerfile .
#
# Boxer's code and the two checkpoints BoxerNet uses are baked in (public; CC-BY-NC 4.0, non-commercial). TensorRT
# engines are not: they are specific to the TensorRT version, so the node exports and builds them into
# /cache/engines on first start (a few minutes) and reuses them after. Mount a volume there.

# rammp_perception_interfaces: rammp-interfaces-ros2 at INTERFACES_REF, unless a checkout was staged into
# interfaces/ (gitignored) to build against unreleased changes: `make boxer INTERFACES=../rammp-interfaces-ros2`.
FROM ghcr.io/rammp-org/rammp-cuda:1.1.0-jp6 AS interfaces
# A branch until rammp-interfaces-ros2 tags a release with the perception package.
ARG INTERFACES_REF=dev
COPY interfaces /staged
RUN if [ -d /staged/rammp_perception_interfaces ]; then \
      echo "using staged rammp_perception_interfaces" && cp -r /staged/rammp_perception_interfaces /rammp_perception_interfaces; \
    else \
      git clone --depth 1 --branch "${INTERFACES_REF}" https://github.com/rammp-org/rammp-interfaces-ros2.git /src \
      && mv /src/rammp_perception_interfaces /rammp_perception_interfaces; \
    fi

FROM ghcr.io/rammp-org/rammp-cuda:1.1.0-jp6

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# TensorRT 10.3 runtime, builder and Python bindings, as JetPack 6.2.1 ships them (L4T r36.4), as in the yoloe image.
ARG TRT_VERSION=10.3.0.30-1+cuda12.5
RUN curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc -o /etc/apt/trusted.gpg.d/jetson-ota-public.asc \
 && echo "deb https://repo.download.nvidia.com/jetson/common r36.4 main" > /etc/apt/sources.list.d/nvidia-l4t-common.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      "libnvinfer10=${TRT_VERSION}" "libnvinfer-plugin10=${TRT_VERSION}" "libnvinfer-vc-plugin10=${TRT_VERSION}" \
      "libnvonnxparsers10=${TRT_VERSION}" "python3-libnvinfer=${TRT_VERSION}" \
 && rm -rf /var/lib/apt/lists/*

# torch comes from rammp-cuda; numpy stays at its pin. onnx exports the engines on first start.
RUN python3 -m pip install --no-cache-dir onnx==1.23.2 numpy==1.26.4 opencv-python-headless==4.11.0.86

# Boxer (facebookresearch/boxer) at the commit measured, and the BoxerNet and DINOv3 checkpoints it loads from
# its ckpts/. Only the model code is used: no dataset loaders, viewers or OWL.
ARG BOXER_COMMIT=1f86542
ARG CKPTS=https://huggingface.co/facebook/boxer/resolve/main
RUN git clone https://github.com/facebookresearch/boxer.git /opt/boxer \
 && git -C /opt/boxer checkout "${BOXER_COMMIT}" \
 && rm -rf /opt/boxer/.git \
 && mkdir -p /opt/boxer/ckpts \
 && for f in boxernet_hw960in2x6d768-c88128f8.ckpt dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth; do \
      curl -fsSL -o "/opt/boxer/ckpts/$f" "$CKPTS/$f"; \
    done
ENV BOXER_ROOT=/opt/boxer BOXER_COMMIT=${BOXER_COMMIT}

# numpy's OpenBLAS starts a worker per core (11 here) that busy-waits between calls. Small per-frame matrix ops
# (trackers, box geometry) wake them every frame: the pool kept ~5 cores busy at 10 Hz for ~40 ms of real work (snorlax:
# YOLOE-26m 5.0 -> 0.32 cores, RF-DETR S 2.4 -> 0.4, same per-frame time). Matrices this small gain nothing from it.
ENV OPENBLAS_NUM_THREADS=1

COPY --from=interfaces /rammp_perception_interfaces /module_ws/src/rammp_perception_interfaces
COPY object_detection/common/rammp_detection_common /module_ws/src/rammp_detection_common
COPY object_detection/boxer/rammp_boxer /module_ws/src/rammp_boxer
WORKDIR /module_ws
RUN . /opt/ros/humble/setup.sh \
 && . /ros2_ws/install/setup.sh \
 && colcon build --event-handlers console_direct- \
 && rm -rf build log

# Build-time check that needs no GPU: the stack imports, Boxer's model code loads, and the interface has the size
# field. TensorRT is only located: importing it needs libraries the NVIDIA runtime mounts from the host.
RUN . /module_ws/install/setup.sh && python3 -c "\
import importlib.util, sys, cv2, onnx, tf2_ros; \
assert importlib.util.find_spec('tensorrt'), 'tensorrt python bindings missing'; \
from rammp_perception_interfaces.msg import DetectedObject; assert hasattr(DetectedObject(), 'size'); \
sys.path.insert(0, '/opt/boxer'); import boxernet.boxernet, utils.tw.obb; \
import rammp_boxer.node, rammp_boxer.geometry, rammp_boxer.sync; \
print('boxer ok')"

# World-writable so a fresh named volume mounted here is writable by the host user sheppy runs as.
RUN mkdir -p /cache/engines && chmod 1777 /cache/engines

# The node binary itself, not `ros2 run`: as PID 1, `ros2 run` does not pass SIGTERM on.
ENTRYPOINT ["/ros_entrypoint.sh", "/module_ws/install/rammp_boxer/lib/rammp_boxer/boxer_node"]
