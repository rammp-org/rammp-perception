# syntax=docker/dockerfile:1
# YOLOE instance segmentation as an always-on detector, on the GPU or a DLA core (`device` parameter).
# Jetson Orin only: FROM rammp-cuda, run with --runtime nvidia. Build from the repo root:
#   docker build -f object_detection/yoloe/Dockerfile .
#
# Weights and the text encoder are baked in (small, public). TensorRT engines are not: they are specific to the
# device, the TensorRT version and the class prompts, so the node builds one into /cache/engines on first start
# and reuses it after. Mount a volume there.

# rammp_perception_interfaces: the tagged rammp-interfaces-ros2 release, unless a checkout was staged into
# interfaces/ (gitignored) to build against unreleased changes: `make yoloe INTERFACES=../rammp-interfaces-ros2`.
FROM ghcr.io/rammp-org/rammp-cuda:1.1.0-jp6 AS interfaces
ARG INTERFACES_REF=v1.2.0
COPY interfaces /staged
RUN if [ -d /staged/rammp_perception_interfaces ]; then \
      echo "using staged rammp_perception_interfaces" && cp -r /staged/rammp_perception_interfaces /rammp_perception_interfaces; \
    else \
      git clone --depth 1 --branch "${INTERFACES_REF}" https://github.com/rammp-org/rammp-interfaces-ros2.git /src \
      && mv /src/rammp_perception_interfaces /rammp_perception_interfaces; \
    fi

FROM ghcr.io/rammp-org/rammp-cuda:1.1.0-jp6

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# TensorRT 10.3 runtime, builder and Python bindings, as JetPack 6.2.1 ships them (L4T r36.4). DLA engines also
# need libnvdla_compiler.so, which the NVIDIA runtime mounts from the host's L4T at run time.
ARG TRT_VERSION=10.3.0.30-1+cuda12.5
RUN curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc -o /etc/apt/trusted.gpg.d/jetson-ota-public.asc \
 && echo "deb https://repo.download.nvidia.com/jetson/common r36.4 main" > /etc/apt/sources.list.d/nvidia-l4t-common.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      "libnvinfer10=${TRT_VERSION}" "libnvinfer-plugin10=${TRT_VERSION}" "libnvinfer-vc-plugin10=${TRT_VERSION}" \
      "libnvonnxparsers10=${TRT_VERSION}" "python3-libnvinfer=${TRT_VERSION}" \
 && rm -rf /var/lib/apt/lists/*

# Pinned to the versions measured on abra. torch and torchvision come from rammp-cuda; numpy stays at its pin.
# clip (Ultralytics' fork, pinned to the commit measured) supplies the tokenizer YOLOE uses to embed prompts. It
# builds from source, which the base image's pip 22 cannot do (it reports the package as "unknown"), hence the
# newer pip. lap is the assignment solver Ultralytics' trackers need.
RUN python3 -m pip install --no-cache-dir pip==24.3.1 \
 && python3 -m pip install --no-cache-dir \
      ultralytics==8.4.174 onnx==1.23.2 onnxslim==0.1.98 numpy==1.26.4 \
      "clip @ git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40" \
      lap==0.5.12

# ultralytics and supervision depend on the desktop opencv-python, which needs libGL. Replace it with the
# headless build (both install into cv2/, so remove both, then reinstall headless).
RUN pip uninstall -y opencv-python opencv-python-headless \
 && pip install --no-cache-dir --no-deps opencv-python-headless==4.11.0.86

# Weights for both sizes and the text encoder YOLOE-26 uses to turn prompts into embeddings at export.
ARG ASSETS=https://github.com/ultralytics/assets/releases/download/v8.4.0
RUN mkdir -p /models/yoloe \
 && for f in yoloe-26s-seg.pt yoloe-26m-seg.pt mobileclip2_b.ts; do curl -fsSL -o "/models/yoloe/$f" "$ASSETS/$f"; done
# Never pip-install or download at run time: what the image has is what runs.
ENV YOLO_AUTOINSTALL=false YOLO_OFFLINE=true YOLO_CONFIG_DIR=/tmp/ultralytics

# sheppy runs containers as the host user (1000), which has no passwd entry here. PyTorch names its compile cache
# after the user and fails without one; give it a fixed directory instead.
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

COPY --from=interfaces /rammp_perception_interfaces /module_ws/src/rammp_perception_interfaces
COPY object_detection/common/rammp_detection_common /module_ws/src/rammp_detection_common
COPY object_detection/yoloe/rammp_yoloe /module_ws/src/rammp_yoloe
WORKDIR /module_ws
RUN . /opt/ros/humble/setup.sh \
 && . /ros2_ws/install/setup.sh \
 && colcon build --event-handlers console_direct- \
 && rm -rf build log

# Build-time check that needs no GPU: the stack imports and the node's modules load. TensorRT is only located:
# importing it needs libnvdla_compiler.so from the host.
RUN . /module_ws/install/setup.sh && python3 -c "\
import importlib.util, ultralytics, clip, lap; \
assert importlib.util.find_spec('tensorrt'), 'tensorrt python bindings missing'; \
from rammp_perception_interfaces.msg import ObjectDetections; \
import rammp_yoloe.node, rammp_yoloe.engine; \
print('ultralytics', ultralytics.__version__)"

COPY object_detection/config/classes.yaml /etc/rammp/perception/classes.yaml
# World-writable so a fresh named volume mounted here is writable by the host user sheppy runs as.
RUN mkdir -p /cache/engines && chmod 1777 /cache/engines

# The whole default invocation, so a deployment only appends: sheppy adds `--ros-args --params-file` for its
# params block, and `-p device:=dla0` moves it to a DLA core.
ENTRYPOINT ["/ros_entrypoint.sh", "ros2", "run", "rammp_yoloe", "yoloe_node"]
