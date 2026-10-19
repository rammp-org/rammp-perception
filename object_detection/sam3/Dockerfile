# syntax=docker/dockerfile:1
# SAM3 open-vocabulary detection and tracking. Jetson Orin only: FROM rammp-cuda, run with --runtime nvidia.
# Build from the repo root: docker build -f object_detection/sam3/Dockerfile .
#
# Deliberately not in the image:
#   * weights. facebook/sam3 is gated on Hugging Face and 3.4 GB. Mount a downloaded snapshot at /models/sam3.
#   * TensorRT engines. They are specific to the GPU and the TensorRT version, so they are built once per machine
#     into a volume at /cache/engines (see sam3.yaml). Without them the node runs the PyTorch encoders: slower,
#     same output.

# The interface package: the tagged release, unless a checkout was staged into interfaces/ (gitignored) to build
# against unreleased changes: `make sam3 INTERFACES=../rammp-interfaces-ros2`.
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

# TensorRT 10.3 runtime and Python bindings, as JetPack 6.2.1 ships them (L4T r36.4). Runtime only: engines are
# built through the Python API, so no trtexec or headers.
ARG TRT_VERSION=10.3.0.30-1+cuda12.5
RUN curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc -o /etc/apt/trusted.gpg.d/jetson-ota-public.asc \
 && echo "deb https://repo.download.nvidia.com/jetson/common r36.4 main" > /etc/apt/sources.list.d/nvidia-l4t-common.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      "libnvinfer10=${TRT_VERSION}" "libnvinfer-plugin10=${TRT_VERSION}" "libnvinfer-vc-plugin10=${TRT_VERSION}" \
      "libnvonnxparsers10=${TRT_VERSION}" "python3-libnvinfer=${TRT_VERSION}" \
 && rm -rf /var/lib/apt/lists/*

# Pinned to the versions the node was measured with. rammp_sam3's batched tracker patches transformers' private
# tracker methods and switches itself off on any other transformers version. numpy stays at rammp-cuda's pin.
RUN pip install --no-cache-dir \
      transformers==5.19.0 tokenizers==0.23.2 huggingface-hub==1.33.0 safetensors==0.8.0 \
      onnx==1.23.2 onnxscript==0.7.2 numpy==1.26.4

# sheppy runs containers as the host user (1000), which has no passwd entry here. PyTorch names its compile cache
# after the user and fails without one; give it a fixed directory instead.
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

COPY --from=interfaces /rammp_perception_interfaces /module_ws/src/rammp_perception_interfaces
COPY object_detection/sam3/rammp_sam3 /module_ws/src/rammp_sam3
WORKDIR /module_ws
RUN . /opt/ros/humble/setup.sh \
 && . /ros2_ws/install/setup.sh \
 && colcon build --event-handlers console_direct- \
 && rm -rf build log

# Build-time check that needs no GPU: the stack imports and the node's modules load. TensorRT is only located,
# not imported: it links libnvdla_compiler.so, which the NVIDIA runtime mounts from the host's L4T at run time.
RUN . /module_ws/install/setup.sh && python3 -c "\
import importlib.util, transformers, torch; \
assert importlib.util.find_spec('tensorrt'), 'tensorrt python bindings missing'; \
from rammp_perception_interfaces.srv import Detect, StartTracking; from rammp_perception_interfaces.msg import ObjectDetections; \
import rammp_sam3.node, rammp_sam3.engine, rammp_sam3.build_engines; \
print('transformers', transformers.__version__, '/ torch', torch.__version__)"

COPY object_detection/sam3/sam3.yaml /etc/rammp/sam3/sam3.yaml
# World-writable so a fresh named volume mounted here is writable by the host user sheppy runs as. Without it a new
# sam3_engines volume is root-owned and build_engines fails with Permission denied writing its ONNX file.
RUN mkdir -p /cache/engines && chmod 1777 /cache/engines

# The whole default invocation, so a deployment only appends: sheppy adds `--ros-args --params-file` for its
# params block. Building engines uses the same image with the entrypoint swapped (see sam3.yaml).
# The node binary itself, not `ros2 run`, so SIGTERM reaches it (see the yoloe Dockerfile).
ENTRYPOINT ["/ros_entrypoint.sh", "/module_ws/install/rammp_sam3/lib/rammp_sam3/sam3_node", \
            "--ros-args", "-r", "__node:=sam3", "--params-file", "/etc/rammp/sam3/sam3.yaml"]
