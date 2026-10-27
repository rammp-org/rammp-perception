# syntax=docker/dockerfile:1
# NanoOWL (OWL-ViT, TensorRT FP16) open-vocabulary detection as an always-on detector on the GPU. Boxes only.
# Jetson Orin only: FROM rammp-cuda, run with --runtime nvidia. Build from the repo root:
#   docker build -f object_detection/nanoowl/Dockerfile .
#
# Weights are baked in (public, about 600 MB per model). TensorRT engines are not: they are specific to the GPU and
# the TensorRT version, so the node builds one into /cache/engines on first start (about 2 minutes) and reuses it
# after. Mount a volume there.

# rammp_perception_interfaces: rammp-interfaces-ros2 at INTERFACES_REF, unless a checkout was staged into
# interfaces/ (gitignored) to build against unreleased changes: `make nanoowl INTERFACES=../rammp-interfaces-ros2`.
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

# TensorRT 10.3 runtime and Python bindings, as JetPack 6.2.1 ships them (L4T r36.4). Engines are built through
# the Python API, so no trtexec or headers.
ARG TRT_VERSION=10.3.0.30-1+cuda12.5
RUN curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc -o /etc/apt/trusted.gpg.d/jetson-ota-public.asc \
 && echo "deb https://repo.download.nvidia.com/jetson/common r36.4 main" > /etc/apt/sources.list.d/nvidia-l4t-common.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      "libnvinfer10=${TRT_VERSION}" "libnvinfer-plugin10=${TRT_VERSION}" "libnvinfer-vc-plugin10=${TRT_VERSION}" \
      "libnvonnxparsers10=${TRT_VERSION}" "python3-libnvinfer=${TRT_VERSION}" \
 && rm -rf /var/lib/apt/lists/*

# Pinned to the versions measured on abra (the same as SAM3's and RF-DETR's). torch and torchvision come from
# rammp-cuda; numpy stays at its pin.
RUN pip install --no-cache-dir \
      transformers==5.19.0 tokenizers==0.23.2 huggingface-hub==1.33.0 safetensors==0.8.0 \
      onnx==1.23.2 supervision==0.30.9 numpy==1.26.4

# supervision depends on the desktop opencv-python, which needs libGL. Replace it with the headless build (both
# install into cv2/, so remove both, then reinstall headless).
RUN pip uninstall -y opencv-python opencv-python-headless \
 && pip install --no-cache-dir --no-deps opencv-python-headless==4.11.0.86

# Roboflow's trackers (Apache-2.0) for ByteTrack, without its numpy>=2 pin (see the rfdetr Dockerfile).
RUN pip install --no-cache-dir --no-deps trackers==2.6.1 \
 && pip install --no-cache-dir rich==13.9.4

# OWL-ViT B/32 and B/16 (Apache-2.0), at the revisions measured. Safetensors weights, config and tokenizer only.
# B/16's main branch has only pytorch_model.bin; its safetensors are Hugging Face's conversion (refs/pr/9), the
# file transformers itself loads. `hf download` skips a missing file silently, hence the check after.
RUN printf '%s\n' \
      "owlvit-b32 google/owlvit-base-patch32 cbc355fb364588351c5d51c7f74465e8e7ec6f72" \
      "owlvit-b16 google/owlvit-base-patch16 10e84248055232dfc4786bfead6206c3128d95ad" \
    | while read -r name repo rev; do \
      hf download "$repo" config.json preprocessor_config.json tokenizer_config.json special_tokens_map.json \
        vocab.json merges.txt model.safetensors --revision "$rev" --local-dir "/models/owlvit/$name" \
      && rm -rf "/models/owlvit/$name/.cache" \
      && test -s "/models/owlvit/$name/model.safetensors" || exit 1; \
    done

# One thread for numpy's OpenBLAS: its pool otherwise spins about 5 cores per detector between frames
# (rammp-org/rammp-deployments#106).
ENV OPENBLAS_NUM_THREADS=1
# sheppy runs containers as the host user (1000), which has no passwd entry here. PyTorch names its compile cache
# after the user and fails without one; give it a fixed directory instead.
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor HF_HUB_OFFLINE=1

COPY --from=interfaces /rammp_perception_interfaces /module_ws/src/rammp_perception_interfaces
COPY object_detection/common/rammp_detection_common /module_ws/src/rammp_detection_common
COPY object_detection/nanoowl/rammp_nanoowl /module_ws/src/rammp_nanoowl
WORKDIR /module_ws
RUN . /opt/ros/humble/setup.sh \
 && . /ros2_ws/install/setup.sh \
 && colcon build --event-handlers console_direct- \
 && rm -rf build log

# Build-time check that needs no GPU: the stack imports, the node's modules load, and the tokenizer reads the
# baked weights. TensorRT is only located: importing it needs libnvdla_compiler.so from the host.
RUN . /module_ws/install/setup.sh && python3 -c "\
import importlib.util, transformers, trackers, supervision; \
assert importlib.util.find_spec('tensorrt'), 'tensorrt python bindings missing'; \
from transformers import OwlViTProcessor; \
[OwlViTProcessor.from_pretrained(f'/models/owlvit/{m}') for m in ('owlvit-b32', 'owlvit-b16')]; \
from rammp_perception_interfaces.msg import ObjectDetections; \
import rammp_nanoowl.node, rammp_nanoowl.owl; \
print('transformers', transformers.__version__)"

COPY object_detection/config/classes.yaml /etc/rammp/perception/classes.yaml
# World-writable so a fresh named volume mounted here is writable by the host user sheppy runs as.
RUN mkdir -p /cache/engines && chmod 1777 /cache/engines

# The node binary itself, not `ros2 run`, so SIGTERM reaches it (see the yoloe Dockerfile).
ENTRYPOINT ["/ros_entrypoint.sh", "/module_ws/install/rammp_nanoowl/lib/rammp_nanoowl/nanoowl_node"]
