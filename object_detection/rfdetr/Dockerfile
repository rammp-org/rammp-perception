# syntax=docker/dockerfile:1
# RF-DETR-Seg (COCO weights) instance segmentation as an always-on detector on the GPU.
# Jetson Orin only: FROM rammp-cuda, run with --runtime nvidia. Build from the repo root:
#   docker build -f object_detection/rfdetr/Dockerfile .

# rammp_perception_interfaces: the tagged rammp-interfaces-ros2 release, unless a checkout was staged into
# interfaces/ (gitignored) to build against unreleased changes: `make rfdetr INTERFACES=../rammp-interfaces-ros2`.
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

RUN apt-get update \
 && apt-get install -y --no-install-recommends ros-humble-vision-msgs \
 && rm -rf /var/lib/apt/lists/*

# Pinned to the versions measured on abra. torch and torchvision come from rammp-cuda; numpy stays at its pin.
RUN pip install --no-cache-dir \
      rfdetr==1.11.2 transformers==5.19.0 supervision==0.30.9 numpy==1.26.4

# ultralytics and supervision depend on the desktop opencv-python, which needs libGL. Replace it with the
# headless build (both install into cv2/, so remove both, then reinstall headless).
RUN pip uninstall -y opencv-python opencv-python-headless \
 && pip install --no-cache-dir --no-deps opencv-python-headless==4.11.0.86

# Roboflow's trackers (Apache-2.0) for ByteTrack. Every release with ByteTrack declares numpy>=2, but the code
# runs on rammp-cuda's numpy 1.26 (checked on abra), so it goes in without its dependency pins.
RUN pip install --no-cache-dir --no-deps trackers==2.6.1 \
 && pip install --no-cache-dir rich==13.9.4

# COCO-pretrained RF-DETR-Seg weights (Apache-2.0), at the names and checksums rfdetr 1.11.2 expects.
ENV RF_HOME=/models/rfdetr
RUN mkdir -p "$RF_HOME" \
 && curl -fsSL -o "$RF_HOME/rf-detr-seg-nano.pt"   https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth \
 && curl -fsSL -o "$RF_HOME/rf-detr-seg-small.pt"  https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth \
 && curl -fsSL -o "$RF_HOME/rf-detr-seg-medium.pt" https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth \
 && printf '%s  %s\n' \
      9995497791d0ff1664a1d9ddee9cfd20 "$RF_HOME/rf-detr-seg-nano.pt" \
      0a2a3006381d0c42853907e700eadd08 "$RF_HOME/rf-detr-seg-small.pt" \
      a49af1562c3719227ad43d0ca53b4c7a "$RF_HOME/rf-detr-seg-medium.pt" | md5sum -c -

# sheppy runs containers as the host user (1000), which has no passwd entry here. PyTorch names its compile cache
# after the user and fails without one; give it a fixed directory instead.
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

COPY --from=interfaces /rammp_perception_interfaces /module_ws/src/rammp_perception_interfaces
COPY object_detection/common/rammp_detection_common /module_ws/src/rammp_detection_common
COPY object_detection/rfdetr/rammp_rfdetr /module_ws/src/rammp_rfdetr
WORKDIR /module_ws
RUN . /opt/ros/humble/setup.sh \
 && . /ros2_ws/install/setup.sh \
 && colcon build --event-handlers console_direct- \
 && rm -rf build log

# Build-time check that needs no GPU: the stack imports and the node's module loads.
RUN . /module_ws/install/setup.sh && python3 -c "\
import rfdetr, trackers; \
from rammp_perception_interfaces.msg import InstanceMasks; \
import rammp_rfdetr.node; \
print('rfdetr ok')"

COPY object_detection/config/classes.yaml /etc/rammp/perception/classes.yaml

ENTRYPOINT ["/ros_entrypoint.sh", "ros2", "run", "rammp_rfdetr", "rfdetr_node"]
