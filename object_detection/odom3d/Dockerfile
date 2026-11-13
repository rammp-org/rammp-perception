# syntax=docker/dockerfile:1
# odom3d world-anchored object tracker (rammp_odom3d). CPU only (numpy/scipy, ~4 ms per detection frame and
# <1 ms per camera frame on a desktop core), so it needs no GPU runtime. Build from the repo root:
#   docker build -f object_detection/odom3d/Dockerfile .

# rammp_perception_interfaces: rammp-interfaces-ros2 at INTERFACES_REF, unless a checkout was staged into
# interfaces/ (gitignored) to build against unreleased changes: `make odom3d INTERFACES=../rammp-interfaces-ros2`.
FROM ghcr.io/rammp-org/rammp-cuda:1.1.0-jp6 AS interfaces
ARG INTERFACES_REF=dev
COPY interfaces /staged
RUN if [ -d /staged/rammp_perception_interfaces ]; then \
      echo "using staged rammp_perception_interfaces" && cp -r /staged/rammp_perception_interfaces /rammp_perception_interfaces; \
    else \
      git clone --depth 1 --branch "${INTERFACES_REF}" https://github.com/rammp-org/rammp-interfaces-ros2.git /src \
      && mv /src/rammp_perception_interfaces /rammp_perception_interfaces; \
    fi

# Same base as the other perception images (ROS Humble, numpy 1.26), so one image cache serves all of them.
FROM ghcr.io/rammp-org/rammp-cuda:1.1.0-jp6

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN python3 -m pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1

# Small per-frame matrix ops: a multi-threaded OpenBLAS pool only burns cores (see the rfdetr Dockerfile).
ENV OPENBLAS_NUM_THREADS=1

COPY --from=interfaces /rammp_perception_interfaces /module_ws/src/rammp_perception_interfaces
COPY object_detection/odom3d/rammp_odom3d /module_ws/src/rammp_odom3d
WORKDIR /module_ws
RUN . /opt/ros/humble/setup.sh \
 && . /ros2_ws/install/setup.sh \
 && colcon build --event-handlers console_direct- \
 && rm -rf build log

# Build-time check: the node's module loads and the core passes its synthetic tests.
RUN . /module_ws/install/setup.sh \
 && python3 -c "import rammp_odom3d.node; print('odom3d ok')" \
 && python3 -m pip install --no-cache-dir pytest==8.3.3 \
 && python3 -m pytest -q /module_ws/src/rammp_odom3d/test

# The node binary itself, not `ros2 run`, so SIGTERM reaches it.
ENTRYPOINT ["/ros_entrypoint.sh", "/module_ws/install/rammp_odom3d/lib/rammp_odom3d/odom3d_node"]
