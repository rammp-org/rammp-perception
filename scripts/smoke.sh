#!/usr/bin/env bash
# Floor-tier smoke test: the image runs, it publishes, and it exits on SIGTERM.
#
# No --init on purpose. The container runs the way sheppyd will run it, so a
# process that ignores SIGTERM fails here rather than on the robot.
#
# Hardware-free: the node runs with mock:=true, which loads no model and needs
# no GPU or camera, and publishes empty results on its real topics.
set -euo pipefail

PATTERN="${SMOKE_PATTERN:-MOCK: publishing empty results}"

IMAGE="${1:?usage: smoke.sh <image>}"
NAME="rammp-smoke-$$"

# A host with its own Cyclone config (abra keeps DDS on loopback) passes it in the way sheppy does. Without it
# the image's default config needs free participant indices on the real NIC, which a busy robot may not have.
DDS_ARGS=()
if [ -n "${CYCLONEDDS_URI:-}" ]; then
  DDS_FILE="${CYCLONEDDS_URI#file://}"
  DDS_ARGS=(-e "CYCLONEDDS_URI=$CYCLONEDDS_URI" -v "$DDS_FILE:$DDS_FILE:ro")
fi

docker run -d --name "$NAME" --network host "${DDS_ARGS[@]}" \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" "$IMAGE" --ros-args -p mock:=true >/dev/null
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

sleep 5

if ! docker logs "$NAME" 2>&1 | grep -q "$PATTERN"; then
  echo "FAIL: no match for pattern '$PATTERN' in the container logs"
  docker logs "$NAME" 2>&1 | tail -20
  exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then
  echo "FAIL: container exited on its own before SIGTERM was sent"
  docker logs "$NAME" 2>&1 | tail -20
  exit 1
fi

if ! docker exec "$NAME" bash -c \
    '. /opt/ros/humble/setup.bash && . /ros2_ws/install/setup.bash && . /module_ws/install/setup.bash \
     && timeout 10 ros2 topic echo --once --no-arr /perception/object_detections' \
    >/dev/null 2>&1; then
  echo "FAIL: nothing received on /perception/object_detections"
  docker logs "$NAME" 2>&1 | tail -20
  exit 1
fi

docker stop -t 10 "$NAME" >/dev/null
CODE="$(docker inspect -f '{{.State.ExitCode}}' "$NAME")"
if [ "$CODE" = "137" ]; then
  echo "FAIL: the node ignored SIGTERM and was killed after the timeout (exit 137)"
  exit 1
fi
echo "PASS: $IMAGE publishes and exits on SIGTERM (exit $CODE)"
