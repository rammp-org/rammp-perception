"""The always-on detector node every backend in this repo runs inside.

A backend only turns an RGB frame into Detections with labels from object_detection/config/classes.yaml. This node does the
rest the same way for all of them, so they are interchangeable behind one sheppy node:

Topics (the node runs as /perception/scene_segmenter):
  /perception/object_detections  rammp_perception_interfaces/ObjectDetections  every processed frame: objects and
                                                                                a mono16 label image of their masks
  ~/overlay/compressed           sensor_msgs/CompressedImage                    masks drawn on the frame, only while
                                                                                subscribed and publish_overlay is true
Ids come from the backend's tracker and stay with an object while it is tracked; a backend that does not
track gets per-frame ids (1..N, highest score first). instance_id is "<class>_<id>"; mask_id is the id.
State is always STATE_VISIBLE: a lost object is dropped.
Headers copy the camera frame's stamp and frame_id.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from typing import Callable, Protocol

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from rammp_perception_interfaces.msg import DetectedObject, ObjectDetections

from .classes import ClassMap, Detection, clean

_CHANNELS = {
    "rgb8": (3, False),
    "bgr8": (3, True),
    "rgba8": (4, False),
    "bgra8": (4, True),
}
MOCK_RATE_HZ = 10.0
_PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 212],
    ],
    np.uint8,
)  # RGB


class Backend(Protocol):
    description: str

    def infer(self, rgb: np.ndarray) -> list[Detection]:
        """Detections for one H x W x 3 RGB frame, labels already mapped to the class list."""


def instance_id(label: str, id: int) -> str:
    """DetectedObject.instance_id: "<class>_<id>", e.g. "food_container_3"."""
    return f"{label.replace(' ', '_')}_{id}"


def image_to_rgb(msg: Image) -> np.ndarray:
    if msg.encoding not in _CHANNELS:
        raise ValueError(
            f"unsupported image encoding {msg.encoding!r}; expected one of {sorted(_CHANNELS)}"
        )
    ch, bgr = _CHANNELS[msg.encoding]
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[
        :, : msg.width * ch
    ]
    a = a.reshape(msg.height, msg.width, ch)[:, :, :3]
    return np.ascontiguousarray(a[:, :, ::-1] if bgr else a)


def label_image(dets: list[Detection], height: int, width: int) -> np.ndarray:
    """uint16, each pixel the id of the instance covering it, 0 for none. Highest score wins overlaps."""
    out = np.zeros((height, width), np.uint16)
    for d in sorted(dets, key=lambda d: d.score):
        if d.mask is not None:
            out[d.mask] = d.id
        elif d.polygon is not None and len(d.polygon) >= 3:
            cv2.fillPoly(out, [np.round(d.polygon).astype(np.int32)], int(d.id))
    return out


class DetectorNode(Node):
    def __init__(
        self,
        make_backend: Callable[[Node, ClassMap], Backend],
        node_name: str = "scene_segmenter",
    ):
        super().__init__(node_name, namespace="perception")
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.image_topic = p("image_topic", "/scene_camera/color/image_raw")
        # Process every frame_stride-th camera frame: 1 = every frame, 3 = 10 Hz from a 30 fps camera.
        self.frame_stride = max(1, int(p("frame_stride", 3)))
        self.publish_overlay = bool(p("publish_overlay", True))
        # Read by backends that track: whether to, and for how many processed frames a lost object keeps its id.
        self.tracking = bool(p("tracking", True))
        self.track_buffer_frames = max(1, int(p("track_buffer_frames", 10)))
        classes_file = p("classes_file", "/etc/rammp/perception/classes.yaml")
        mock = bool(p("mock", False))

        self.classes = ClassMap.load(classes_file)
        self.colors = {
            label: _PALETTE[i % len(_PALETTE)]
            for i, label in enumerate(self.classes.labels)
        }
        self.get_logger().info(f"classes from {classes_file}: {self.classes.labels}")

        # Not under the node's name: one topic, whichever segmenter is running.
        self.pub_detections = self.create_publisher(
            ObjectDetections, "object_detections", 10
        )
        self.pub_overlay = self.create_publisher(
            CompressedImage, "~/overlay/compressed", 10
        )

        if mock:
            # Hardware-free stand-in: empty results on the real topics at the real rate, no model, no camera.
            self.backend = None
            self.create_timer(1.0 / MOCK_RATE_HZ, self._tick_mock)
            self.get_logger().info(
                f"MOCK: publishing empty results at {MOCK_RATE_HZ} Hz"
            )
            return

        self.backend = make_backend(self, self.classes)
        self.get_logger().info(f"backend ready: {self.backend.description}")

        self.latest: Image | None = None
        self.frames_received = 0  # camera frames so far
        self.frames_at_last = -(
            10**9
        )  # frames_received when the last processed frame arrived
        self.frame_ready = threading.Condition()
        self.create_subscription(
            Image, self.image_topic, self._on_image, qos_profile_sensor_data
        )
        self.times: list[float] = []
        self.counts: Counter = Counter()
        self.last_log = time.monotonic()
        self.frames_logged = 0
        self.get_logger().info(
            f"detecting on {self.image_topic}, every {self.frame_stride} camera frames"
        )
        threading.Thread(target=self._loop, daemon=True).start()

    def _on_image(self, msg: Image) -> None:
        with self.frame_ready:
            self.latest = msg
            self.frames_received += 1
            self.frame_ready.notify()

    def _loop(self) -> None:
        """Process every frame_stride-th camera frame.

        Frames are counted as they arrive. When frame_stride frames have come in since the last processed one,
        the newest is processed. A frame that takes longer than frame_stride frame periods is not waited out to
        the next exact multiple (that would halve the rate): the newest frame is taken as soon as the node is
        free, so a slow backend runs as fast as it can and never faster than the stride.
        """
        while rclpy.ok():
            with self.frame_ready:
                if not self.frame_ready.wait_for(
                    lambda: self.latest is not None
                    and self.frames_received - self.frames_at_last >= self.frame_stride,
                    timeout=0.5,
                ):
                    continue
                msg, self.frames_at_last = self.latest, self.frames_received
            self._process(msg)

    def _tick_mock(self) -> None:
        header = Image().header
        header.stamp = self.get_clock().now().to_msg()
        self.pub_detections.publish(
            ObjectDetections(
                header=header, label_image=Image(header=header, encoding="mono16")
            )
        )

    def _process(self, msg: Image) -> None:
        t0 = time.perf_counter()
        try:
            rgb = image_to_rgb(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=10.0)
            return

        dets = clean(self.backend.infer(rgb), self.classes)
        h, w = rgb.shape[:2]
        labels = label_image(dets, h, w)
        self._publish(msg, dets, labels)
        if self.publish_overlay and self.pub_overlay.get_subscription_count():
            self._publish_overlay(msg, rgb, dets, labels)

        self.times.append(time.perf_counter() - t0)
        self.counts.update(d.label for d in dets)
        self._log_stats()

    def _publish(self, msg: Image, dets: list[Detection], labels: np.ndarray) -> None:
        h, w = labels.shape
        out = ObjectDetections(header=msg.header)
        out.label_image = Image(
            header=msg.header,
            height=h,
            width=w,
            encoding="mono16",
            is_bigendian=0,
            step=w * 2,
            data=labels.tobytes(),
        )
        for d in dets:
            out.objects.append(
                DetectedObject(
                    instance_id=instance_id(d.label, d.id),
                    perception_class=d.label,
                    score=float(d.score),
                    state=DetectedObject.STATE_VISIBLE,
                    box=[float(v) for v in d.box],
                    mask_id=d.id,
                )
            )
        self.pub_detections.publish(out)

    def _publish_overlay(
        self, msg: Image, rgb: np.ndarray, dets: list[Detection], labels: np.ndarray
    ) -> None:
        out = rgb.copy()
        for d in dets:
            m = labels == d.id
            out[m] = (0.5 * out[m] + 0.5 * self.colors[d.label]).astype(np.uint8)
        bgr = np.ascontiguousarray(out[:, :, ::-1])
        for d in dets:
            c = self.colors[d.label][::-1].tolist()
            x0, y0, x1, y1 = (int(v) for v in d.box)
            cv2.rectangle(bgr, (x0, y0), (x1, y1), c, 2)
            cv2.putText(
                bgr,
                f"{d.label} {d.score:.2f}",
                (x0, max(y0 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                c,
                2,
            )
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.pub_overlay.publish(
                CompressedImage(header=msg.header, format="jpeg", data=jpg.tobytes())
            )

    def _log_stats(self) -> None:
        elapsed = time.monotonic() - self.last_log
        if elapsed < 10.0 or not self.times:
            return
        t = np.array(self.times) * 1000
        per_frame = {k: round(v / len(t), 2) for k, v in sorted(self.counts.items())}
        self.get_logger().info(
            f"{len(t) / elapsed:.1f} Hz (camera {(self.frames_received - self.frames_logged) / elapsed:.1f} Hz), latency median {np.median(t):.1f} ms "
            f"p95 {np.percentile(t, 95):.1f} ms, instances per frame {per_frame}"
        )
        self.times, self.counts, self.last_log = [], Counter(), time.monotonic()
        self.frames_logged = self.frames_received


def run(make_backend: Callable[[Node, ClassMap], Backend], args=None) -> None:
    rclpy.init(args=args)
    node = DetectorNode(make_backend)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
