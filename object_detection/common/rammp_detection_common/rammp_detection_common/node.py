"""The always-on detector node every backend in this repo runs inside.

A backend only turns an RGB frame into Detections with labels from object_detection/config/classes.yaml. This node does the
rest the same way for all of them, so they are interchangeable behind one sheppy node:

Topics (relative to the node name, default /scene_segmenter):
  ~/instances           rammp_perception_interfaces/InstanceMasks   every processed frame; mono16 label image
  ~/detections          vision_msgs/Detection2DArray                the same instances as boxes;
                                                                    Detection2D.id = Instance.id
  ~/overlay/compressed  sensor_msgs/CompressedImage                 masks drawn on the frame, only while
                                                                    subscribed and publish_overlay is true
Ids are per frame (1..N, highest score first): this node detects, it does not track.
Headers copy the camera frame's stamp and frame_id.
"""

from __future__ import annotations

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
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from rammp_perception_interfaces.msg import Instance, InstanceMasks

from .classes import ClassMap, Detection, clean

_CHANNELS = {
    "rgb8": (3, False),
    "bgr8": (3, True),
    "rgba8": (4, False),
    "bgra8": (4, True),
}
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
        super().__init__(node_name)
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.image_topic = p("image_topic", "/scene_camera/color/image_raw")
        self.rate_hz = float(p("rate_hz", 10.0))
        self.publish_overlay = bool(p("publish_overlay", True))
        classes_file = p("classes_file", "/etc/rammp/perception/classes.yaml")
        mock = bool(p("mock", False))

        self.classes = ClassMap.load(classes_file)
        self.colors = {
            label: _PALETTE[i % len(_PALETTE)]
            for i, label in enumerate(self.classes.labels)
        }
        self.get_logger().info(f"classes from {classes_file}: {self.classes.labels}")

        self.pub_instances = self.create_publisher(InstanceMasks, "~/instances", 10)
        self.pub_detections = self.create_publisher(
            Detection2DArray, "~/detections", 10
        )
        self.pub_overlay = self.create_publisher(
            CompressedImage, "~/overlay/compressed", 10
        )

        if mock:
            # Hardware-free stand-in: empty results on the real topics at the real rate, no model, no camera.
            self.backend = None
            self.create_timer(1.0 / self.rate_hz, self._tick_mock)
            self.get_logger().info(
                f"MOCK: publishing empty results at {self.rate_hz} Hz"
            )
            return

        self.backend = make_backend(self, self.classes)
        self.get_logger().info(f"backend ready: {self.backend.description}")

        self.latest: Image | None = None
        self.last_stamp = None
        self.create_subscription(
            Image, self.image_topic, self._on_image, qos_profile_sensor_data
        )
        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.times: list[float] = []
        self.counts: Counter = Counter()
        self.last_log = time.monotonic()
        self.get_logger().info(
            f"detecting on {self.image_topic} at up to {self.rate_hz} Hz"
        )

    def _on_image(self, msg: Image) -> None:
        self.latest = msg

    def _tick_mock(self) -> None:
        header = Image().header
        header.stamp = self.get_clock().now().to_msg()
        self.pub_instances.publish(
            InstanceMasks(
                header=header, label_image=Image(header=header, encoding="mono16")
            )
        )
        self.pub_detections.publish(Detection2DArray(header=header))

    def _tick(self) -> None:
        msg = self.latest
        if msg is None or msg.header.stamp == self.last_stamp:
            return
        self.last_stamp = msg.header.stamp
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
        out = InstanceMasks(header=msg.header)
        out.label_image = Image(
            header=msg.header,
            height=h,
            width=w,
            encoding="mono16",
            is_bigendian=0,
            step=w * 2,
            data=labels.tobytes(),
        )
        arr = Detection2DArray(header=msg.header)
        for d in dets:
            box = [float(v) for v in d.box]
            out.instances.append(
                Instance(
                    id=d.id,
                    label=d.label,
                    score=float(d.score),
                    state=Instance.STATE_VISIBLE,
                    box=box,
                )
            )
            det = Detection2D(header=msg.header, id=str(d.id))
            det.bbox.center.position.x = (box[0] + box[2]) / 2
            det.bbox.center.position.y = (box[1] + box[3]) / 2
            det.bbox.size_x, det.bbox.size_y = box[2] - box[0], box[3] - box[1]
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = d.label, float(d.score)
            det.results.append(hyp)
            arr.detections.append(det)
        self.pub_instances.publish(out)
        self.pub_detections.publish(arr)

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
            f"{len(t) / elapsed:.1f} Hz, latency median {np.median(t):.1f} ms "
            f"p95 {np.percentile(t, 95):.1f} ms, instances per frame {per_frame}"
        )
        self.times, self.counts, self.last_log = [], Counter(), time.monotonic()


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
