"""3D boxes for any detector's objects: BoxerNet lifts each 2D box to a gravity-aligned 3D box.

Runs as /perception/boxer, separate from the detectors, so it works with whichever one is running and never slows
it down. For every ObjectDetections message it finds the color frame it came from (same stamp), the registered depth
frame nearest in time, the color camera's intrinsics, and the camera's rotation in a gravity-aligned frame from TF at
that stamp. It republishes the same message with each visible object's pose (3D box center and axes, in the
message's header.frame_id) and size filled in, has_pose true where BoxerNet's confidence is at least min_prob.
Everything else, ids and label_image included, passes through unchanged.

Topics:
  detections_topic   (/perception/object_detections)    in   ObjectDetections from any detector
  image_topic        (/scene_camera/color/image_raw)     in   the color frames those detections came from
  depth_topic        (/scene_camera/depth/image_raw)     in   depth registered to color (Orbbec: depth_registration)
  camera_info_topic  (/scene_camera/color/camera_info)   in   the color camera's intrinsics
  output_topic       (/perception/object_detections_3d)  out  the same objects with pose and size

The four input topics are parameters and can be changed at run time to move to another detector or camera:
  ros2 param set /perception/boxer detections_topic /perception/sam3/object_detections

The frame published to is gravity_frame (base_link): any frame with +z up. Only its rotation is used.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3
from rcl_interfaces.msg import SetParametersResult
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from rammp_detection_common.node import image_to_rgb
from rammp_perception_interfaces.msg import DetectedObject, ObjectDetections

from .geometry import matrix_from_quaternion, quaternion_from_matrix
from .sync import FrameCache

TOPICS = {
    "detections_topic": "/perception/object_detections",
    "image_topic": "/scene_camera/color/image_raw",
    "depth_topic": "/scene_camera/depth/image_raw",
    "camera_info_topic": "/scene_camera/color/camera_info",
}
COLOR_WAIT_S = 0.2  # a detection can arrive before its color frame reaches this node; wait this long for it


def stamp_ns(header) -> int:
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def depth_to_meters(msg: Image) -> np.ndarray:
    if msg.encoding in ("16UC1", "mono16"):
        a = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[
            :, : msg.width
        ]
        return a.astype(np.float32) / 1000.0
    if msg.encoding == "32FC1":
        a = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)[
            :, : msg.width
        ]
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    raise ValueError(
        f"unsupported depth encoding {msg.encoding!r}; expected 16UC1 (mm) or 32FC1 (m)"
    )


class BoxerNode(Node):
    def __init__(self):
        super().__init__("boxer", namespace="perception")
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.topics = {name: p(name, default) for name, default in TOPICS.items()}
        output_topic = p("output_topic", "/perception/object_detections_3d")
        self.gravity_frame = p("gravity_frame", "base_link")
        self.min_prob = float(p("min_prob", 0.5))
        self.depth_slop_ns = int(float(p("depth_slop_s", 0.02)) * 1e9)
        engine_dir = p("engine_dir", "/cache/engines")

        from .lifter import Lifter  # heavy: torch, TensorRT, Boxer

        self.lifter = Lifter(engine_dir, log=self.get_logger().info)
        self.pub = self.create_publisher(ObjectDetections, output_topic, 10)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.colors, self.depths = FrameCache(), FrameCache()
        self.camera_info: CameraInfo | None = None
        self.subs = {}
        for name in TOPICS:
            self._subscribe(name)
        self.add_on_set_parameters_callback(self._on_parameters)

        self.latest: ObjectDetections | None = None
        self.ready = threading.Condition()
        self.stopping = threading.Event()
        self.times, self.lifted, self.last_log = [], 0, time.monotonic()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
        self.get_logger().info(
            f"lifting {self.topics['detections_topic']} -> {output_topic}, gravity from TF frame {self.gravity_frame}"
        )

    # Subscriptions, re-made whenever a topic parameter changes.

    def _subscribe(self, name: str) -> None:
        if name in self.subs:
            self.destroy_subscription(self.subs.pop(name))
        topic = self.topics[name]
        if name == "detections_topic":
            self.subs[name] = self.create_subscription(
                ObjectDetections, topic, self._on_detections, 10
            )
        elif name == "image_topic":
            self.colors.clear()
            self.subs[name] = self.create_subscription(
                Image,
                topic,
                lambda m: self.colors.put(stamp_ns(m.header), m),
                qos_profile_sensor_data,
            )
        elif name == "depth_topic":
            self.depths.clear()
            self.subs[name] = self.create_subscription(
                Image,
                topic,
                lambda m: self.depths.put(stamp_ns(m.header), m),
                qos_profile_sensor_data,
            )
        else:
            self.camera_info = None
            self.subs[name] = self.create_subscription(
                CameraInfo,
                topic,
                lambda m: setattr(self, "camera_info", m),
                qos_profile_sensor_data,
            )

    def _on_parameters(self, params) -> SetParametersResult:
        for prm in params:
            if prm.name in TOPICS:
                self.topics[prm.name] = prm.value
                self._subscribe(prm.name)
                self.get_logger().info(f"{prm.name} -> {prm.value}")
            elif prm.name == "gravity_frame":
                self.gravity_frame = prm.value
            elif prm.name == "min_prob":
                self.min_prob = float(prm.value)
            elif prm.name == "depth_slop_s":
                self.depth_slop_ns = int(float(prm.value) * 1e9)
        return SetParametersResult(successful=True)

    # Lifting, on a worker thread: always the newest detections, so a slow frame never builds a queue.

    def _on_detections(self, msg: ObjectDetections) -> None:
        with self.ready:
            self.latest = msg
            self.ready.notify()

    def _loop(self) -> None:
        while rclpy.ok() and not self.stopping.is_set():
            with self.ready:
                if not self.ready.wait_for(
                    lambda: self.latest is not None, timeout=0.5
                ):
                    continue
                msg, self.latest = self.latest, None
            try:
                self._process(msg)
            except Exception as e:  # noqa: BLE001 -- keep publishing; a dead worker looks healthy from outside
                if not rclpy.ok():  # shutting down mid-frame: the context is gone, so nothing can be published
                    return
                self.get_logger().error(
                    f"frame failed, publishing it without 3D boxes: {e!r}",
                    throttle_duration_sec=5.0,
                )
                self.pub.publish(msg)

    def stop(self) -> None:
        self.stopping.set()
        self.worker.join(timeout=10.0)

    def _inputs(self, msg: ObjectDetections):
        """rgb, depth (m), K and R_wc for msg's frame, or None with the reason logged."""
        t = stamp_ns(msg.header)
        deadline = time.monotonic() + COLOR_WAIT_S
        color = self.colors.nearest(t, 0)
        while color is None and time.monotonic() < deadline:
            time.sleep(0.01)
            color = self.colors.nearest(t, 0)
        depth = self.depths.nearest(t, self.depth_slop_ns)
        info = self.camera_info
        missing = [
            n
            for n, v in (
                ("color frame", color),
                ("depth frame", depth),
                ("camera_info", info),
            )
            if v is None
        ]
        if missing:
            self.get_logger().warn(
                f"no {', '.join(missing)} for this detection stamp",
                throttle_duration_sec=10.0,
            )
            return None
        rgb, depth_m = image_to_rgb(color), depth_to_meters(depth)
        if depth_m.shape != rgb.shape[:2] or (info.width, info.height) != (
            rgb.shape[1],
            rgb.shape[0],
        ):
            self.get_logger().warn(
                f"color {rgb.shape[1]}x{rgb.shape[0]}, depth {depth_m.shape[1]}x{depth_m.shape[0]}, camera_info "
                f"{info.width}x{info.height} differ: depth must be registered to color",
                throttle_duration_sec=10.0,
            )
            return None
        try:
            tf = self.tf.lookup_transform(
                self.gravity_frame,
                msg.header.frame_id,
                Time.from_msg(msg.header.stamp),
                Duration(seconds=0.1),
            )
        except TransformException as e:
            self.get_logger().warn(
                f"no TF {self.gravity_frame} <- {msg.header.frame_id}: {e}",
                throttle_duration_sec=10.0,
            )
            return None
        q = tf.transform.rotation
        return (
            rgb,
            depth_m,
            np.array(info.k).reshape(3, 3),
            matrix_from_quaternion(q.x, q.y, q.z, q.w),
        )

    def _process(self, msg: ObjectDetections) -> None:
        t0 = time.perf_counter()
        visible = [
            o
            for o in msg.objects
            if o.state == DetectedObject.STATE_VISIBLE and any(o.box)
        ]
        inputs = self._inputs(msg) if visible else None
        if inputs is not None:
            boxes = self.lifter.lift(*inputs, np.array([o.box for o in visible]))
            for o, b in zip(visible, boxes):
                if b.prob < self.min_prob:
                    continue
                o.has_pose = True
                o.pose.position.x, o.pose.position.y, o.pose.position.z = (
                    float(v) for v in b.center
                )
                qx, qy, qz, qw = quaternion_from_matrix(b.rotation)
                (
                    o.pose.orientation.x,
                    o.pose.orientation.y,
                    o.pose.orientation.z,
                    o.pose.orientation.w,
                ) = qx, qy, qz, qw
                o.size = Vector3(
                    x=float(b.size[0]), y=float(b.size[1]), z=float(b.size[2])
                )
                self.lifted += 1
        self.pub.publish(msg)
        self.times.append(time.perf_counter() - t0)
        self._log_stats()

    def _log_stats(self) -> None:
        elapsed = time.monotonic() - self.last_log
        if elapsed < 10.0 or not self.times:
            return
        t = np.array(self.times) * 1000
        self.get_logger().info(
            f"{len(t) / elapsed:.1f} Hz, latency median {np.median(t):.1f} ms p95 {np.percentile(t, 95):.1f} ms, "
            f"3D boxes per frame {self.lifted / len(t):.1f}"
        )
        self.times, self.lifted, self.last_log = [], 0, time.monotonic()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BoxerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()
