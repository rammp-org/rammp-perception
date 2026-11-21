"""Stable object tracking for the UI: world-anchored 3D tracks from any detector + aligned depth + odometry.

Runs as /perception/odom3d between a detector and everything that reads /perception/object_detections. The
detector (e.g. the RF-DETR scene segmenter with tracking: false) publishes raw detections on detections_topic; this
node lifts each detected object into the odom frame with depth, keeps one track per physical object, and publishes
ObjectDetections on output_topic at *camera* rate (one message per camera_info frame), each tracked object's box
re-projected from the camera's current pose. Ids are stable through camera rotation, missed detections, label
flips and the object leaving view. See core.py for the algorithm.

Topics:
  detections_topic   (/perception/object_detections_raw)     in   ObjectDetections (masks in label_image if any)
  depth_topic        (/scene_camera/depth/image_rect_raw)    in   depth registered to color (16UC1 mm or 32FC1 m)
  camera_info_topic  (/scene_camera/color/camera_info)       in   color intrinsics; each message = one output frame
  output_topic       (/perception/object_detections)         out  tracked objects, one message per camera frame
  markers_topic      (/perception/odom3d/markers)            out  3D boxes in odom_frame for Foxglove/RViz

Output per object: instance_id "<label>_<track id>" (stable), perception_class, score (smoothed), state VISIBLE (box
= re-projected box) or OCCLUDED (in view behind something, or with publish_out_of_view also out of view; box all 0),
has_pose with pose/size of the track's gravity-aligned 3D box in the camera optical frame. mask_id is 0 and
label_image is empty (masks are not re-projected).

Pose source: TF odom_frame <- camera optical frame at each stamp. If the TF tree has no base->camera link (the
flatbot bags), set camera_extrinsic = [x, y, z, qx, qy, qz, qw] of the optical frame in base_frame (from
a hand-eye fit on bag data) and only odom_frame <- base_frame is looked up.

Detector classes: perception_class is mapped to the tracker's kinds with label_kinds (defaults below: the rammp
classes.yaml labels, plus "tv" which may only ever confirm a fridge-shaped object).
"""

from __future__ import annotations

import collections
import functools
import threading
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, Vector3
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from rammp_perception_interfaces.msg import DetectedObject, ObjectDetections

from .core import Odom3DTracker

LABEL_KINDS = {
    "fridge": "refrigerator",
    "microwave": "microwave",
    "table": "dining table",
    "food container": "bowl",
    "drink cup": "cup",
    "tv": "tv",
}


# A detection waits this long for its depth frame. Large depth frames can arrive hundreds of ms late over DDS; an
# update that arrives late is harmless here (tracks live in odom and rendering always uses the newest pose).
DEPTH_WAIT_S = 1.0


def stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def matrix_from_tf(tf) -> np.ndarray:
    t, q = tf.transform.translation, tf.transform.rotation
    return matrix_from_pose(t.x, t.y, t.z, q.x, q.y, q.z, q.w)


def matrix_from_pose(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]
    T[:3, 3] = (x, y, z)
    return T


def rot_z(a: float) -> np.ndarray:
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def quaternion_from_matrix(R) -> tuple[float, float, float, float]:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        return (
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
            0.25 * s,
        )
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    return tuple(q)


def depth_mm(msg: Image) -> np.ndarray:
    if msg.encoding in ("16UC1", "mono16"):
        return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[
            :, : msg.width
        ]
    if msg.encoding == "32FC1":
        a = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)[
            :, : msg.width
        ]
        return (
            (np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0)
            .clip(0, 65535)
            .astype(np.uint16)
        )
    raise ValueError(
        f"unsupported depth encoding {msg.encoding!r}; expected 16UC1 (mm) or 32FC1 (m)"
    )


def label_image(msg: Image) -> np.ndarray | None:
    if msg.height == 0 or msg.width == 0:
        return None
    return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[
        :, : msg.width
    ]


class StampCache:
    """The last few messages by stamp; nearest lookup within a slop."""

    def __init__(self, size=30):
        self.items: collections.OrderedDict[float, object] = collections.OrderedDict()
        self.size = size
        self.lock = threading.Lock()

    def put(self, t: float, msg) -> None:
        with self.lock:
            self.items[t] = msg
            while len(self.items) > self.size:
                self.items.popitem(last=False)

    def nearest(self, t: float, slop: float):
        with self.lock:
            if not self.items:
                return None
            k = min(self.items, key=lambda s: abs(s - t))
            return self.items[k] if abs(k - t) <= slop else None


class Odom3dNode(Node):
    def __init__(self):
        super().__init__("odom3d", namespace="perception")
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.detections_topic = p(
            "detections_topic", "/perception/object_detections_raw"
        )
        self.depth_topic = p("depth_topic", "/scene_camera/depth/image_rect_raw")
        self.info_topic = p("camera_info_topic", "/scene_camera/color/camera_info")
        output_topic = p("output_topic", "/perception/object_detections")
        markers_topic = p("markers_topic", "/perception/odom3d/markers")
        self.odom_frame = p("odom_frame", "odom")
        self.base_frame = p("base_frame", "base_link")
        extr = list(
            p("camera_extrinsic", [0.0])
        )  # [x y z qx qy qz qw] optical frame in base_frame; [0.0] = use TF
        self.T_base_cam = matrix_from_pose(*extr) if len(extr) == 7 else None
        self.render_stride = max(1, int(p("render_stride", 1)))
        self.publish_out_of_view = bool(p("publish_out_of_view", False))
        self.depth_slop = float(p("depth_slop_s", 0.02))
        self.tf_wait = float(p("tf_wait_s", 0.08))
        # sensor_data (best effort) matches any camera driver; reliable avoids losing large depth frames when the
        # publisher is reliable (e.g. bag playback on a loaded machine)
        sensor_qos = (
            qos_profile_sensor_data
            if p("sensor_qos", "sensor_data") == "sensor_data"
            else 5
        )
        kinds = p(
            "label_kinds", ""
        )  # YAML mapping label -> kind, overrides LABEL_KINDS
        self.label_kinds = dict(
            LABEL_KINDS, **(yaml.safe_load(kinds) or {} if kinds else {})
        )
        tracker_params = p("tracker_params", "")  # YAML mapping, see core.DEFAULTS
        self.tracker_params = yaml.safe_load(tracker_params) if tracker_params else {}

        self.tracker: Odom3DTracker | None = None
        self.lock = threading.Lock()
        self.depths = StampCache()
        self.pub = self.create_publisher(ObjectDetections, output_topic, 10)
        self.markers = self.create_publisher(MarkerArray, markers_topic, 2)
        self.tf = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf, self, spin_thread=True)

        # detections are processed on a worker thread (newest first); rendering runs on camera_info
        self.det_group = MutuallyExclusiveCallbackGroup()
        self.render_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            ObjectDetections,
            self.detections_topic,
            self._on_detections,
            10,
            callback_group=self.det_group,
        )
        self.create_subscription(
            Image,
            self.depth_topic,
            self._on_depth,
            sensor_qos,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.create_subscription(
            CameraInfo,
            self.info_topic,
            self._on_camera_info,
            sensor_qos,
            callback_group=self.render_group,
        )
        self.pending: collections.deque = collections.deque(maxlen=10)
        self.create_timer(0.005, self._render_pending, callback_group=self.render_group)
        self.queue: collections.deque = collections.deque(maxlen=20)
        self.ready = threading.Condition()
        self.stopping = threading.Event()
        self.frames = 0
        self.stats = dict(
            upd=[],
            ren=[],
            det_msgs=0,
            out_msgs=0,
            tf_miss=0,
            objs=0,
            no_depth=0,
            depth_lag=[],
            dropped=0,
        )
        self.last_log = time.monotonic()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
        how = "camera_extrinsic parameter" if self.T_base_cam is not None else "TF"
        self.get_logger().info(
            f"tracking {self.detections_topic} -> {output_topic} at camera rate ({self.info_topic}); "
            f"camera pose from {self.odom_frame} <- {self.base_frame} TF and {how}"
        )

    def _on_depth(self, m: Image) -> None:
        t = stamp_s(m.header)
        self.depths.put(t, m)
        self.stats["depth_lag"].append(self.get_clock().now().nanoseconds * 1e-9 - t)

    # ------------------------------------------------------------------ poses
    def _T_odom_cam(self, frame_id: str, stamp, wait: float):
        """4x4 camera -> odom at stamp; falls back to the newest TF when the stamp is not available yet."""
        target = self.base_frame if self.T_base_cam is not None else frame_id
        try:
            tf = self.tf.lookup_transform(
                self.odom_frame, target, Time.from_msg(stamp), Duration(seconds=wait)
            )
        except TransformException:
            try:
                tf = self.tf.lookup_transform(self.odom_frame, target, Time())
                self.stats["tf_miss"] += 1
            except TransformException as e:
                self.get_logger().warn(
                    f"no TF {self.odom_frame} <- {target}: {e}",
                    throttle_duration_sec=10.0,
                )
                return None
        T = matrix_from_tf(tf)
        return T @ self.T_base_cam if self.T_base_cam is not None else T

    # ------------------------------------------------------------------ detections
    def _on_detections(self, msg: ObjectDetections) -> None:
        with self.ready:
            if len(self.queue) == self.queue.maxlen:
                self.stats["dropped"] += 1
            self.queue.append((time.monotonic(), msg))
            self.ready.notify()

    def _loop(self) -> None:
        """Detections are applied in order; each waits (up to DEPTH_WAIT_S) for its depth frame."""
        while rclpy.ok() and not self.stopping.is_set():
            with self.ready:
                if not self.ready.wait_for(lambda: len(self.queue) > 0, timeout=0.5):
                    continue
                arrived, msg = self.queue[0]
            t = stamp_s(msg.header)
            if (
                self.depths.nearest(t, self.depth_slop) is None
                and time.monotonic() - arrived < DEPTH_WAIT_S
            ):
                time.sleep(0.005)
                continue
            with self.ready:
                self.queue.popleft()
            try:
                self._process(msg)
            except Exception as e:  # noqa: BLE001 -- a dead worker looks healthy from outside
                if not rclpy.ok():
                    return
                self.get_logger().error(
                    f"detection frame failed: {e!r}", throttle_duration_sec=5.0
                )

    def _process(self, msg: ObjectDetections) -> None:
        if self.tracker is None:
            return  # waiting for camera_info
        t0 = time.perf_counter()
        t = stamp_s(msg.header)
        T = self._T_odom_cam(msg.header.frame_id, msg.header.stamp, 0.2)
        if T is None:
            return
        dmsg = self.depths.nearest(t, self.depth_slop)  # _loop already waited for it
        if dmsg is None:
            self.stats["no_depth"] += 1
            with self.depths.lock:
                keys = list(self.depths.items)
            gap = min((abs(k - t) for k in keys), default=float("nan"))
            self.get_logger().debug(
                f"no depth for {t:.3f}: nearest {gap * 1e3:.0f} ms, cache "
                f"{len(keys)} [{keys[0] if keys else 0:.3f} .. {keys[-1] if keys else 0:.3f}]"
            )
            self.get_logger().warn(
                f"no depth within {self.depth_slop * 1e3:.0f} ms of a detection on "
                f"{self.depth_topic}; using 2D only",
                throttle_duration_sec=10.0,
            )
        D = depth_mm(dmsg) if dmsg is not None else None
        li = label_image(msg.label_image)
        dets = []
        for o in msg.objects:
            kind = self.label_kinds.get(o.perception_class)
            if (
                kind is None
                or o.state != DetectedObject.STATE_VISIBLE
                or not any(o.box)
            ):
                continue
            box = np.array(o.box, dtype=np.float64)
            if li is not None and o.mask_id:
                mask = functools.partial(np.equal, li, o.mask_id)
            else:  # box-only detector: the central part of the box stands in for the mask
                mask = functools.partial(self._box_mask, box)
            dets.append(dict(kind=kind, score=float(o.score), box=box, mask=mask))
        with self.lock:
            self.tracker.update(t, T, D, dets)
        self.stats["upd"].append(time.perf_counter() - t0)
        self.stats["det_msgs"] += 1

    def _box_mask(self, b):
        m = np.zeros((self.tracker.H, self.tracker.W), bool)
        w, h = b[2] - b[0], b[3] - b[1]
        m[
            int(b[1] + 0.2 * h) : int(b[3] - 0.2 * h) + 1,
            int(b[0] + 0.2 * w) : int(b[2] - 0.2 * w) + 1,
        ] = True
        return m

    # ------------------------------------------------------------------ rendering (camera rate)
    # No callback ever blocks: camera_info only queues the frame, and a fast timer renders each queued frame as soon
    # as TF covers its stamp (odometry usually lags the camera by a few tens of ms), or with the newest TF once the
    # frame has waited tf_wait_s.
    def _on_camera_info(self, info: CameraInfo) -> None:
        if self.tracker is None:
            self.tracker = Odom3DTracker(
                np.array(info.k).reshape(3, 3),
                info.width,
                info.height,
                self.tracker_params,
            )
            self.get_logger().info(
                f"camera {info.width}x{info.height}, frame {info.header.frame_id}"
            )
        self.frames += 1
        if (self.frames - 1) % self.render_stride:
            return
        self.pending.append((time.monotonic(), info))

    def _render_pending(self) -> None:
        target = None
        while self.pending:
            queued, info = self.pending[0]
            target = target or (
                self.base_frame if self.T_base_cam is not None else info.header.frame_id
            )
            ready = self.tf.can_transform(
                self.odom_frame, target, Time.from_msg(info.header.stamp)
            )
            if not ready and time.monotonic() - queued < self.tf_wait:
                return  # oldest frame still waiting for TF; keep order
            self.pending.popleft()
            self._render(info)

    def _render(self, info: CameraInfo) -> None:
        t0 = time.perf_counter()
        T = self._T_odom_cam(info.header.frame_id, info.header.stamp, 0.0)
        if T is None:
            return
        dmsg = self.depths.nearest(stamp_s(info.header), self.depth_slop)
        D = depth_mm(dmsg) if dmsg is not None else None
        with self.lock:
            objs = self.tracker.render(
                stamp_s(info.header), T, D, self.publish_out_of_view
            )
        R_co = np.linalg.inv(T)[:3, :3]
        out = ObjectDetections()
        out.header = info.header
        for o in objs:
            d = DetectedObject()
            d.instance_id = f"{o['label'].replace(' ', '_')}_{o['tid']}"
            d.perception_class = o["label"]
            d.score = float(o["score"])
            visible = o["state"] == "visible"
            d.state = (
                DetectedObject.STATE_VISIBLE
                if visible
                else DetectedObject.STATE_OCCLUDED
            )
            d.box = [float(v) for v in o["box"]] if visible else [0.0] * 4
            d.mask_id = 0
            d.has_pose = True
            d.pose.position = Point(
                x=float(o["pos_cam"][0]),
                y=float(o["pos_cam"][1]),
                z=float(o["pos_cam"][2]),
            )
            (
                d.pose.orientation.x,
                d.pose.orientation.y,
                d.pose.orientation.z,
                d.pose.orientation.w,
            ) = (
                float(v) for v in quaternion_from_matrix(R_co @ rot_z(o["yaw"]))
            )  # camera <- box: gravity-aligned, turned by the object's heading about odom z
            d.size = Vector3(
                x=float(o["size"][0]), y=float(o["size"][1]), z=float(o["size"][2])
            )
            out.objects.append(d)
        self.pub.publish(out)
        self.stats["ren"].append(time.perf_counter() - t0)
        self.stats["out_msgs"] += 1
        self.stats["objs"] += len(objs)
        if self.frames % 3 == 1:
            self._publish_markers(info.header.stamp)
        self._log_stats()

    def _publish_markers(self, stamp) -> None:
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        with self.lock:
            boxes = self.tracker.boxes()
            tracks = [
                (tr.id, self.tracker.label(tr), *boxes[tr.id], tr.dormant)
                for tr in self.tracker.tracks
                if tr.confirmed
            ]
        for tid, label, c, s, yaw, dormant in tracks:
            for kind in (Marker.CUBE, Marker.TEXT_VIEW_FACING):
                m = Marker()
                m.header.frame_id, m.header.stamp = self.odom_frame, stamp
                m.ns, m.id, m.type, m.action = (
                    "odom3d",
                    tid * 2 + (kind == Marker.TEXT_VIEW_FACING),
                    kind,
                    Marker.ADD,
                )
                m.pose.position = Point(
                    x=float(c[0]),
                    y=float(c[1]),
                    z=float(c[2] + (s[2] / 2 + 0.1 if kind != Marker.CUBE else 0.0)),
                )
                m.pose.orientation.z = float(np.sin(yaw / 2))
                m.pose.orientation.w = float(np.cos(yaw / 2))
                if kind == Marker.CUBE:
                    m.scale = Vector3(
                        x=float(max(s[0], 0.05)),
                        y=float(max(s[1], 0.05)),
                        z=float(max(s[2], 0.05)),
                    )
                    m.color.r, m.color.g, m.color.b, m.color.a = (
                        (0.5, 0.5, 0.5, 0.2) if dormant else (0.1, 0.8, 0.3, 0.4)
                    )
                else:
                    m.scale.z = 0.1
                    m.text = f"{label}_{tid}"
                    m.color.r = m.color.g = m.color.b = m.color.a = 1.0
                arr.markers.append(m)
        self.markers.publish(arr)

    def _log_stats(self) -> None:
        elapsed = time.monotonic() - self.last_log
        if elapsed < 10.0:
            return
        s = self.stats
        u = np.array(s["upd"]) * 1e3 if s["upd"] else np.zeros(1)
        r = np.array(s["ren"]) * 1e3 if s["ren"] else np.zeros(1)
        with self.lock:
            n_live = sum(
                1 for tr in self.tracker.tracks if tr.confirmed and not tr.dormant
            )
        self.get_logger().info(
            f"in {s['det_msgs'] / elapsed:.1f} Hz (update median {np.median(u):.1f} ms, p95 "
            f"{np.percentile(u, 95):.1f}), out {s['out_msgs'] / elapsed:.1f} Hz (render median {np.median(r):.2f} ms), "
            f"objects/frame {s['objs'] / max(1, s['out_msgs']):.1f}, live tracks {n_live}, "
            f"TF fallbacks {s['tf_miss']}, detections without depth {s['no_depth']}, depth arrival lag p50/max "
            f"{np.median(s['depth_lag'] or [0]) * 1e3:.0f}/{np.max(s['depth_lag'] or [0]) * 1e3:.0f} ms, "
            f"detections dropped {s['dropped']}"
        )
        self.stats = dict(
            upd=[],
            ren=[],
            det_msgs=0,
            out_msgs=0,
            tf_miss=0,
            objs=0,
            no_depth=0,
            depth_lag=[],
            dropped=0,
        )
        self.last_log = time.monotonic()

    def stop(self) -> None:
        self.stopping.set()
        self.worker.join(timeout=5.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Odom3dNode()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        ex.shutdown(timeout_sec=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
