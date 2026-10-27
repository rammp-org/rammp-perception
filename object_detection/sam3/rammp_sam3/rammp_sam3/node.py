"""ROS 2 front end for SAM3: one-shot expert detection plus tracking sessions.

Services (rammp_perception_interfaces):
  ~/detect            Detect          one detection on the next frame; ~0.4-0.8 s
  ~/start_tracking    StartTracking   detect, then track and publish until stopped (or the lease runs out, when
                                      lease_enabled is true)
  ~/renew_tracking    RenewTracking   extend the open session's lease; a no-op while leases are disabled
  ~/stop_tracking     StopTracking
Topics:
  ~/object_detections ObjectDetections every tracked frame (and each detection that seeds or re-seeds a session)
  ~/status            TrackingStatus  at status_rate_hz, always
  ~/overlay           sensor_msgs/Image (rgb8) masks drawn on the frame, only when publish_overlay is true
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image

from rammp_perception_interfaces.msg import (
    DetectedObject,
    ObjectDetections,
    TrackingStatus,
)
from rammp_perception_interfaces.srv import (
    Detect,
    RenewTracking,
    StartTracking,
    StopTracking,
)

from .masks import Detection, assign_ids, label_image

_CHANNELS = {
    "rgb8": (3, False),
    "bgr8": (3, True),
    "rgba8": (4, False),
    "bgra8": (4, True),
}


def image_to_rgb(msg: Image) -> np.ndarray:
    if msg.encoding not in _CHANNELS:
        raise ValueError(
            f"unsupported image encoding {msg.encoding!r}; expected one of {sorted(_CHANNELS)}"
        )
    ch, bgr = _CHANNELS[msg.encoding]
    a = (
        np.frombuffer(msg.data, np.uint8)
        .reshape(msg.height, msg.step)[:, : msg.width * ch]
        .reshape(msg.height, msg.width, ch)
    )
    a = a[:, :, :3]
    return np.ascontiguousarray(a[:, :, ::-1] if bgr else a)


class Frame:
    """A received image. Converted to RGB only when something uses it: the camera publishes far faster than the
    tracker consumes, and converting every frame in the callback competes with the tracker for the GIL."""

    __slots__ = ("msg", "received", "_rgb")

    def __init__(self, msg: Image, received: float):
        self.msg, self.received, self._rgb = msg, received, None

    @property
    def rgb(self) -> np.ndarray:
        if self._rgb is None:
            self._rgb = image_to_rgb(self.msg)
        return self._rgb


class FrameHub:
    """Latest frame per image topic. Subscribes to a topic the first time anyone asks for it."""

    def __init__(self, node: Node, group):
        self.node, self.group = node, group
        self.frames: dict[str, Frame] = {}
        self.subs = {}
        self.cv = threading.Condition()

    def _ensure(self, topic: str) -> None:
        with self.cv:
            if topic in self.subs:
                return
            self.subs[topic] = self.node.create_subscription(
                Image,
                topic,
                lambda m, t=topic: self._on_image(t, m),
                qos_profile_sensor_data,
                callback_group=self.group,
            )
            self.node.get_logger().info(f"subscribed to {topic}")

    def _on_image(self, topic: str, msg: Image) -> None:
        if msg.encoding not in _CHANNELS:
            self.node.get_logger().error(
                f"{topic}: unsupported image encoding {msg.encoding!r}; expected one of "
                f"{sorted(_CHANNELS)}",
                throttle_duration_sec=10.0,
            )
            return
        with self.cv:
            self.frames[topic] = Frame(msg, time.monotonic())
            self.cv.notify_all()

    def wait(self, topic: str, newer_than: float, timeout: float) -> Frame | None:
        """The first frame on `topic` received after `newer_than` (monotonic seconds), or None on timeout."""
        self._ensure(topic)
        deadline = time.monotonic() + timeout
        with self.cv:
            while True:
                f = self.frames.get(topic)
                if f is not None and f.received > newer_than:
                    return f
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self.cv.wait(left)


@dataclass
class Session:
    id: int
    prompts: list[str]
    topic: str
    threshold: float
    lease_s: float
    redetect_period_s: float
    max_objects: int
    deadline: float
    state: object = None  # engine.TrackState, or None while nothing is being tracked
    dets: list[Detection] = field(default_factory=list)
    next_id: int = 1
    last_detect: float = 0.0
    last_frame: float = 0.0
    last_step_start: float = 0.0
    step_times: list[float] = field(default_factory=list)


class Sam3Node(Node):
    def __init__(self):
        super().__init__("sam3", namespace="perception")
        p = self.declare_parameter
        self.model_path = p("model_path", "/models/sam3").value
        self.engine_dir = p("engine_dir", "/cache/engines").value
        self.use_tensorrt = p("use_tensorrt", True).value
        self.detector_res = p("detector_resolution", 1008).value
        self.tracker_res = p("tracker_resolution", 336).value
        self.default_topic = p(
            "default_image_topic", "/scene_camera/color/image_raw"
        ).value
        self.default_threshold = p("score_threshold", 0.5).value
        self.default_max_objects = p("max_objects", 10).value
        self.max_rate = p("max_tracking_rate_hz", 15.0).value
        self.frame_timeout = p("frame_timeout_s", 2.0).value
        status_rate = p("status_rate_hz", 2.0).value
        self.publish_overlay = p("publish_overlay", False).value
        # false: a session runs until StopTracking or a new StartTracking, and lease_s is ignored.
        self.lease_enabled = p("lease_enabled", False).value
        self.timing_log_period = p(
            "timing_log_period_s", 0.0
        ).value  # > 0: log per-stage tracking times
        self._timing = {}

        from .engine import (
            Sam3Engine,
        )  # imports torch/transformers: slow, so after parameters are declared

        self.engine = Sam3Engine(
            self.model_path,
            self.detector_res,
            self.tracker_res,
            self.engine_dir,
            self.use_tensorrt,
            log=self.get_logger().info,
        )
        t = time.monotonic()
        self.engine.warmup()
        self.get_logger().info(f"warm-up done in {time.monotonic() - t:.1f} s")

        self.gpu = threading.Lock()
        self.session_lock = threading.Lock()
        self.session: Session | None = None
        self.next_session_id = 1

        self.frames = FrameHub(self, ReentrantCallbackGroup())
        services = MutuallyExclusiveCallbackGroup()
        self.pub_detections = self.create_publisher(
            ObjectDetections, "~/object_detections", QoSProfile(depth=2)
        )
        self.pub_status = self.create_publisher(
            TrackingStatus, "~/status", QoSProfile(depth=2)
        )
        self.pub_overlay = (
            self.create_publisher(Image, "~/overlay", QoSProfile(depth=2))
            if self.publish_overlay
            else None
        )
        self.create_service(Detect, "~/detect", self.on_detect, callback_group=services)
        self.create_service(
            StartTracking, "~/start_tracking", self.on_start, callback_group=services
        )
        self.create_service(
            RenewTracking,
            "~/renew_tracking",
            self.on_renew,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.create_service(
            StopTracking,
            "~/stop_tracking",
            self.on_stop,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.create_timer(
            1.0 / status_rate,
            self.publish_status,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.frames._ensure(self.default_topic)
        self.stopping = threading.Event()
        self.worker = threading.Thread(target=self.track_loop, daemon=True)
        self.worker.start()
        self.get_logger().info("ready")

    # ---- messages ----------------------------------------------------------------------------------------
    @staticmethod
    def _object(d: Detection) -> DetectedObject:
        return DetectedObject(
            instance_id=f"{d.label.replace(' ', '_')}_{d.id}",
            perception_class=d.label,
            score=float(d.score),
            state=DetectedObject.STATE_VISIBLE
            if d.visible
            else DetectedObject.STATE_OCCLUDED,
            box=[float(v) for v in d.box],
            mask_id=d.id if d.visible else 0,
        )

    def _masks_msg(self, frame: Frame, dets: list[Detection]) -> ObjectDetections:
        h, w = frame.rgb.shape[:2]
        lab = label_image(dets, h, w)
        img = Image(
            header=frame.msg.header,
            height=h,
            width=w,
            encoding="mono16",
            is_bigendian=0,
            step=w * 2,
            data=lab.tobytes(),
        )
        return ObjectDetections(
            header=frame.msg.header,
            objects=[self._object(d) for d in dets],
            label_image=img,
        )

    def _publish(self, frame: Frame, dets: list[Detection]) -> ObjectDetections:
        msg = self._masks_msg(frame, dets)
        self.pub_detections.publish(msg)
        if self.pub_overlay is not None:
            self.pub_overlay.publish(self._overlay(frame, dets))
        return msg

    @staticmethod
    def _overlay(frame: Frame, dets: list[Detection]) -> Image:
        out = frame.rgb.copy()
        for d in sorted(dets, key=lambda d: d.score):
            if d.visible:
                color = np.array(
                    [
                        (d.id * 97) % 255,
                        (d.id * 57 + 80) % 255,
                        (d.id * 151 + 160) % 255,
                    ],
                    np.uint16,
                )
                out[d.mask] = ((out[d.mask].astype(np.uint16) + color) // 2).astype(
                    np.uint8
                )
        h, w = out.shape[:2]
        return Image(
            header=frame.msg.header,
            height=h,
            width=w,
            encoding="rgb8",
            is_bigendian=0,
            step=w * 3,
            data=out.tobytes(),
        )

    # ---- services ----------------------------------------------------------------------------------------
    def _frame_or_error(self, topic: str):
        frame = self.frames.wait(
            topic, newer_than=time.monotonic(), timeout=self.frame_timeout
        )
        if frame is None:
            return None, f"no image on {topic} within {self.frame_timeout:.1f} s"
        return frame, ""

    def on_detect(self, req, resp):
        if not req.prompts:
            resp.success, resp.message = False, "prompts must not be empty"
            return resp
        topic = req.image_topic or self.default_topic
        frame, err = self._frame_or_error(topic)
        if frame is None:
            resp.success, resp.message = False, err
            return resp
        t = time.monotonic()
        with self.gpu:
            dets = self.engine.detect(
                frame.rgb,
                list(req.prompts),
                req.score_threshold or self.default_threshold,
            )
        for i, d in enumerate(dets, start=1):
            d.id = i
        resp.success = True
        resp.message = f"{len(dets)} instances in {(time.monotonic() - t) * 1e3:.0f} ms"
        resp.detections = self._masks_msg(frame, dets)
        self.get_logger().info(f"detect {list(req.prompts)} on {topic}: {resp.message}")
        return resp

    def on_start(self, req, resp):
        if not req.prompts or (self.lease_enabled and req.lease_s <= 0):
            resp.accepted = False
            resp.message = (
                "prompts must not be empty"
                if not req.prompts
                else "lease_s must be > 0"
            )
            return resp
        topic = req.image_topic or self.default_topic
        # Wait for the frame first: a start that fails leaves the open session running.
        frame, err = self._frame_or_error(topic)
        if frame is None:
            resp.accepted, resp.message = False, err
            return resp
        with self.session_lock:
            old, self.session = (
                self.session,
                None,
            )  # one session at a time: a new start ends the old one
        if old is not None:
            self.get_logger().info(
                f"session {old.id} ended: replaced by a new StartTracking"
            )
        s = Session(
            id=self.next_session_id,
            prompts=list(req.prompts),
            topic=topic,
            threshold=req.score_threshold or self.default_threshold,
            lease_s=req.lease_s,
            redetect_period_s=req.redetect_period_s,
            max_objects=req.max_objects or self.default_max_objects,
            deadline=time.monotonic() + req.lease_s if self.lease_enabled else math.inf,
        )
        self.next_session_id += 1
        with self.gpu:
            self._redetect(s, frame)
        with self.session_lock:
            self.session = s
        resp.accepted, resp.session_id = True, s.id
        resp.message = (
            f"tracking {len(s.dets)} instances"
            if s.dets
            else "no instances found"
            + ("; will re-detect" if s.redetect_period_s > 0 else "; nothing to track")
        )
        resp.detections = self._publish(frame, s.dets)
        self.get_logger().info(f"session {s.id} {s.prompts} on {topic}: {resp.message}")
        return resp

    def on_renew(self, req, resp):
        with self.session_lock:
            s = self.session
            if s is None or s.id != req.session_id:
                resp.renewed, resp.message = (
                    False,
                    f"session {req.session_id} is not open",
                )
                return resp
            if not self.lease_enabled:
                resp.renewed, resp.message = (
                    True,
                    "leases are disabled; the session runs until stopped",
                )
                return resp
            s.deadline = time.monotonic() + (
                req.lease_s if req.lease_s > 0 else s.lease_s
            )
        resp.renewed, resp.message = True, ""
        return resp

    def on_stop(self, req, resp):
        with self.session_lock:
            s = self.session
            if s is None or (req.session_id and s.id != req.session_id):
                resp.stopped, resp.message = (
                    False,
                    f"session {req.session_id} is not open",
                )
                return resp
            self.session = None
        resp.stopped, resp.message = True, f"session {s.id} stopped"
        self.get_logger().info(resp.message)
        return resp

    # ---- tracking ----------------------------------------------------------------------------------------
    def _redetect(self, s: Session, frame: Frame) -> None:
        """Detect on `frame` and re-seed the tracker; ids carry over to instances that overlap their old mask."""
        dets = self.engine.detect(frame.rgb, s.prompts, s.threshold)[: s.max_objects]
        s.next_id = assign_ids(dets, s.dets, s.next_id)
        s.state = self.engine.start_track(frame.rgb, dets) if dets else None
        s.dets = dets
        s.last_detect = s.last_frame = frame.received

    def track_loop(self) -> None:
        while rclpy.ok() and not self.stopping.is_set():
            try:
                self._track_step()
            except Exception as e:  # noqa: BLE001 -- a dead worker would leave the session "active" forever
                with self.session_lock:
                    s, self.session = self.session, None
                self.get_logger().error(
                    f"tracking failed, session {s.id if s else '-'} ended: {e!r}"
                )
                time.sleep(0.5)

    def stop(self) -> None:
        """Stop tracking and wait for the step in flight, before the node is destroyed."""
        self.stopping.set()
        self.worker.join(timeout=10.0)

    def _track_step(self) -> None:
        """One tracking step: the next frame of the open session, if any."""
        with self.session_lock:
            s = self.session
            if s is not None and time.monotonic() > s.deadline:
                self.get_logger().info(f"session {s.id} ended: lease expired")
                self.session = s = None
        if s is None:
            time.sleep(0.05)
            return
        t0 = time.monotonic()
        # Rate cap: steps are due 1/max_rate apart, so a step that already takes longer than that never
        # sleeps. The cap is waited out before taking a frame, so the step never tracks a frame that sat out
        # the wait (at 5 Hz that added ~120 ms of camera-to-result latency).
        period = 1.0 / self.max_rate
        due_at = s.last_step_start + period if s.last_step_start else 0.0
        if due_at > t0:
            time.sleep(due_at - t0)
        t1 = time.monotonic()
        # On time, take the first frame received after the step was due, not the buffered one, which has aged
        # up to a whole camera period (~35 ms less camera-to-result latency at 15 fps). A late step (the last
        # one overran) takes the newest frame.
        on_time = t1 - due_at < 0.01
        frame = self.frames.wait(
            s.topic,
            newer_than=max(s.last_frame, due_at) if on_time else s.last_frame,
            timeout=0.2,
        )
        if frame is None:
            return
        t1s = time.monotonic()
        # On time, the next step is due one period after this one was, so waiting for the frame doesn't push
        # the rate under the cap. Late, the period counts from now, so a stall is not caught up in a burst.
        s.last_step_start = due_at if on_time else t1s
        due = (
            s.redetect_period_s > 0
            and time.monotonic() - s.last_detect >= s.redetect_period_s
        )
        if s.state is None and not due:
            s.last_frame = frame.received
            return
        with self.gpu:
            if self.session is not s:  # stopped or replaced while we waited for the GPU
                return
            if due:
                self._redetect(s, frame)
                dets = s.dets
            else:
                dets = self.engine.track(s.state, frame.rgb)
                s.dets = (
                    dets  # latest masks: what the next re-detection matches ids against
                )
                s.last_frame = frame.received
        if (
            self.session is not s
        ):  # stopped while this frame was on the GPU: do not publish it
            return
        t2 = time.monotonic()
        self._publish(frame, dets)
        t3 = time.monotonic()
        s.step_times = (s.step_times + [t3])[-30:]
        if self.timing_log_period > 0:
            self._log_timing(
                {
                    "rate_cap": t1 - t0,
                    "wait_frame": t1s - t1,
                    "gpu": t2 - t1s,
                    "publish": t3 - t2,
                    "age": t3 - frame.received,
                }
            )

    def _log_timing(self, sample: dict) -> None:
        for k, v in sample.items():
            self._timing.setdefault(k, []).append(v)
        start = self._timing.setdefault("_start", [time.monotonic()])[0]
        if time.monotonic() - start >= self.timing_log_period:
            n = len(self._timing["gpu"])
            parts = ", ".join(
                f"{k} {1e3 * sum(v) / len(v):.0f} ms"
                for k, v in self._timing.items()
                if k != "_start"
            )
            self.get_logger().info(
                f"tracking {n / (time.monotonic() - start):.1f} Hz: {parts}"
            )
            self._timing = {}

    def publish_status(self) -> None:
        with self.session_lock:
            s = self.session
        msg = TrackingStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        if s is not None:
            msg.active, msg.session_id, msg.prompts, msg.image_topic = (
                True,
                s.id,
                s.prompts,
                s.topic,
            )
            msg.lease_remaining_s = max(
                0.0, s.deadline - time.monotonic()
            )  # inf while leases are disabled
            st = [t for t in s.step_times if t > time.monotonic() - 2.0]
            msg.rate_hz = (
                float((len(st) - 1) / (st[-1] - st[0]))
                if len(st) > 1 and st[-1] > st[0]
                else 0.0
            )
            msg.objects = [self._object(d) for d in s.dets]
        self.pub_status.publish(msg)


def main():
    rclpy.init()
    node = Sam3Node()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
