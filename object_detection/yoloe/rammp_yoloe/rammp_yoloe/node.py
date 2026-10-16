"""YOLOE instance segmentation, on the GPU or a DLA core, as an always-on detector node.

Parameters (besides rammp_detection_common's image_topic, frame_stride, classes_file, publish_overlay, mock,
tracking, track_buffer_frames):
  model       yoloe-26s-seg | yoloe-26m-seg      26m has the better masks; 26s is the one that fits a DLA core
  device      gpu | dla0 | dla1                  dla runs most layers on a DLA core and frees the GPU
  imgsz       640
  model_dir   /models/yoloe                       weights and text encoder, baked into the image
  engine_dir  /cache/engines                      TensorRT engines; mount a volume so they survive restarts
  gmc_method  sparseOptFlow                       BoT-SORT camera-motion compensation (CPU): sparseOptFlow|orb|
                                                  sift|ecc|none; none saves CPU when the camera does not move
An engine missing from engine_dir is built at startup (a few minutes), then reused.
Tracking is Ultralytics' own: model.track() with its BoT-SORT, thresholds taken from the class list.
"""

from __future__ import annotations

import os

import numpy as np

from rammp_detection_common.classes import ClassMap, Detection, track_id_to_instance_id
from rammp_detection_common.node import run

from . import engine


def tracker_config(classes: ClassMap, buffer_frames: int, gmc_method: str) -> str:
    """Ultralytics' botsort.yaml with thresholds matching the class list. Returns the file path."""
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    cfg = YAML.load(check_yaml("botsort.yaml"))
    low = classes.lowest_min_score
    # A track starts at the lowest class threshold: Ultralytics' default (0.25) would never track a table (0.15).
    cfg.update(
        track_high_thresh=low,
        new_track_thresh=low,
        track_low_thresh=min(0.1, low / 2),  # must match YoloeBackend.track_low_thresh
        # Fusing multiplies each match's IoU by the detection score, so a weak object (the table at ~0.17) can
        # never clear match_thresh and is never tracked. Match on IoU alone; the class thresholds still apply.
        fuse_score=False,
        # Processed frames a lost track survives (Ultralytics' trackers assume 30 fps, which leaves it as is).
        track_buffer=buffer_frames,
        gmc_method=gmc_method,
    )
    path = "/tmp/rammp_yoloe_botsort.yaml"
    YAML.save(path, cfg)
    return path


class YoloeBackend:
    def __init__(self, node, classes: ClassMap):
        from ultralytics import YOLO

        p = lambda name, default: node.declare_parameter(name, default).value  # noqa: E731
        model = p("model", "yoloe-26s-seg")
        device = p("device", "gpu")
        self.imgsz = int(p("imgsz", 640))
        model_dir = p("model_dir", "/models/yoloe")
        engine_dir = p("engine_dir", "/cache/engines")
        gmc_method = p("gmc_method", "sparseOptFlow")

        self.classes = classes
        prompts = classes.prompts
        path = engine.engine_path(engine_dir, model, prompts, self.imgsz, device)
        if not os.path.exists(path):
            path = engine.build(
                model_dir,
                engine_dir,
                model,
                prompts,
                self.imgsz,
                device,
                log=node.get_logger().info,
            )
        self.model = YOLO(path, task="segment")
        names = list(self.model.names.values())
        if names != prompts:  # the hash in the file name should make this impossible
            raise RuntimeError(
                f"{path} was built for prompts {names}, expected {prompts}"
            )
        self.model.predict(
            np.zeros((480, 640, 3), np.uint8), imgsz=self.imgsz, verbose=False
        )  # warm up

        self.tracker = (
            tracker_config(classes, node.track_buffer_frames, gmc_method)
            if node.tracking
            else None
        )
        self.track_low_thresh = min(0.1, classes.lowest_min_score / 2)
        tracking = f"BoT-SORT ({gmc_method})" if self.tracker else "no tracking"
        self.description = f"{model} on {device}, {self.imgsz}px, {tracking}, engine {os.path.basename(path)}"

    def infer(self, rgb: np.ndarray) -> list[Detection]:
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        if self.tracker:
            # Boxes down to BoT-SORT's low threshold reach the tracker, so its second association pass can hold a
            # weak object (the table) through a bad frame; clean() still applies the class thresholds after.
            r = self.model.track(
                bgr,
                persist=True,
                tracker=self.tracker,
                imgsz=self.imgsz,
                conf=self.track_low_thresh,
                verbose=False,
            )[0]
            # A frame where the tracker holds nothing comes back as raw, untracked detections: drop them rather
            # than publish ids outside the tracker's.
            if r.boxes.id is None:
                return []
        else:
            r = self.model.predict(
                bgr, imgsz=self.imgsz, conf=self.classes.lowest_min_score, verbose=False
            )[0]
        if not len(r.boxes):
            return []
        polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
        ids = (
            r.boxes.id.int().cpu().numpy()
            if r.boxes.id is not None
            else [None] * len(r.boxes)
        )
        return [
            Detection(
                label=self.classes.prompt_to_label[r.names[int(c)]],
                score=float(s),
                box=b,
                polygon=poly,
                id=track_id_to_instance_id(t) if t is not None else 0,
            )
            for c, s, b, poly, t in zip(
                r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.xyxy.cpu().numpy(),
                polys,
                ids,
            )
        ]


def main(args=None) -> None:
    run(YoloeBackend, args)
