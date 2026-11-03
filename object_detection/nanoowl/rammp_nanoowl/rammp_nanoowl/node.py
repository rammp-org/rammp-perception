"""NanoOWL (OWL-ViT, TensorRT FP16) open-vocabulary detection on the GPU as an always-on detector node.

Parameters (besides rammp_detection_common's image_topic, frame_stride, classes_file, publish_overlay, mock,
tracking, track_buffer_frames):
  model       owlvit-b32 | owlvit-b16       b16 sees smaller objects and costs about 4x the GPU time
  nms_iou     0.3                           boxes of one label overlapping a higher-scoring one this much are dropped
  model_dir   /models/owlvit                weights, baked into the image
  engine_dir  /cache/engines                TensorRT engines; mount a volume so they survive restarts
Boxes only: each object's label-image region is its box. Prompts are the class list's `prompts`, thresholds its
`owl_min_score` (OWL-ViT scores run far lower than YOLOE's). An engine missing from engine_dir is built at startup
(about 2 minutes), then reused.
Tracking is ByteTrack from Roboflow's `trackers`, as for RF-DETR.
"""

from __future__ import annotations

import numpy as np
import torch

from rammp_detection_common.classes import ClassMap, Detection, track_id_to_instance_id
from rammp_detection_common.node import run

from . import owl
from .boxes import nms_per_label, rectangle, to_source_xyxy


class OwlBackend:
    def __init__(self, node, classes: ClassMap):
        p = lambda name, default: node.declare_parameter(name, default).value  # noqa: E731
        model = p("model", "owlvit-b32")
        self.nms_iou = float(p("nms_iou", 0.3))
        model_dir = p("model_dir", "/models/owlvit")
        engine_dir = p("engine_dir", "/cache/engines")

        # The node's clean() applies the class list's min_score after this backend: make it the OWL thresholds.
        classes.min_scores = dict(classes.owl_min_scores)
        classes.default_min_score = classes.default_owl_min_score
        self.labels = classes.labels
        groups = [
            [pr for pr, label in classes.prompt_to_label.items() if label == g]
            for g in self.labels
        ]
        self.detector = owl.OwlDetector(
            model_dir, engine_dir, model, groups, log=node.get_logger().info
        )
        self.thresholds = torch.tensor(
            [classes.min_score(g) for g in self.labels], device="cuda"
        )
        self.detector(np.zeros((720, 1280, 3), np.uint8), self.thresholds)  # warm up

        self.tracker = None
        if node.tracking:
            from trackers import ByteTrackTracker

            low = classes.lowest_min_score
            # As for RF-DETR: start tracks at the class list's thresholds, not ByteTrack's 0.7.
            self.tracker = ByteTrackTracker(
                frame_rate=30.0,
                lost_track_buffer=node.track_buffer_frames,
                track_activation_threshold=low,
                high_conf_det_threshold=low,
            )
        tracking = "ByteTrack" if self.tracker else "no tracking"
        self.description = f"NanoOWL {model}, TensorRT FP16 {owl.SIZE}px, {tracking}, engine {self.detector.engine_file}"

    def infer(self, rgb: np.ndarray) -> list[Detection]:
        boxes, scores, groups = self.detector(rgb, self.thresholds)
        boxes = to_source_xyxy(boxes, rgb.shape[0], rgb.shape[1])
        keep = nms_per_label(boxes, scores, groups, self.nms_iou)
        boxes, scores, groups = boxes[keep], scores[keep], groups[keep]
        ids = np.zeros(len(keep), int)
        if self.tracker is not None:
            import supervision as sv

            d = self.tracker.update(
                sv.Detections(xyxy=boxes, confidence=scores, class_id=groups)
            )
            d = d[d.tracker_id >= 0]  # -1: not yet confirmed as a track
            boxes, scores, groups = d.xyxy, d.confidence, d.class_id
            ids = [track_id_to_instance_id(t) for t in d.tracker_id]
        return [
            Detection(
                label=self.labels[int(g)],
                score=float(s),
                box=b,
                polygon=rectangle(b),
                id=int(i),
            )
            for b, s, g, i in zip(boxes, scores, groups, ids)
        ]


def main(args=None) -> None:
    run(OwlBackend, args)
