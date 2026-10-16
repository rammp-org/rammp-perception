"""RF-DETR-Seg instance segmentation (COCO weights) on the GPU as an always-on detector node.

Parameters (besides rammp_detection_common's image_topic, frame_stride, classes_file, publish_overlay, mock,
tracking, track_buffer_frames):
  model   nano | small | medium      COCO-pretrained RF-DETR-Seg size; weights baked into the image
RF-DETR knows the 80 COCO classes only. Each class-list entry's `coco` names are reported as its label; a
label with no COCO name is never reported, and every other COCO class is dropped.
Tracking is Roboflow's recommended pairing for RF-DETR: ByteTrack from their `trackers` package.
"""

from __future__ import annotations

import numpy as np

from rammp_detection_common.classes import ClassMap, Detection, track_id_to_instance_id
from rammp_detection_common.node import run

MODELS = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
}


class RfdetrBackend:
    def __init__(self, node, classes: ClassMap):
        import rfdetr
        import torch
        from rfdetr.assets.coco_classes import COCO_CLASSES

        size = node.declare_parameter("model", "small").value
        if size not in MODELS:
            raise ValueError(f"model must be one of {sorted(MODELS)}, got {size!r}")
        self.classes = classes
        # COCO class id -> our label, for the COCO classes the class list asks for.
        self.id_to_label = {
            i: classes.coco_to_label[n]
            for i, n in COCO_CLASSES.items()
            if n in classes.coco_to_label
        }
        unknown = set(classes.coco_to_label) - set(COCO_CLASSES.values())
        if unknown:
            raise ValueError(f"not COCO class names: {sorted(unknown)}")
        missing = [
            label
            for label in classes.labels
            if label not in classes.coco_to_label.values()
        ]
        if missing:
            node.get_logger().warn(
                f"no COCO class for {missing}; RF-DETR will never report them"
            )

        self.model = getattr(rfdetr, MODELS[size])()
        self.model.inference(
            dtype=torch.float16
        )  # TorchScript trace in FP16, as on the bench
        self.model.predict(np.zeros((480, 640, 3), np.uint8), threshold=0.5)  # warm up

        self.tracker = None
        if node.tracking:
            from trackers import ByteTrackTracker

            low = classes.lowest_min_score
            # ByteTrack's defaults start tracks at 0.7, far above the class list's thresholds. It keeps a lost
            # track for lost_track_buffer * frame_rate / 30 frames: at frame_rate 30 that is track_buffer_frames.
            self.tracker = ByteTrackTracker(
                frame_rate=30.0,
                lost_track_buffer=node.track_buffer_frames,
                track_activation_threshold=low,
                high_conf_det_threshold=low,
            )
        tracking = "ByteTrack" if self.tracker else "no tracking"
        self.description = f"RF-DETR-Seg {size} (COCO), PyTorch FP16, {tracking}"

    def infer(self, rgb: np.ndarray) -> list[Detection]:
        d = self.model.predict(rgb, threshold=self.classes.lowest_min_score)
        keep = [i for i in range(len(d)) if int(d.class_id[i]) in self.id_to_label]
        d = d[np.array(keep, int)]
        if self.tracker is not None:
            d = self.tracker.update(d)
            d = d[d.tracker_id >= 0]  # -1: not yet confirmed as a track
        out = []
        for i in range(len(d)):
            out.append(
                Detection(
                    label=self.id_to_label[int(d.class_id[i])],
                    score=float(d.confidence[i]),
                    box=d.xyxy[i],
                    mask=d.mask[i].astype(bool) if d.mask is not None else None,
                    id=track_id_to_instance_id(d.tracker_id[i])
                    if self.tracker is not None
                    else 0,
                )
            )
        return out


def main(args=None) -> None:
    run(RfdetrBackend, args)
