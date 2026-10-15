"""The shared class list (config/classes.yaml) and the per-frame clean-up every backend applies to it.

No ROS and no GPU here, so it is unit-tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass
class Detection:
    """One object in source-image pixels. Exactly one of mask and polygon is set."""

    label: str
    score: float
    box: np.ndarray  # [x_min, y_min, x_max, y_max]
    mask: np.ndarray | None = None  # bool, H x W
    polygon: np.ndarray | None = None  # N x 2, x then y
    id: int = 0


@dataclass
class ClassMap:
    labels: list[str]
    prompt_to_label: dict[str, str]
    coco_to_label: dict[str, str]
    min_scores: dict[str, float] = field(default_factory=dict)
    default_min_score: float = 0.25

    @classmethod
    def load(cls, path: str) -> ClassMap:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, doc: dict) -> ClassMap:
        default = float(doc.get("default_min_score", 0.25))
        labels, prompts, coco, mins = [], {}, {}, {}
        for entry in doc["classes"]:
            label = entry["label"]
            if label in labels:
                raise ValueError(f"class {label!r} is listed twice")
            labels.append(label)
            for p in entry.get("prompts") or [label]:
                if prompts.setdefault(p, label) != label:
                    raise ValueError(
                        f"prompt {p!r} maps to both {prompts[p]!r} and {label!r}"
                    )
            for c in entry.get("coco") or []:
                if coco.setdefault(c, label) != label:
                    raise ValueError(
                        f"COCO class {c!r} maps to both {coco[c]!r} and {label!r}"
                    )
            mins[label] = float(entry.get("min_score", default))
        return cls(labels, prompts, coco, mins, default)

    @property
    def prompts(self) -> list[str]:
        return list(self.prompt_to_label)

    @property
    def lowest_min_score(self) -> float:
        return min(self.min_scores.values(), default=self.default_min_score)

    def min_score(self, label: str) -> float:
        return self.min_scores.get(label, self.default_min_score)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def clean(
    dets: list[Detection], classes: ClassMap, iou: float = 0.7
) -> list[Detection]:
    """Drop detections below their class threshold, then merge duplicates of one label.

    Several prompts (or COCO classes) can map to one label and fire on the same object, e.g. "bowl" and
    "food container". Within a label, a detection whose box overlaps a higher-scoring one by IoU >= iou is
    dropped. Returns the survivors, highest score first, with ids 1..N.
    """
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if d.score < classes.min_score(d.label):
            continue
        if any(k.label == d.label and box_iou(k.box, d.box) >= iou for k in kept):
            continue
        kept.append(d)
    for i, d in enumerate(kept, start=1):
        d.id = i
    return kept
