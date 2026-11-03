"""Box geometry for OWL-ViT outputs, in numpy so it is testable without a GPU."""

from __future__ import annotations

import numpy as np


def to_source_xyxy(cxcywh: np.ndarray, height: int, width: int) -> np.ndarray:
    """OWL-ViT boxes (cx, cy, w, h as fractions of the square input) as [x0, y0, x1, y1] source pixels.

    The frame was scaled so its longer side fills the square and padded on the right and bottom, so one unit of
    the square is max(height, width) source pixels.
    """
    cx, cy, w, h = cxcywh.T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1) * max(
        height, width
    )
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
    return xyxy


def fit(height: int, width: int, size: int) -> tuple[int, int]:
    """The frame's size once scaled so its longer side is `size`."""
    s = size / max(height, width)
    return round(height * s), round(width * s)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy non-maximum suppression. Returns the indices kept, highest score first."""
    order = np.argsort(-scores, kind="stable")
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        x0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        union = area[i] + area[rest] - inter
        overlap = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[overlap < iou]
    return np.array(keep, int)


def nms_per_label(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou: float
) -> np.ndarray:
    """nms() within each label. Returns the indices kept, highest score first."""
    keep = [
        np.flatnonzero(labels == g)[nms(boxes[labels == g], scores[labels == g], iou)]
        for g in np.unique(labels)
    ]
    keep = np.concatenate(keep) if keep else np.array([], int)
    return keep[np.argsort(-scores[keep], kind="stable")]


def rectangle(box: np.ndarray) -> np.ndarray:
    """A box as a 4-point polygon (x, y), for the label image of a detector that gives boxes only."""
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
