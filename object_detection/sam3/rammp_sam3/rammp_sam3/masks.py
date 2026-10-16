"""Mask bookkeeping that needs neither ROS nor a GPU: label images, boxes, and id matching."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_ID = 65535


@dataclass
class Detection:
    """One instance from a detector or tracker, in source-image pixels."""

    label: str
    score: float
    mask: np.ndarray  # bool, H x W
    id: int = 0
    visible: bool = True
    box: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


def mask_box(mask: np.ndarray) -> list[float]:
    """[x_min, y_min, x_max, y_max] of the mask's pixels; all zero for an empty mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def label_image(dets: list[Detection], height: int, width: int) -> np.ndarray:
    """uint16 image where each pixel is the id of the instance covering it, 0 for none.

    Instances are painted lowest score first, so where masks overlap the higher score wins.
    """
    out = np.zeros((height, width), np.uint16)
    for d in sorted(dets, key=lambda d: d.score):
        if d.visible and d.id > 0:
            out[d.mask] = d.id
    return out


def assign_ids(
    new: list[Detection],
    previous: list[Detection],
    next_id: int,
    iou_threshold: float = 0.3,
) -> int:
    """Give each new detection an id: the id of the previous instance with the same label it overlaps
    most (IoU >= iou_threshold, each previous id used once), else a fresh one. Returns the next free id."""
    pairs = []
    for i, n in enumerate(new):
        for j, p in enumerate(previous):
            if n.label == p.label and p.visible:
                iou = mask_iou(n.mask, p.mask)
                if iou >= iou_threshold:
                    pairs.append((iou, i, j))
    taken_new, taken_prev = set(), set()
    for _, i, j in sorted(pairs, reverse=True):
        if i not in taken_new and j not in taken_prev:
            new[i].id = previous[j].id
            taken_new.add(i)
            taken_prev.add(j)
    for i, n in enumerate(new):
        if i not in taken_new:
            n.id = next_id
            next_id = next_id % MAX_ID + 1  # wraps: ids index a mono16 label image
    return next_id
