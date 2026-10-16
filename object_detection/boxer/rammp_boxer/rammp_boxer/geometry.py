"""Frame and camera bookkeeping between ROS and BoxerNet, in plain numpy.

BoxerNet works in a gravity-aligned frame with +z up. The node builds that frame per image as the camera's rotation
from TF (gravity frame <- camera optical frame) with zero translation, so "world" below is the camera's own
position with the gravity frame's axes. Only rotation matters: BoxerNet ignores yaw and translation.
"""

from __future__ import annotations

import numpy as np


def matrix_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    """3x3 rotation matrix of a unit quaternion (ROS order x, y, z, w)."""
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion_from_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    """Unit quaternion (x, y, z, w) of a 3x3 rotation matrix, w >= 0."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
            0.25 * s,
        ]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[2, 1] - R[1, 2]) / s,
        ]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
            (R[0, 2] - R[2, 0]) / s,
        ]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
            (R[1, 0] - R[0, 1]) / s,
        ]
    q = np.array(q) / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return tuple(float(v) for v in q)


def scale_intrinsics(
    K: np.ndarray, width: int, height: int, hw: int
) -> tuple[float, float, float, float]:
    """fx, fy, cx, cy after stretching a width x height image to BoxerNet's square hw x hw input."""
    sx, sy = hw / width, hw / height
    return K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy


def boxes_to_boxer(xyxy: np.ndarray, width: int, height: int, hw: int) -> np.ndarray:
    """[x_min, y_min, x_max, y_max] source pixels -> BoxerNet's [x_min, x_max, y_min, y_max] in the hw x hw image."""
    b = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    return np.stack(
        [
            b[:, 0] * hw / width,
            b[:, 2] * hw / width,
            b[:, 1] * hw / height,
            b[:, 3] * hw / height,
        ],
        axis=1,
    )


def points_from_depth(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    R_wc: np.ndarray,
    num_samples: int = 10000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Up to num_samples points from a depth image, rotated into the gravity frame, NaN-padded to num_samples rows.

    The same grid-then-random subsampling Boxer's RGB-D loaders use (BaseLoader.sdp_from_depth), so the point
    density matches what the model was trained with. depth_m is in meters, 0 where invalid.
    """
    rng = rng or np.random.default_rng(0)
    h, w = depth_m.shape
    step = max(1, int(np.sqrt(h * w / (num_samples * 2))))
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    ys, xs = ys.ravel(), xs.ravel()
    z = depth_m[ys, xs]
    ok = np.isfinite(z) & (z > 0)
    ys, xs, z = ys[ok], xs[ok], z[ok]
    if len(z) > num_samples:
        keep = rng.choice(len(z), size=num_samples, replace=False)
        ys, xs, z = ys[keep], xs[keep], z[keep]
    pts = np.full((num_samples, 3), np.nan, dtype=np.float32)
    cam = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], axis=-1)
    pts[: len(z)] = cam @ R_wc.T
    return pts


def box_in_camera(
    center_w: np.ndarray, R_w_obj: np.ndarray, R_wc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A box's center and axes in the gravity frame -> in the camera optical frame (shared origin)."""
    R_cw = R_wc.T
    return R_cw @ center_w, R_cw @ R_w_obj
