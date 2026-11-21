"""odom3d: world-anchored 3D object tracking from 2D detections + aligned depth + odometry (no ROS, numpy/scipy only).

The demo objects (fridge, microwave, table, cups/containers) are static or nearly so. Instead of tracking boxes in
pixels (which breaks under camera rotation), each object is tracked in the odom frame and re-projected into every
camera frame from the current camera pose. The detector only confirms and corrects tracks, so:
  * ids are stable through rotations, missed detections, label flips and the object leaving view (tracks go
    dormant instead of being deleted, and a later detection revives the same id),
  * boxes are drawn at camera rate even when the detector runs slower, and detector latency is compensated,
  * label noise inside a class group (bowl<->cup, refrigerator<->tv) never changes an id.

Inputs per detector result (update) and per camera frame (render):
  t            stamp [s]
  T_odom_cam   4x4, camera optical frame -> odom (z up, z = 0 on the floor)
  depth_mm     HxW uint16 depth registered to color, millimetres (0 = invalid); may be None in render()
  dets         list of dict(kind, score, box[x0,y0,x1,y1], mask HxW bool or callable -> mask)
               kind: COCO name: refrigerator | tv | microwave | dining table | bowl | cup | wine glass
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.optimize import linear_sum_assignment

# kind -> class group. A label flip inside a group never changes a track's id.
GROUP = {
    "refrigerator": "fridge",
    "tv": "fridge",
    "microwave": "microwave",
    "dining table": "table",
    "bowl": "small",
    "cup": "small",
    "wine glass": "small",
}
LARGE = ("fridge", "microwave", "table")
# kind -> published label (rammp classes.yaml). 'tv' is only ever evidence for a fridge.
KIND_LABEL = {
    "refrigerator": "fridge",
    "tv": "fridge",
    "microwave": "microwave",
    "dining table": "table",
    "bowl": "food container",
    "cup": "drink cup",
    "wine glass": "drink cup",
}
GROUP_LABEL = {
    "fridge": "fridge",
    "microwave": "microwave",
    "table": "table",
}  # "small": by vote

DEFAULTS = dict(
    min_det_score=0.10,  # detections below this are ignored entirely
    # starting a track: a score above the class threshold, a plausible 3D shape, 3 hits over >= 0.3 s within
    # 1.5 s, and at least one confident hit (CONFIRM_SCORE). Keeping a track needs much less (ByteTrack idea).
    birth_score={
        "fridge": 0.25,
        "microwave": 0.25,
        "table": 0.15,
        "small": 0.25,
        "tv": 0.40,
    },
    confirm_score={"fridge": 0.5, "microwave": 0.5, "table": 0.3, "small": 0.35},
    birth_hits=3,
    birth_min_span_s=0.3,
    birth_window_s=1.5,
    gate_m={"fridge": 0.25, "microwave": 0.2, "table": 0.3, "small": 0.12},
    show_p=0.35,
    dormant_p=0.05,
    decay={
        "fridge": 0.93,
        "microwave": 0.93,
        "table": 0.95,
        "small": 0.85,
    },  # per missed-while-visible update
    q_pos={
        "fridge": 0.002,
        "microwave": 0.003,
        "table": 0.002,
        "small": 0.02,
    },  # m^2/s random walk
    billboard_alpha=0.5,
    depth_band={"fridge": 0.6, "microwave": 0.35, "table": 1.0, "small": 0.12},
    max_size={"fridge": 1.4, "microwave": 0.8, "table": 2.5, "small": 0.45},
    max_range_m=5.0,
    # orientation: a large object takes the heading of its own vertical faces once its histogram has
    # yaw_min_mass normals and a peak yaw_min_ratio x the mean; otherwise (and for small objects not standing on a
    # large one) the room's heading from all vertical surfaces; with neither, the odom axes.
    yaw_min_mass=400,
    yaw_min_ratio=2.0,
    yaw_decay=0.99,  # per matched measurement
    room_min_mass=3000,
    room_min_ratio=1.5,
    room_decay=0.998,  # per detector frame
    room_alpha=0.1,  # low-pass on the room heading; the last confident value holds while the peak is weak
    room_every=5,  # detector frames between room-heading updates (the room does not move)
    yaw_enough=3000,  # a large object's face histogram this full needs no new normals (they cost most of an update)
    pts_keep={
        "fridge": 30,
        "microwave": 30,
        "table": 30,
        "small": 1,
    },  # measurements' points kept per track
    pts_per_meas=300,
    # size of large objects: six face positions (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi in the box frame). Each is the
    # weighted outer quantile (face_quantile; the lo faces the mirror one) of its measurement history: partial masks
    # pull a face in and mask bleed pushes it out, both biased, so the face is what the good views agree on rather
    # than a mean. Weights: 1 / sigma^2 with sigma = a + b z^2 (close views count most), times the detection score,
    # fading with age (face_tau, follows odom drift). A face cut by the image border, hidden behind something nearer,
    # or turned away from the camera is only a bound (the object reaches at least there) and counts at half weight.
    face_sigma=(0.02, 0.012),
    face_quantile=(
        0.75,
        0.75,
        0.5,
    ),  # x, y, z faces: z by median (things on a table top bleed upward)
    face_tau=60.0,
    face_hist=150,
    face_back_cos=0.7,  # a face this far turned away from the camera is not seen
    # free space: when the latest view puts a side face carve_min inside the box and the camera sees through
    # the slab between (depth behind it) on carve_frames detector frames in a row, that slab is empty (a door that
    # closed, mask bleed) and the face restarts from that view
    carve_min=0.08,
    carve_frames=3,
    carve_through=0.6,  # fraction of the slab's in-view samples with depth clearly behind them
    min_depth={
        "fridge": 0.45,
        "microwave": 0.3,
        "table": 0.4,
    },  # for an axis whose far face was never seen
    floor_groups=("table",),  # boxes that reach the floor (z_lo = 0)
)


# ----------------------------------------------------------------------------- geometry helpers
def transform(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def containment(a, b):
    """Fraction of the smaller box inside the larger."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-9, min(sa, sb))


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def aabb_gap(alo, ahi, blo, bhi):
    """Euclidean gap between two axis-aligned boxes (0 if they overlap)."""
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(alo - bhi, blo - ahi))))


def aabb_iou2d(alo, ahi, blo, bhi):
    w = np.maximum(0.0, np.minimum(ahi[:2], bhi[:2]) - np.maximum(alo[:2], blo[:2]))
    inter = w[0] * w[1]
    a = np.prod(np.maximum(ahi[:2] - alo[:2], 1e-3))
    b = np.prod(np.maximum(bhi[:2] - blo[:2], 1e-3))
    return float(inter / (a + b - inter))


def _inside(p, lo, hi, m=0.05):
    return bool(np.all(p[:2] >= lo[:2] - m) and np.all(p[:2] <= hi[:2] + m))


def lift_points(mask, depth_mm, K, box=None, erode=2, step=2, zmax=5.0):
    """3D points (optical frame) of a mask from registered depth. Reads only the box crop if box is given."""
    x0 = y0 = 0
    if box is not None:
        H, W = mask.shape
        x0, y0 = max(int(box[0]) // step * step, 0), max(int(box[1]) // step * step, 0)
        x1, y1 = min(int(np.ceil(box[2])) + 1, W), min(int(np.ceil(box[3])) + 1, H)
        mask, depth_mm = mask[y0:y1, x0:x1], depth_mm[y0:y1, x0:x1]
    if erode > 0:
        # border_value=1: pixels past the crop do not erode the mask (what cv2.erode does)
        mask = binary_erosion(
            mask, np.ones((2 * erode + 1, 2 * erode + 1), bool), border_value=1
        )
    m = mask[::step, ::step]
    z = depth_mm[::step, ::step][m].astype(np.float64) / 1000.0
    vv, uu = np.nonzero(m)
    ok = (z > 0.15) & (z < zmax)
    if ok.sum() < 20:
        return None
    z, vv, uu = z[ok], vv[ok] * step + y0, uu[ok] * step + x0
    lo, hi = np.percentile(
        z, [5, 70]
    )  # near (front) surface; background leaking into the mask is far
    keep = (z >= lo) & (z <= max(hi, np.median(z)) + 0.25)
    z, vv, uu = z[keep], vv[keep], uu[keep]
    if len(z) < 20:
        return None
    return np.stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z], 1)


# ----------------------------------------------------------------------------- orientation
# Boxes are gravity-aligned and turn only about odom z. Kitchens are close to "Manhattan" (walls, cabinets and
# appliances square to each other), so a box's heading is only defined mod 90 deg. It is estimated from the
# directions of the vertical surfaces seen in depth: per object from its own faces, and for the whole room from
# the full image. Histograms hold the heading mod 90 in 1 deg bins.
YAW_BINS = 90


def vertical_face_yaws(
    depth_mm, K, R_oc, mask=None, box=None, step=2, s=3, zlim=(0.3, 4.0)
):
    """Headings mod 90 deg (degrees, odom frame) of the vertical surfaces in depth, from finite-difference normals
    (s px baseline) sampled every `step` px; restricted to `mask` (and the `box` crop) if given."""
    x0 = y0 = 0
    D = depth_mm
    if box is not None:
        H, W = depth_mm.shape
        x0, y0 = max(int(box[0]), 0), max(int(box[1]), 0)
        x1, y1 = min(int(np.ceil(box[2])) + 1, W), min(int(np.ceil(box[3])) + 1, H)
        D = depth_mm[y0:y1, x0:x1]
        if mask is not None:
            mask = mask[y0:y1, x0:x1]
    # sample every `step` px first; neighbours are k cells (~s px) away on that grid
    z = np.asarray(D[::step, ::step], dtype=np.float64) / 1000.0
    k = max(1, int(round(s / step)))
    if z.shape[0] <= 2 * k or z.shape[1] <= 2 * k:
        return np.zeros(0)
    v, u = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]
    P = np.stack(
        [
            (u * step + x0 - K[0, 2]) * z / K[0, 0],
            (v * step + y0 - K[1, 2]) * z / K[1, 1],
            z,
        ],
        -1,
    )
    dx = P[k:-k, 2 * k :] - P[k:-k, : -2 * k]
    dy = P[2 * k :, k:-k] - P[: -2 * k, k:-k]
    zc = z[k:-k, k:-k]
    ok = (zc > zlim[0]) & (zc < zlim[1])
    ok &= (z[k:-k, 2 * k :] > 0) & (z[k:-k, : -2 * k] > 0)
    ok &= (z[2 * k :, k:-k] > 0) & (z[: -2 * k, k:-k] > 0)
    # not across a depth edge
    ok &= (np.abs(dx[..., 2]) < 0.1 * zc) & (np.abs(dy[..., 2]) < 0.1 * zc)
    if mask is not None:
        ok &= np.asarray(mask)[::step, ::step][k:-k, k:-k]
    n = np.cross(dx[ok], dy[ok])
    nn = np.linalg.norm(n, axis=1)
    n = n[nn > 1e-9] / nn[nn > 1e-9, None]
    no = n @ np.asarray(R_oc).T
    no = no[np.abs(no[:, 2]) < 0.25]  # vertical surfaces only
    return np.degrees(np.arctan2(no[:, 1], no[:, 0])) % 90.0


def yaw_histogram(yaws_deg):
    idx = np.minimum(np.asarray(yaws_deg).astype(int), YAW_BINS - 1)
    return np.bincount(idx, minlength=YAW_BINS).astype(np.float64)


def yaw_peak(hist, min_mass, min_ratio):
    """Heading (rad, in [-45, 45) deg) at the peak of a mod-90 histogram, or None if it is weak."""
    if hist is None or hist.sum() < min_mass:
        return None
    w = 7
    sm = np.convolve(
        np.r_[hist[-(w // 2) :], hist, hist[: w // 2]], np.ones(w), "valid"
    )
    i = int(np.argmax(sm))
    if sm[i] < min_ratio * sm.mean():
        return None
    a, b, c = sm[i - 1], sm[i], sm[(i + 1) % YAW_BINS]
    den = a - 2 * b + c
    off = 0.5 * (a - c) / den if den < 0 else 0.0  # parabolic sub-bin peak
    deg = (i + 0.5 + off + 45.0) % 90.0 - 45.0
    return float(np.radians(deg))


def rot2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


# ----------------------------------------------------------------------------- tracker
class Track:
    __slots__ = (
        "id",
        "group",
        "c",
        "P",
        "size",
        "lo",
        "hi",
        "bb",
        "p_exist",
        "hits",
        "first_t",
        "last_hit_t",
        "hist",
        "confirmed",
        "score",
        "max_score",
        "label_shown",
        "yaw_hist",
        "pts",
        "anchor",
        "faces",
        "fhist",
        "fseen",
        "fcarve",
        "face_yaw",
        "face_t",
    )

    @property
    def dormant(self):
        return self.confirmed and self.p_exist <= 0.0


class Odom3DTracker:
    def __init__(self, K, width, height, params: dict | None = None):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.W, self.H = int(width), int(height)
        self.p = {
            k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()
        }
        for k, v in (params or {}).items():
            if isinstance(self.p.get(k), dict) and isinstance(v, dict):
                self.p[k].update(v)
            else:
                self.p[k] = v
        self.tracks: list[Track] = []
        self.next_id = 1
        self.room_hist = np.zeros(YAW_BINS)
        self._room_yaw = None
        self._n_updates = 0
        self._boxes = None  # boxes() cache, valid until the next update()

    # ------------------------------------------------------------ measurements
    def _measure(self, d, depth_mm, T_oc):
        g = GROUP.get(d["kind"])
        if g is None:
            return None
        mask = d.get("mask")
        if callable(mask):
            mask = mask()
        b = np.asarray(d["box"], dtype=np.float64)
        if mask is not None and g == "small" and _area(b) > 0:
            x0, y0 = max(int(b[0]), 0), max(int(b[1]), 0)
            if mask[y0 : int(b[3]) + 1, x0 : int(b[2]) + 1].sum() < 0.25 * _area(b):
                # mostly overwritten in a shared label image (e.g. a jar's pixels given to the table under it):
                # the central part of the box stands in for the mask
                mask = np.zeros(mask.shape, bool)
                w, h = b[2] - b[0], b[3] - b[1]
                mask[
                    int(b[1] + 0.25 * h) : int(b[3] - 0.25 * h) + 1,
                    int(b[0] + 0.25 * w) : int(b[2] - 0.25 * w) + 1,
                ] = True
        P = (
            lift_points(mask, depth_mm, self.K, box=b)
            if (mask is not None and depth_mm is not None)
            else None
        )
        e = 4
        border = dict(
            l=b[0] <= e, t=b[1] <= e, r=b[2] >= self.W - 1 - e, b=b[3] >= self.H - 1 - e
        )
        m = dict(
            group=g,
            box=b,
            score=float(d["score"]),
            kind=d["kind"],
            c=None,
            size=None,
            z=None,
            lo=None,
            hi=None,
            border=border,
            partial=any(border.values()),
        )
        if P is not None:
            # class-sized depth band around the front surface (glass doors / mask bleed)
            P = P[
                np.abs(P[:, 2] - np.percentile(P[:, 2], 30)) < self.p["depth_band"][g]
            ]
            if len(P) < 15:
                P = None
        if P is not None:
            Po = transform(T_oc, P)
            lo, hi = np.percentile(Po, [8, 92], axis=0)
            m.update(
                lo=lo,
                hi=hi,
                c=(lo + hi) / 2,
                size=np.maximum(hi - lo, 0.03),
                z=float(np.median(P[:, 2])),
                bottom=float(lo[2]),
                pc=P,
                po=Po,
                depth=depth_mm,
                pts=None
                if g in LARGE  # large objects use their faces, not kept points
                else Po[
                    np.linspace(0, len(Po) - 1, min(len(Po), self.p["pts_per_meas"]))
                    .round()
                    .astype(int)
                ],
                mask=mask,
                R_oc=T_oc[:3, :3],
            )
            if (
                np.max(m["size"]) > self.p["max_size"][g]
            ):  # gross lift failure -> 2D-only measurement
                m.update(c=None, size=None, lo=None, hi=None, pts=None)
        m["birth_ok"] = m["c"] is not None and self._plausible(m)
        if m["c"] is None and d["kind"] == "tv":
            return None  # a tv we cannot place in 3D is never evidence for a fridge
        return m

    def _plausible(self, m):
        """Shape prior for *starting* a track (odom z = 0 is the floor). Rejects COCO false positives on
        whiteboards / wall cabinets / monitors / boxes, and lets a 'tv' start a fridge only if it stands like one."""
        bottom, top, s = m["bottom"], m["bottom"] + m["size"][2], m["size"]
        if m["group"] == "fridge":
            floor_ok = (
                bottom < 0.6 or m["border"]["b"]
            )  # floor or a low base (the bench fridge is on ~0.4 m mats)
            tall = (s[2] > 0.45 and top < 2.1) or m["border"]["t"] or m["border"]["b"]
            wide = max(s[0], s[1]) > (0.3 if m["kind"] == "tv" else 0.2)
            return floor_ok and tall and wide and max(s[:2]) < 1.3 and m["z"] < 4.0
        if m["group"] == "table":
            return 0.35 < top < 1.25
        if m["group"] == "microwave":
            return (
                0.2 < bottom < 1.6 and max(s[:2]) < 0.9 and s[2] < 0.7 and m["z"] < 4.0
            )
        return -0.1 < bottom < 1.6 and max(s) < 0.5

    def _sigma2(self, m):
        z = m["z"] if m["z"] is not None else 2.0
        return (0.05 + 0.03 * z * z) ** 2

    # ------------------------------------------------------------ projection
    def _billboard(self, box, z, T_oc):
        """Box corners back-projected at depth z, in odom: a camera-facing card that re-projects as the camera moves."""
        K = self.K
        u = np.array([box[0], box[2], box[0], box[2]])
        v = np.array([box[1], box[1], box[3], box[3]])
        return transform(
            T_oc,
            np.stack(
                [
                    (u - K[0, 2]) * z / K[0, 0],
                    (v - K[1, 2]) * z / K[1, 1],
                    np.full(4, z),
                ],
                1,
            ),
        )

    def _project_bb(self, bb, T_co):
        P = transform(T_co, bb)
        if (P[:, 2] < 0.1).any():
            return None, None
        u = self.K[0, 0] * P[:, 0] / P[:, 2] + self.K[0, 2]
        v = self.K[1, 1] * P[:, 1] / P[:, 2] + self.K[1, 2]
        return np.array([u.min(), v.min(), u.max(), v.max()]), float(P[:, 2].mean())

    def _clip(self, b):
        return np.array(
            [max(b[0], 0), max(b[1], 0), min(b[2], self.W - 1), min(b[3], self.H - 1)]
        )

    def _visibility(self, tr, T_co, depth_mm):
        """(unclipped box, in_view, occluded, depth)."""
        box, z = self._project_bb(tr.bb, T_co)
        if box is None:
            return None, False, False, None
        cb = self._clip(box)
        area, carea = _area(box), _area(cb)
        frac = carea / area if area > 0 else 0
        in_view = (
            (frac > 0.5 or carea > 0.25 * self.W * self.H)
            and 0.2 < z < self.p["max_range_m"]
            and carea > 100
        )
        occluded = False
        if in_view and depth_mm is not None:
            x0, y0, x1, y1 = cb.astype(int)
            v = depth_mm[y0:y1:4, x0:x1:4]
            v = v[v > 0]
            if len(v) > 10:
                occluded = (v / 1000.0 < z - 0.35).mean() > 0.6
        return box, in_view, occluded, z

    # ------------------------------------------------------------ update
    def update(self, t, T_odom_cam, depth_mm, dets):
        p = self.p
        T_oc = np.asarray(T_odom_cam, dtype=np.float64)
        T_co = np.linalg.inv(T_oc)
        self._boxes = None
        self._n_updates += 1
        if depth_mm is not None and (
            self._room_yaw is None or self._n_updates % p["room_every"] == 0
        ):  # room heading from every vertical surface in view
            self.room_hist = self.p["room_decay"] * self.room_hist + yaw_histogram(
                vertical_face_yaws(depth_mm, self.K, T_oc[:3, :3], step=6, s=6)
            )
            y = yaw_peak(self.room_hist, p["room_min_mass"], p["room_min_ratio"])
            if y is not None:
                if self._room_yaw is None:
                    self._room_yaw = y
                else:  # step along the shorter way round mod 90 deg
                    d = (y - self._room_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
                    self._room_yaw = (
                        self._room_yaw + p["room_alpha"] * d + np.pi / 4
                    ) % (np.pi / 2) - np.pi / 4
        # 2D NMS per class group before the 3D lift (table top + whole table, microwave window + microwave)
        cand = sorted(
            (
                d
                for d in dets
                if d["score"] >= p["min_det_score"] and d["kind"] in GROUP
            ),
            key=lambda d: -d["score"],
        )
        kept = []
        for d in cand:
            g = GROUP[d["kind"]]
            if not any(
                GROUP[k["kind"]] == g
                and (
                    iou(d["box"], k["box"]) > 0.5
                    or (g in LARGE and containment(d["box"], k["box"]) > 0.8)
                )
                for k in kept
            ):
                kept.append(d)
        meas = [
            m for m in (self._measure(d, depth_mm, T_oc) for d in kept) if m is not None
        ]
        vis = {tr.id: self._visibility(tr, T_co, depth_mm) for tr in self.tracks}

        # association: Hungarian, gated per class group
        C = np.full((len(self.tracks), len(meas)), 1e6)
        for a, tr in enumerate(self.tracks):
            pbox = vis[tr.id][0]
            for b, m in enumerate(meas):
                if m["group"] != tr.group:
                    continue
                io = iou(self._clip(pbox), m["box"]) if pbox is not None else 0.0
                cost = None
                if m["c"] is not None:
                    gate = p["gate_m"][tr.group]
                    if (
                        tr.group in LARGE
                    ):  # gap to the accumulated extent: invariant to which face is seen
                        d3 = aabb_gap(m["lo"][:2], m["hi"][:2], tr.lo[:2], tr.hi[:2])
                    else:
                        gate += 0.3 * float(np.max(tr.size[:2]))
                        d3 = float(np.linalg.norm((m["c"] - tr.c)[:2]))
                    if d3 < gate and abs(m["c"][2] - tr.c[2]) < 0.6:
                        cost = 0.6 * d3 / gate + 0.4 * (1 - io)
                    elif io > 0.5:
                        cost = 0.9
                elif io > 0.3:
                    cost = 1 - io
                if cost is not None:
                    cost += (0.2 if m["kind"] == "tv" and not tr.confirmed else 0.0) + (
                        0.1 if tr.dormant else 0.0
                    )
                    C[a, b] = cost
        matched_t, matched_m = set(), set()
        if C.size:
            for a, b in zip(*linear_sum_assignment(C)):
                if C[a, b] < 1e5:
                    self._update_track(self.tracks[a], meas[b], t, T_oc, T_co)
                    matched_t.add(a)
                    matched_m.add(b)

        # misses count only when the track should have been seen (in view, not occluded)
        for a, tr in enumerate(self.tracks):
            if a in matched_t:
                continue
            if not tr.confirmed:
                if t - tr.first_t > p["birth_window_s"]:
                    tr.p_exist = 0.0
                continue
            _, in_view, occ, _ = vis[tr.id]
            if in_view and not occ:
                tr.p_exist *= p["decay"][tr.group]

        # births
        for b, m in enumerate(meas):
            if b in matched_m or not m["birth_ok"]:
                continue
            side_cut = (m["border"]["l"] or m["border"]["r"]) and m[
                "group"
            ] != "table"  # tables are usually cut
            if side_cut and _area(m["box"]) < 0.4 * self.W * self.H:
                continue  # truncated at the side: centroid is biased; wait for a full view
            thr = (
                p["birth_score"]["tv"]
                if m["kind"] == "tv"
                else p["birth_score"][m["group"]]
            )
            if m["score"] < thr or self._in_shadow(m, vis):
                continue
            self._birth(m, t, T_oc)

        # confirmed tracks are never forgotten: they go dormant (hidden) and a matching detection revives the id
        for tr in self.tracks:
            if tr.confirmed and tr.p_exist <= p["dormant_p"]:
                tr.p_exist = 0.0
        self.tracks = [
            tr for tr in self.tracks if tr.confirmed or tr.p_exist > p["dormant_p"]
        ]
        self._merge()

    def _in_shadow(self, m, vis):
        """A same-group box inside a live track's projection is that object again (e.g. depth through a glass door)."""
        for tr in self.tracks:
            v = vis.get(tr.id)
            if tr.group != m["group"] or tr.p_exist <= 0 or v is None or v[0] is None:
                continue
            pb = self._clip(v[0])
            if containment(m["box"], pb) > 0.7 and _area(m["box"]) <= 1.2 * _area(pb):
                return True
        return False

    def _birth(self, m, t, T_oc):
        tr = Track()
        tr.id = self.next_id
        self.next_id += 1
        tr.group = m["group"]
        tr.c, tr.P = m["c"].copy(), np.full(3, self._sigma2(m))
        tr.lo, tr.hi, tr.size = m["lo"].copy(), m["hi"].copy(), m["size"].copy()
        tr.bb = self._billboard(m["box"], m["z"], T_oc)
        tr.p_exist, tr.hits, tr.first_t, tr.last_hit_t = 0.5, 1, t, t
        tr.hist = {}
        tr.pts = [m["pts"]] if m["pts"] is not None else []
        tr.yaw_hist = np.zeros(YAW_BINS)
        if tr.group in LARGE:
            tr.yaw_hist += self._face_yaw_hist(m)
        tr.faces = None
        if tr.group in LARGE:
            self._init_faces(tr, m, t, T_oc)
        self._vote(tr, m)
        tr.confirmed, tr.score, tr.label_shown = False, m["score"], None
        tr.max_score = m["score"] if m["kind"] != "tv" else 0.0
        self.tracks.append(tr)

    def _vote(self, tr, m):
        lab = KIND_LABEL[m["kind"]]
        tr.hist[lab] = tr.hist.get(lab, 0.0) * 0.98 + m["score"] * (
            0.5 if m["kind"] == "tv" else 1.0
        )

    def _update_track(self, tr, m, t, T_oc, T_co):
        p = self.p
        if m["c"] is not None:
            P = tr.P + p["q_pos"][tr.group] * max(1e-3, t - tr.last_hit_t)
            R = self._sigma2(m) * (
                25.0 if m["partial"] else 1.0
            )  # truncated view -> biased centroid
            Kg = P / (P + R)
            tr.c = tr.c + Kg * (m["c"] - tr.c)
            tr.P = (1 - Kg) * P
            if not m["partial"]:
                tr.size = 0.8 * tr.size + 0.2 * m["size"]
            # accumulated extent: grow quickly toward new surface evidence, shrink slowly
            for arr, new, outward in (
                (tr.lo, m["lo"], m["lo"] < tr.lo),
                (tr.hi, m["hi"], m["hi"] > tr.hi),
            ):
                arr += np.where(outward, 0.1 if m["partial"] else 0.3, 0.03) * (
                    new - arr
                )
            if m["pts"] is not None:
                tr.pts = (tr.pts + [m["pts"]])[-p["pts_keep"][tr.group] :]
            if tr.group in LARGE:
                tr.yaw_hist *= p["yaw_decay"]
                if tr.yaw_hist.sum() < p["yaw_enough"]:
                    tr.yaw_hist += self._face_yaw_hist(m)
            if tr.faces is not None:
                self._update_faces(tr, m, t, T_oc)
            z = m["z"]
        else:
            z = float(transform(T_co, tr.c[None])[0, 2])
        # billboard: smoothed in odom; sides cut by the image border keep their previous (projected) extent
        new_box = m["box"].copy()
        old_box, _ = self._project_bb(tr.bb, T_co)
        if old_box is not None:
            e = 3
            if new_box[0] <= e:
                new_box[0] = min(new_box[0], old_box[0])  # noqa: E701
            if new_box[1] <= e:
                new_box[1] = min(new_box[1], old_box[1])  # noqa: E701
            if new_box[2] >= self.W - 1 - e:
                new_box[2] = max(new_box[2], old_box[2])  # noqa: E701
            if new_box[3] >= self.H - 1 - e:
                new_box[3] = max(new_box[3], old_box[3])  # noqa: E701
            a = p["billboard_alpha"]
            tr.bb = a * self._billboard(new_box, z, T_oc) + (1 - a) * self._billboard(
                old_box, z, T_oc
            )
        else:
            tr.bb = self._billboard(new_box, z, T_oc)
        tr.hits += 1
        tr.last_hit_t = t
        tr.score = 0.7 * tr.score + 0.3 * m["score"]
        tr.p_exist = min(1.0, tr.p_exist + 0.35 * m["score"] + 0.1)
        self._vote(tr, m)
        if m["kind"] != "tv":
            tr.max_score = max(tr.max_score, m["score"])
        need = p["confirm_score"][tr.group] - (
            0.1 if tr.group == "fridge" and "fridge" in tr.hist else 0.0
        )
        if (
            not tr.confirmed
            and tr.hits >= p["birth_hits"]
            and t - tr.first_t >= p["birth_min_span_s"]
            and tr.max_score >= need
        ):
            tr.confirmed = True
            tr.p_exist = max(tr.p_exist, 0.8)

    def _merge(self):
        conf = [t for t in self.tracks if t.confirmed]
        drop = set()
        for a in range(len(conf)):
            for b in range(a + 1, len(conf)):
                A, B = conf[a], conf[b]
                if A.group != B.group or A.id in drop or B.id in drop:
                    continue
                if A.group in LARGE:
                    same = aabb_iou2d(A.lo, A.hi, B.lo, B.hi) > 0.2 or (
                        aabb_gap(A.lo[:2], A.hi[:2], B.lo[:2], B.hi[:2]) == 0.0
                        and (_inside(A.c, B.lo, B.hi) or _inside(B.c, A.lo, A.hi))
                    )
                else:
                    same = np.linalg.norm((A.c - B.c)[:2]) < 0.5 * max(
                        np.max(A.size[:2]), np.max(B.size[:2]), 0.1
                    )
                if same and abs(A.c[2] - B.c[2]) < 0.5 * max(
                    A.size[2], B.size[2], 0.15
                ):
                    keep, rm = (
                        (A, B) if A.id < B.id else (B, A)
                    )  # the older id survives
                    for k, v in rm.hist.items():
                        keep.hist[k] = keep.hist.get(k, 0) + v
                    keep.p_exist = max(keep.p_exist, rm.p_exist)
                    keep.lo, keep.hi = (
                        np.minimum(keep.lo, rm.lo),
                        np.maximum(keep.hi, rm.hi),
                    )
                    keep.pts = (rm.pts + keep.pts)[-self.p["pts_keep"][keep.group] :]
                    keep.yaw_hist = keep.yaw_hist + rm.yaw_hist
                    drop.add(rm.id)
        for t_ in self.tracks:  # tentative tracks sitting on a confirmed one
            if t_.confirmed or t_.id in drop:
                continue
            if any(
                c.group == t_.group
                and c.id not in drop
                and np.linalg.norm((c.c - t_.c)[:2])
                < 0.5 * max(np.max(c.size[:2]), 0.1)
                for c in conf
            ):
                drop.add(t_.id)
        if drop:
            self.tracks = [t for t in self.tracks if t.id not in drop]

    def label(self, tr):
        if tr.group in GROUP_LABEL:
            return GROUP_LABEL[tr.group]
        best = max(tr.hist, key=tr.hist.get)
        if tr.label_shown is None or (
            best != tr.label_shown
            and tr.hist[best] > 1.5 * tr.hist.get(tr.label_shown, 0)
        ):
            tr.label_shown = best  # hysteresis: a new label must clearly win
        return tr.label_shown

    def _face_yaw_hist(self, m):
        if m.get("depth") is None or m.get("mask") is None:
            return np.zeros(YAW_BINS)
        return yaw_histogram(
            vertical_face_yaws(m["depth"], self.K, m["R_oc"], m["mask"], m["box"])
        )

    # ------------------------------------------------------------ box faces (large objects)
    def _track_yaw(self, tr):
        y = yaw_peak(tr.yaw_hist, self.p["yaw_min_mass"], self.p["yaw_min_ratio"])
        if y is None:
            y = self._room_yaw if self._room_yaw is not None else 0.0
        return y

    def _face_frame(self, tr, yaw):
        """Re-express the faces in heading yaw. Headings are mod 90 deg, so a step past 45 deg is a quarter turn of
        the box axes: turning the frame +90 deg makes new x = old y and new y = -old x."""
        k = int(np.round((yaw - tr.face_yaw) / (np.pi / 2))) % 4
        for _ in range(k):
            h = tr.fhist
            neg = lambda a: a * np.array([1.0, -1.0, 1.0])  # noqa: E731
            tr.fhist = [h[2], h[3], neg(h[1]), neg(h[0]), h[4], h[5]]
            tr.fseen = tr.fseen[[2, 3, 1, 0, 4, 5]]
            tr.fcarve = tr.fcarve[[2, 3, 1, 0, 4, 5]]
            f = tr.faces
            tr.faces = np.array([f[2], f[3], -f[1], -f[0], f[4], f[5]])
        tr.face_yaw = yaw

    def _face_meas(self, tr, m, T_oc, yaw):
        """Per face (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi; xy relative to the track's anchor in the box frame): the
        measured position, its variance, and whether it is only a bound: the object reaches at least there."""
        p = self.p
        Pc, Po = m["pc"], m["po"]
        if (
            len(Pc) > 500
        ):  # percentiles and edge tests need far fewer points than the lift gives
            i = np.linspace(0, len(Pc) - 1, 500).astype(int)
            Pc, Po = Pc[i], Po[i]
        R = rot2(yaw)
        loc = np.c_[(Po[:, :2] - tr.anchor) @ R, Po[:, 2]]
        lo, hi = np.percentile(loc, [4, 96], axis=0)
        # surface points spread evenly along x and y: the trimmed 4 % per side is the extent's ends, not noise
        ext = (hi[:2] - lo[:2]) * 0.04 / 0.92
        lo[:2], hi[:2] = lo[:2] - ext, hi[:2] + ext
        u = self.K[0, 0] * Pc[:, 0] / Pc[:, 2] + self.K[0, 2]
        v = self.K[1, 1] * Pc[:, 1] / Pc[:, 2] + self.K[1, 2]
        uc, vc = float(np.median(u)), float(np.median(v))
        cam = (T_oc[:2, 3] - tr.anchor) @ R
        z = m["z"]
        sig = (p["face_sigma"][0] + p["face_sigma"][1] * z * z) / np.sqrt(
            max(m["score"], 0.1)
        )
        D, e = m.get("depth"), 4
        val = np.zeros(6)
        var = np.full(6, sig * sig)
        # 0: seen; 1: turned away (only its far edge, from the near faces: a fair estimate that may come up short);
        # 2: cut by the image border, hidden, or too few points (the object surely reaches further: push-out only)
        bound = np.zeros(6, int)
        for j in range(6):
            ax, side = j // 2, (1 if j % 2 else -1)
            val[j] = hi[ax] if side > 0 else lo[ax]
            sel = (
                loc[:, ax] >= hi[ax] - 0.03 if side > 0 else loc[:, ax] <= lo[ax] + 0.03
            )
            if sel.sum() < 3:
                bound[j] = 2
                continue
            if (
                ax < 2
            ):  # a face turned away from the camera: only the near faces' far edge is seen
                n = np.zeros(2)
                n[ax] = side
                w = loc[sel, :2].mean(0) - cam
                if n @ w / (np.linalg.norm(w) + 1e-9) > p["face_back_cos"]:
                    bound[j] = 1
            uu, vv = u[sel], v[sel]
            if (
                uu.min() <= e
                or uu.max() >= self.W - 1 - e
                or vv.min() <= e
                or vv.max() >= self.H - 1 - e
            ):
                bound[j] = 2  # cut by the image border
                continue
            if (
                D is not None
            ):  # hidden: just outside the silhouette, depth is clearly nearer
                idx = np.linspace(0, sel.sum() - 1, min(15, int(sel.sum()))).astype(int)
                us, vs, zs = uu[idx], vv[idx], Pc[sel, 2][idx]
                d = np.stack([us - uc, vs - vc], 1)
                d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
                uo = np.clip((us + 8 * d[:, 0]).astype(int), 0, self.W - 1)
                vo = np.clip((vs + 8 * d[:, 1]).astype(int), 0, self.H - 1)
                zo = np.asarray(D)[vo, uo] / 1000.0
                if ((zo > 0.1) & (zo < zs - 0.15)).mean() > 0.4:
                    bound[j] = 2
        return val, var, bound

    def _init_faces(self, tr, m, t, T_oc):
        tr.anchor = m["c"][:2].copy()
        tr.face_yaw = self._track_yaw(tr)
        tr.faces = np.zeros(6)
        tr.fhist = [np.zeros((0, 3)) for _ in range(6)]  # rows: t, value, weight
        tr.fseen = np.zeros(6, bool)
        tr.fcarve = np.zeros(6, int)
        self._add_face_meas(tr, m, t, T_oc)

    def _update_faces(self, tr, m, t, T_oc):
        self._face_frame(tr, self._track_yaw(tr))
        val, bound = self._add_face_meas(tr, m, t, T_oc)
        if m.get("depth") is not None:
            self._carve(tr, val, bound, t, T_oc, m["depth"])

    def _slab_seen_through(self, tr, j, val, T_oc, depth_mm):
        """Is the slab of the box between side face j and this view's face (val[j]) seen empty: in view, and the
        depth image showing surfaces clearly behind it? A slab hidden behind the object's own front or anything else
        is not. Across the other axes it spans only what this view measured, so a box deeper than the object (a
        depth never seen) cannot make the floor behind the object count as seeing through it."""
        f = tr.faces
        ax, side = j // 2, (1 if j % 2 else -1)
        lo, hi = sorted((val[j] + side * 0.02, f[j]))
        o0, o1 = (2, 3) if ax == 0 else (0, 1)
        other = (max(f[o0], val[o0]), min(f[o1], val[o1]))
        if other[1] - other[0] < 0.03:
            return False
        g = np.linspace(0.1, 0.9, 5)
        a = lo + (hi - lo) * g
        b = other[0] + (other[1] - other[0]) * g
        # a table is mostly empty under its top (the camera sees between the legs): only its top band can be
        # seen through
        z1 = min(f[5], val[5])
        z0 = max(
            f[4], val[4], z1 - 0.08 if tr.group in self.p["floor_groups"] else -np.inf
        )
        if z1 - z0 < 0.02:
            return False
        zz = z0 + (z1 - z0) * g
        A, B, Z = np.meshgrid(a, b, zz, indexing="ij")
        local = np.stack([A, B] if ax == 0 else [B, A], -1).reshape(-1, 2)
        P = np.c_[tr.anchor + local @ rot2(tr.face_yaw).T, Z.reshape(-1)]
        Pc = transform(np.linalg.inv(T_oc), P)
        ok = Pc[:, 2] > 0.2
        u = np.full(len(Pc), -1)
        v = np.full(len(Pc), -1)
        u[ok] = np.round(self.K[0, 0] * Pc[ok, 0] / Pc[ok, 2] + self.K[0, 2]).astype(
            int
        )
        v[ok] = np.round(self.K[1, 1] * Pc[ok, 1] / Pc[ok, 2] + self.K[1, 2]).astype(
            int
        )
        ok &= (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        if ok.sum() < 0.5 * len(Pc):
            return False
        d = np.asarray(depth_mm)[v[ok], u[ok]] / 1000.0
        valid = d > 0.1
        if valid.sum() < 0.5 * ok.sum():
            return False
        through = d[valid] > Pc[ok, 2][valid] + 0.1
        return through.mean() >= self.p["carve_through"]

    def _carve(self, tr, val, bound, t, T_oc, depth_mm):
        p = self.p
        for j in range(4):
            side = 1 if j % 2 else -1
            # a bound face is checked too: if its slab is hidden (behind the object or something else) or out of
            # the image, it is never seen through, so the check itself decides what can be carved
            inside = (tr.faces[j] - val[j]) * side
            if inside < p["carve_min"]:
                tr.fcarve[j] = 0
                continue
            if not self._slab_seen_through(tr, j, val, T_oc, depth_mm):
                tr.fcarve[j] = 0
                continue
            tr.fcarve[j] += 1
            if tr.fcarve[j] >= p["carve_frames"]:
                # the face restarts from the view that saw the slab empty: dropping only the history beyond it
                # would hand the face to whatever partial views remain inside
                h = tr.fhist[j]
                tr.fhist[j] = np.array([[t, val[j], np.median(h[:, 2])]])
                tr.faces[j] = val[j]
                tr.fcarve[j] = 0

    def _add_face_meas(self, tr, m, t, T_oc):
        p = self.p
        val, var, bound = self._face_meas(tr, m, T_oc, tr.face_yaw)
        w0 = m["score"] / var
        for j in range(6):
            if j == 4 and tr.group in p["floor_groups"]:
                tr.faces[4] = 0.0
                continue
            side = 1 if j % 2 else -1
            if (
                bound[j] == 2
                and len(tr.fhist[j])
                and (val[j] - tr.faces[j]) * side <= 0
            ):
                continue  # a cut-off or hidden face inside the box says nothing
            if not bound[j]:
                tr.fseen[j] = True
            row = [[t, val[j], w0[j] * (0.5 if bound[j] else 1.0)]]
            tr.fhist[j] = np.concatenate([tr.fhist[j], row])[-p["face_hist"] :]
            tr.faces[j] = self._face_quantile(tr, j, t)
        return val, bound

    @staticmethod
    def _wquantile(v, w, q):
        o = np.argsort(v)
        cw = np.cumsum(w[o])
        return float(v[o][min(int(np.searchsorted(cw, q * cw[-1])), len(v) - 1)])

    def _face_quantile(self, tr, j, t):
        """Face j: the weighted outer quantile of its whole history (age-faded weights; a bound counts at half
        weight). Bounds inside the face count too, so a few stray masks cannot ratchet a face outward."""
        p, h = self.p, tr.fhist[j]
        v = h[:, 1]
        ww = h[:, 2] * np.exp(-(t - h[:, 0]) / p["face_tau"])
        q = p["face_quantile"][j // 2]
        return self._wquantile(v, ww, q if j % 2 else 1 - q)

    def _face_box(self, tr):
        """(centre in odom, size along the box axes, heading) from the faces; an axis whose far face was never seen
        is given at least the class's min_depth, on the unseen side."""
        f = tr.faces.copy()
        need = self.p["min_depth"][tr.group]
        unseen = ~tr.fseen
        for ax in (0, 1):
            a, b = 2 * ax, 2 * ax + 1
            if f[b] - f[a] < need:
                if unseen[b] and not unseen[a]:
                    f[b] = f[a] + need
                elif unseen[a] and not unseen[b]:
                    f[a] = f[b] - need
        mid = np.array([(f[0] + f[1]) / 2, (f[2] + f[3]) / 2])
        c = np.r_[tr.anchor + rot2(tr.face_yaw) @ mid, (f[4] + f[5]) / 2]
        size = np.maximum([f[1] - f[0], f[3] - f[2], f[5] - f[4]], 0.03)
        return c, size, tr.face_yaw

    # ------------------------------------------------------------ orientation
    def room_yaw(self):
        """Room heading in odom (rad, mod 90 deg), or None before the first confident estimate."""
        return self._room_yaw

    def _fit(self, tr, yaw):
        """(centre, size) of the box with heading yaw around the track's kept points (robust percentiles)."""
        P = np.concatenate(tr.pts)
        m = P[:, :2].mean(0)
        local = (P[:, :2] - m) @ rot2(yaw)  # rows: R(yaw)^T q
        lo, hi = np.percentile(local, [4, 96], axis=0)
        zlo, zhi = np.percentile(P[:, 2], [4, 96])
        c = np.r_[m + rot2(yaw) @ ((lo + hi) / 2), (zlo + zhi) / 2]
        return c, np.maximum(np.r_[hi - lo, zhi - zlo], 0.03)

    def boxes(self, tracks=None):
        """tid -> (centre in odom, size along the box axes, heading in odom [rad]) for confirmed tracks.
        Large objects: own faces, else the room. Small objects: the large object they stand on/in, else the room.
        Small objects keep the filtered centre (they can be carried); large ones are fit to their kept points."""
        if tracks is None and self._boxes is not None:
            return self._boxes
        cache = tracks is None
        tracks = [
            t for t in (tracks if tracks is not None else self.tracks) if t.confirmed
        ]
        room = self.room_yaw()
        room = 0.0 if room is None else room
        out = {}
        for tr in tracks:
            if tr.group not in LARGE:
                continue
            if tr.faces is not None:
                out[tr.id] = self._face_box(tr)
                continue
            yaw = yaw_peak(tr.yaw_hist, self.p["yaw_min_mass"], self.p["yaw_min_ratio"])
            yaw = room if yaw is None else yaw
            c, size = self._fit(tr, yaw) if tr.pts else (tr.c, tr.size)
            out[tr.id] = (c, size, yaw)
        for tr in tracks:
            if tr.group in LARGE:
                continue
            yaw = room
            bottom = tr.c[2] - tr.size[2] / 2
            best = None
            for sid, (sc, ss, sy) in out.items():
                q = np.abs(rot2(sy).T @ (tr.c[:2] - sc[:2]))
                top = sc[2] + ss[2] / 2
                if np.all(q < ss[:2] / 2 + 0.05) and bottom > top - 0.15 - ss[2]:
                    # on (or inside) it; the lowest-lying support wins over e.g. a fridge behind
                    if best is None or abs(bottom - top) < best[0]:
                        best = (abs(bottom - top), sy)
            if best is not None:
                yaw = best[1]
            size = self._fit(tr, yaw)[1] if tr.pts else tr.size
            out[tr.id] = (tr.c.copy(), size, yaw)
        if cache:
            self._boxes = out
        return out

    # ------------------------------------------------------------ output
    def render(self, t, T_odom_cam, depth_mm=None, include_out_of_view=False):
        """Objects to show in the camera frame at stamp t: list of dict(tid, label, box, score, state,
        pos_odom, size, yaw, pos_cam). state is "visible" (in view; box valid), "occluded" (in view behind something)
        or, with include_out_of_view, "out_of_view" (remembered, live, not in the image; box meaningless).
        The 3D box is centred at pos_odom, gravity-aligned and turned by yaw (rad) about odom z; size is along
        its own axes (x along yaw, y along yaw + 90 deg, z up)."""
        T_co = np.linalg.inv(np.asarray(T_odom_cam, dtype=np.float64))
        shown = [
            tr for tr in self.tracks if tr.confirmed and tr.p_exist >= self.p["show_p"]
        ]
        bx = self.boxes()
        out = []
        for tr in shown:
            box, in_view, occ, _ = self._visibility(tr, T_co, depth_mm)
            if box is None or not in_view:
                if not include_out_of_view:
                    continue
                box, state = np.zeros(4), "out_of_view"
            else:
                box, state = self._clip(box), "occluded" if occ else "visible"
            c, size, yaw = bx[tr.id]
            out.append(
                dict(
                    tid=tr.id,
                    label=self.label(tr),
                    box=box,
                    score=float(tr.score),
                    state=state,
                    pos_odom=np.asarray(c, dtype=np.float64).copy(),
                    pos_cam=transform(T_co, np.asarray(c)[None])[0],
                    size=np.asarray(size, dtype=np.float64).copy(),
                    yaw=float(yaw),
                )
            )
        return out
