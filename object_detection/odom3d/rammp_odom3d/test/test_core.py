"""Synthetic scenes for the odom3d core: no ROS, no GPU. Run: python3 -m pytest object_detection/odom3d/rammp_odom3d."""

import numpy as np
import pytest

from rammp_odom3d.core import Odom3DTracker

W, H = 640, 400
K = np.array([[284.0, 0, 320], [0, 284.0, 200], [0, 0, 1]])
# optical (x right, y down, z fwd) -> level body (x fwd, y left, z up)
R_BODY_OPT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
CAM_H = 0.9


def T_odom_cam(yaw, x=0.0, y=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R_BODY_OPT
    T[:3, 3] = (x, y, CAM_H)
    return T


def render_scene(T_oc, boxes3d):
    """Depth image (mm) and per-object (2D box, mask) of axis-aligned boxes (lo, hi) in odom, by ray casting."""
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack(
        [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1
    )
    d_odom = rays @ T_oc[:3, :3].T
    o = T_oc[:3, 3]
    depth = np.full((H, W), np.inf)
    idx = np.full((H, W), -1)
    for n, (lo, hi) in enumerate(boxes3d):
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = (np.asarray(lo) - o) / d_odom
            t1 = (np.asarray(hi) - o) / d_odom
        tmin = np.nanmax(np.minimum(t0, t1), -1)
        tmax = np.nanmin(np.maximum(t0, t1), -1)
        hit = (tmax >= tmin) & (tmin > 0.1) & (tmin < depth)
        depth[hit] = tmin[hit]  # t along a ray with z_opt = 1 is the optical depth
        idx[hit] = n
    out = []
    for n in range(len(boxes3d)):
        m = idx == n
        if m.sum() < 50:
            out.append(None)
            continue
        ys, xs = np.nonzero(m)
        out.append((np.array([xs.min(), ys.min(), xs.max(), ys.max()], float), m))
    D = np.where(np.isfinite(depth), depth * 1000, 0).astype(np.uint16)
    return D, out


FRIDGE = (np.array([2.0, 0.3, 0.0]), np.array([2.5, 0.8, 0.9]))
MICROWAVE = (np.array([1.8, -0.6, 0.75]), np.array([2.1, -0.15, 1.05]))
TABLE = (np.array([1.6, -1.0, 0.0]), np.array([2.4, 0.1, 0.75]))


def sweep(n=150, amp=0.9):
    """Camera yawing +-amp rad (the bag motion), 15 Hz."""
    for k in range(n):
        yield k, k / 15.0, amp * np.sin(2 * np.pi * k / 90)


def run(scene, kinds, stride=3, drop=(), flip=None, score=0.7):
    tr = Odom3DTracker(K, W, H)
    shown = []
    for k, t, yaw in sweep():
        T = T_odom_cam(yaw)
        D, objs = render_scene(T, scene)
        if k % stride == 0 and k not in drop:
            dets = []
            for n, ob in enumerate(objs):
                if ob is None:
                    continue
                kind = flip(k, kinds[n]) if flip else kinds[n]
                dets.append(dict(kind=kind, score=score, box=ob[0], mask=ob[1]))
            tr.update(t, T, D, dets)
        shown.append((k, tr.render(t, T, D), objs))
    return tr, shown


def ids_by_label(shown):
    ids = {}
    for _, out, _ in shown:
        for o in out:
            ids.setdefault(o["label"], set()).add(o["tid"])
    return ids


def test_one_id_per_object_through_rotation():
    _, shown = run([FRIDGE, MICROWAVE], ["refrigerator", "microwave"])
    ids = ids_by_label(shown)
    assert {k: len(v) for k, v in ids.items()} == {"fridge": 1, "microwave": 1}


def test_boxes_follow_camera_between_detections():
    """At 5 Hz detections, the re-projected box on in-between frames stays on the object."""
    from rammp_odom3d.core import iou

    _, shown = run([MICROWAVE], ["microwave"], stride=3)
    ious = [
        iou(out[0]["box"], objs[0][0])
        for k, out, objs in shown
        if k % 3
        and out
        and objs[0] is not None
        and objs[0][0][0] > 5
        and objs[0][0][2] < W - 6
    ]
    assert len(ious) > 30
    assert np.median(ious) > 0.8


def test_id_survives_detection_gap_and_leaving_view():
    # no detections for 2 s in the middle; the object also leaves the view at the sweep extremes
    _, shown = run([MICROWAVE], ["microwave"], drop=set(range(45, 75)))
    assert len(ids_by_label(shown)["microwave"]) == 1


def test_label_flips_do_not_change_id():
    def flip(k, kind):
        return ("cup" if (k // 3) % 2 else "bowl") if kind == "bowl" else kind

    cup = (np.array([1.9, -0.1, 0.75]), np.array([1.98, -0.02, 0.85]))
    _, shown = run([TABLE, cup], ["dining table", "bowl"], flip=flip)
    ids = ids_by_label(shown)
    small = ids.get("food container", set()) | ids.get("drink cup", set())
    assert len(small) == 1 and len(ids["table"]) == 1


def test_tv_alone_needs_fridge_shape():
    wall_tv = (
        np.array([2.5, -0.4, 1.2]),
        np.array([2.55, 0.4, 1.7]),
    )  # high on a wall: never a fridge
    _, shown = run([wall_tv], ["tv"])
    assert "fridge" not in ids_by_label(shown)


def test_low_scores_never_start_tracks():
    _, shown = run([MICROWAVE], ["microwave"], score=0.2)
    assert ids_by_label(shown) == {}


@pytest.mark.parametrize("stride", [1, 3])
def test_ghost_behind_object_is_suppressed(stride):
    """A same-class detection whose depth lands behind a tracked object (glass door) is not a new object."""
    tr = Odom3DTracker(K, W, H)
    for k, t, yaw in sweep(90, amp=0.3):
        T = T_odom_cam(yaw)
        D, objs = render_scene(T, [MICROWAVE])
        if k % stride:
            continue
        dets = [dict(kind="microwave", score=0.8, box=objs[0][0], mask=objs[0][1])]
        if (
            k > 30
        ):  # window detection inside the microwave, with depth from far behind it
            b = objs[0][0]
            wb = np.array(
                [
                    b[0] + 0.2 * (b[2] - b[0]),
                    b[1] + 0.2 * (b[3] - b[1]),
                    b[2] - 0.3 * (b[2] - b[0]),
                    b[3] - 0.2 * (b[3] - b[1]),
                ]
            )
            m = np.zeros((H, W), bool)
            m[int(wb[1]) : int(wb[3]), int(wb[0]) : int(wb[2])] = True
            Dg = D.copy()
            Dg[m] = 3500
            dets = [dict(kind="microwave", score=0.6, box=wb, mask=m)] + (
                dets if k % 2 else []
            )
            tr.update(t, T, Dg, dets)
        else:
            tr.update(t, T, D, dets)
    confirmed = [x for x in tr.tracks if x.confirmed]
    assert len(confirmed) == 1


def test_boxes_take_the_heading_of_their_faces():
    """Scene square to a 'room' frame turned 25 deg from odom: the boxes come out turned 25 deg (mod 90) with their
    real widths, and a cup on the table takes the table's heading."""
    th = np.radians(-25.0)
    c, s = np.cos(th), np.sin(th)
    R_or = np.eye(4)
    R_or[:2, :2] = [[c, -s], [s, c]]  # room -> odom
    cup = (np.array([1.7, -0.1, 0.75]), np.array([1.78, -0.02, 0.85]))
    scene = [FRIDGE, MICROWAVE, TABLE, cup]
    kinds = ["refrigerator", "microwave", "dining table", "cup"]
    tr = Odom3DTracker(K, W, H)
    for k, t, yaw in sweep(150, amp=0.6):
        T_room = T_odom_cam(yaw)
        D, objs = render_scene(T_room, scene)
        if k % 3 == 0:
            dets = [
                dict(kind=kinds[n], score=0.8, box=ob[0], mask=ob[1])
                for n, ob in enumerate(objs)
                if ob is not None
            ]
            tr.update(t, R_or @ T_room, D, dets)
    bx = tr.boxes()
    by_label = {tr.label(x): bx[x.id] for x in tr.tracks if x.confirmed}
    assert {"fridge", "microwave", "table", "drink cup"} <= set(by_label)
    for lab, (_, size, y) in by_label.items():
        err = (np.degrees(y - th) + 45) % 90 - 45
        assert abs(err) < 3.0, (lab, np.degrees(y))
    # the fridge's front face (0.5 m along room y) is fully seen: its width along that axis is recovered
    _, size, y = by_label["fridge"]
    width = size[1] if abs((np.degrees(y - th) + 45) % 90 - 45) < 45 else size[0]
    assert abs(width - 0.5) < 0.08, size
