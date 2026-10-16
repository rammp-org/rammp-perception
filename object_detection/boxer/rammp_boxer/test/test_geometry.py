import numpy as np
import pytest

from rammp_boxer.geometry import (
    box_in_camera,
    boxes_to_boxer,
    matrix_from_quaternion,
    points_from_depth,
    quaternion_from_matrix,
    scale_intrinsics,
)

# Gravity frame <- camera optical frame for a level camera: optical z (forward) is gravity-frame +y, optical y
# (down) is gravity-frame -z. Boxer's loaders use this same matrix when they have no tilt.
LEVEL = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])


def rot_x(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def test_quaternion_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = matrix_from_quaternion(*q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(
            matrix_from_quaternion(*quaternion_from_matrix(R)), R, atol=1e-9
        )


def test_quaternion_of_identity():
    assert quaternion_from_matrix(np.eye(3)) == pytest.approx((0, 0, 0, 1))


def test_scale_intrinsics_stretches_to_square():
    K = np.array([[612.7, 0, 635.4], [0, 612.7, 361.0], [0, 0, 1]])
    fx, fy, cx, cy = scale_intrinsics(K, 1280, 720, 960)
    assert fx == pytest.approx(612.7 * 0.75)
    assert fy == pytest.approx(612.7 * 960 / 720)
    assert (cx, cy) == pytest.approx((635.4 * 0.75, 361.0 * 960 / 720))


def test_boxes_to_boxer_reorders_and_scales():
    out = boxes_to_boxer([[128, 72, 640, 360]], 1280, 720, 960)
    np.testing.assert_allclose(out, [[96, 480, 96, 480]])
    assert boxes_to_boxer(np.zeros((0, 4)), 1280, 720, 960).shape == (0, 4)


def test_points_from_level_camera_lie_ahead():
    depth = np.full((96, 96), 2.0, np.float32)
    pts = points_from_depth(depth, 100, 100, 48, 48, LEVEL, num_samples=500)
    assert pts.shape == (500, 3)
    valid = pts[~np.isnan(pts).any(1)]
    assert len(valid) == 500
    np.testing.assert_allclose(valid[:, 1], 2.0)  # forward is gravity-frame +y
    # rows above and below the principal point
    assert valid[:, 2].min() < 0 < valid[:, 2].max()


def test_points_skip_invalid_depth_and_pad():
    depth = np.zeros((40, 40), np.float32)
    depth[:10] = 1.0
    depth[10:12] = np.nan
    pts = points_from_depth(depth, 50, 50, 20, 20, LEVEL, num_samples=10000)
    valid = ~np.isnan(pts).any(1)
    assert 0 < valid.sum() <= 400
    assert np.isnan(pts[valid.sum() :]).all()


def test_box_in_camera_level():
    center_c, R_c_obj = box_in_camera(np.array([0.0, 2.0, 0.0]), np.eye(3), LEVEL)
    np.testing.assert_allclose(center_c, [0, 0, 2.0], atol=1e-12)
    # box up is optical -y
    np.testing.assert_allclose(R_c_obj[:, 2], [0, -1, 0], atol=1e-12)


def test_box_in_camera_pitched_down():
    # Camera pitched 30 deg down: an object straight ahead and level with the camera appears above the image center
    # (negative optical y), and up leans back toward the camera (negative optical z), as the Gemini's accelerometer
    # reads when its camera looks down.
    R_wc = LEVEL @ rot_x(-30)
    center_c, R_c_obj = box_in_camera(np.array([0.0, 2.0, 0.0]), np.eye(3), R_wc)
    assert center_c[1] < 0
    np.testing.assert_allclose(np.linalg.norm(center_c), 2.0)
    np.testing.assert_allclose(
        R_c_obj[:, 2], [0, -np.cos(np.radians(30)), -0.5], atol=1e-12
    )
