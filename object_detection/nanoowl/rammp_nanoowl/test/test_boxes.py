import numpy as np

from rammp_nanoowl.boxes import fit, nms, nms_per_label, rectangle, to_source_xyxy


def test_fit_scales_the_longer_side():
    assert fit(720, 1280, 768) == (432, 768)
    assert fit(1280, 720, 768) == (768, 432)
    assert fit(500, 500, 768) == (768, 768)


def test_to_source_xyxy_undoes_the_square_padding():
    # A 1280 x 720 frame fills the top 9/16 of the square: one unit of the square is 1280 px either way.
    box = np.array([[0.5, 0.25, 0.25, 0.125]])  # cx, cy, w, h
    np.testing.assert_allclose(to_source_xyxy(box, 720, 1280), [[480, 240, 800, 400]])


def test_to_source_xyxy_clips_to_the_frame():
    # A box reaching into the padding below the frame ends at the frame's bottom edge.
    box = np.array([[0.5, 0.55, 0.2, 0.1]])
    xyxy = to_source_xyxy(box, 720, 1280)
    assert xyxy[0, 3] == 720
    np.testing.assert_allclose(xyxy[0, :3], [512, 640, 768])


def test_nms_drops_overlaps_and_keeps_the_rest():
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [0, 0, 10, 4]], float
    )
    scores = np.array([0.5, 0.9, 0.4, 0.3])
    # 1 beats 0 (IoU 0.68); 2 is apart; 3 overlaps 1 by IoU 0.24 (and 0, which is gone, by 0.4).
    assert nms(boxes, scores, 0.3).tolist() == [1, 2, 3]
    assert nms(boxes, scores, 0.2).tolist() == [1, 2]


def test_nms_of_nothing():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.3).tolist() == []
    assert (
        nms_per_label(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), 0.3).tolist()
        == []
    )


def test_nms_per_label_keeps_overlapping_boxes_of_different_labels():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]], float)
    scores = np.array([0.2, 0.6, 0.4])
    labels = np.array([0, 1, 0])
    assert nms_per_label(boxes, scores, labels, 0.3).tolist() == [1, 2]


def test_rectangle_is_the_box_outline():
    np.testing.assert_array_equal(
        rectangle(np.array([1, 2, 3, 4])), [[1, 2], [3, 2], [3, 4], [1, 4]]
    )
