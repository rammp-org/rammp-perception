import numpy as np

from rammp_sam3.masks import Detection, assign_ids, label_image, mask_box, mask_iou


def square(x0, y0, x1, y1, h=20, w=20):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_mask_box_and_empty():
    assert mask_box(square(2, 3, 5, 9)) == [2.0, 3.0, 5.0, 9.0]
    assert mask_box(np.zeros((4, 4), bool)) == [0.0, 0.0, 0.0, 0.0]


def test_mask_iou():
    assert mask_iou(square(0, 0, 10, 10), square(0, 0, 10, 10)) == 1.0
    assert mask_iou(square(0, 0, 10, 10), square(10, 10, 20, 20)) == 0.0
    assert abs(mask_iou(square(0, 0, 10, 10), square(5, 0, 15, 10)) - 50 / 150) < 1e-9
    assert mask_iou(np.zeros((3, 3), bool), np.zeros((3, 3), bool)) == 0.0


def test_label_image_higher_score_wins_overlap():
    low = Detection("cabinet", 0.6, square(0, 0, 10, 10), id=1)
    high = Detection("microwave", 0.9, square(5, 5, 15, 15), id=2)
    img = label_image([high, low], 20, 20)
    assert img.dtype == np.uint16
    assert img[7, 7] == 2  # overlap goes to the higher score
    assert img[1, 1] == 1
    assert img[19, 19] == 0


def test_label_image_skips_occluded():
    d = Detection("table", 0.8, square(0, 0, 5, 5), id=3, visible=False)
    assert label_image([d], 20, 20).max() == 0


def test_assign_ids_keeps_matching_ids_and_issues_new_ones():
    prev = [
        Detection("microwave", 0.9, square(0, 0, 10, 10), id=4),
        Detection("table", 0.7, square(10, 10, 20, 20), id=7),
    ]
    new = [
        Detection("microwave", 0.8, square(1, 0, 11, 10)),  # overlaps id 4
        Detection(
            "table", 0.6, square(0, 0, 10, 10)
        ),  # same place as the microwave, wrong label
        Detection("chair", 0.5, square(12, 0, 18, 6)),
    ]
    nxt = assign_ids(new, prev, next_id=8)
    assert [d.id for d in new] == [4, 8, 9]
    assert nxt == 10


def test_assign_ids_uses_each_previous_id_once():
    prev = [Detection("cabinet", 0.9, square(0, 0, 10, 10), id=1)]
    new = [
        Detection("cabinet", 0.9, square(0, 0, 10, 10)),
        Detection("cabinet", 0.8, square(0, 0, 9, 10)),
    ]
    assign_ids(new, prev, next_id=2)
    assert sorted(d.id for d in new) == [1, 2]
    assert new[0].id == 1  # the better overlap keeps the id
