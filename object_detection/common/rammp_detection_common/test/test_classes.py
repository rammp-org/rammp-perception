from pathlib import Path

import numpy as np
import pytest

from rammp_detection_common.classes import (
    MAX_ID,
    ClassMap,
    Detection,
    clean,
    track_id_to_instance_id,
)

CLASSES_YAML = Path(__file__).resolve().parents[3] / "config" / "classes.yaml"


def det(label, score, box, id=0):
    return Detection(label=label, score=score, box=np.array(box, float), id=id)


def test_repo_class_list_loads():
    classes = ClassMap.load(str(CLASSES_YAML))
    assert classes.labels == [
        "food container",
        "drink cup",
        "fridge",
        "microwave",
        "table",
    ]
    assert classes.prompt_to_label["bowl"] == "food container"
    assert classes.coco_to_label["dining table"] == "table"
    assert classes.min_score("table") == 0.15
    assert classes.min_score("microwave") == 0.25
    assert classes.lowest_min_score == 0.15


def test_label_without_prompts_prompts_with_its_own_name():
    classes = ClassMap.from_dict({"classes": [{"label": "microwave"}]})
    assert classes.prompts == ["microwave"]
    assert classes.coco_to_label == {}


def test_a_prompt_mapping_to_two_labels_is_rejected():
    with pytest.raises(ValueError, match="cup"):
        ClassMap.from_dict(
            {
                "classes": [
                    {"label": "a", "prompts": ["cup"]},
                    {"label": "b", "prompts": ["cup"]},
                ]
            }
        )


def test_a_label_listed_twice_is_rejected():
    with pytest.raises(ValueError, match="twice"):
        ClassMap.from_dict({"classes": [{"label": "a"}, {"label": "a"}]})


def test_clean_applies_per_class_thresholds():
    classes = ClassMap.load(str(CLASSES_YAML))
    out = clean(
        [det("table", 0.18, [0, 0, 10, 10]), det("microwave", 0.18, [20, 20, 30, 30])],
        classes,
    )
    assert [d.label for d in out] == ["table"]


def test_clean_merges_duplicates_of_one_label_and_keeps_the_higher_score():
    classes = ClassMap.load(str(CLASSES_YAML))
    out = clean(
        [
            det("food container", 0.4, [0, 0, 100, 100]),
            det("food container", 0.65, [2, 2, 100, 100]),
            det("food container", 0.5, [200, 200, 300, 300]),
        ],
        classes,
    )
    assert [(d.score, d.id) for d in out] == [(0.65, 1), (0.5, 2)]


def test_clean_keeps_overlapping_objects_of_different_labels():
    classes = ClassMap.load(str(CLASSES_YAML))
    out = clean(
        [
            det("drink cup", 0.5, [0, 0, 10, 10]),
            det("food container", 0.5, [0, 0, 10, 10]),
        ],
        classes,
    )
    assert len(out) == 2


def test_clean_keeps_tracker_ids():
    classes = ClassMap.load(str(CLASSES_YAML))
    out = clean(
        [
            det("microwave", 0.9, [0, 0, 10, 10], id=7),
            det("table", 0.5, [20, 20, 90, 90], id=3),
        ],
        classes,
    )
    assert [d.id for d in out] == [7, 3]


def test_a_merged_duplicate_keeps_the_older_track_id():
    classes = ClassMap.load(str(CLASSES_YAML))
    out = clean(
        [
            det(
                "food container", 0.65, [0, 0, 100, 100], id=12
            ),  # "bowl" track, scoring higher this frame
            det(
                "food container", 0.40, [2, 2, 100, 100], id=4
            ),  # "food container" track, older
        ],
        classes,
    )
    assert [(d.score, d.id) for d in out] == [(0.65, 4)]


def test_track_ids_map_into_the_label_image_range():
    assert track_id_to_instance_id(0) == 1
    assert track_id_to_instance_id(MAX_ID - 1) == MAX_ID
    assert track_id_to_instance_id(MAX_ID) == 1


def test_owl_thresholds_are_separate():
    classes = ClassMap.from_dict(
        {
            "default_owl_min_score": 0.12,
            "classes": [
                {"label": "microwave"},
                {"label": "table", "min_score": 0.15, "owl_min_score": 0.08},
            ],
        }
    )
    assert classes.owl_min_scores == {"microwave": 0.12, "table": 0.08}
    assert classes.min_score("table") == 0.15
