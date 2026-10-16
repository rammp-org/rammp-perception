from pathlib import Path

import numpy as np
import pytest

from rammp_detection_common.classes import ClassMap, Detection, clean

CLASSES_YAML = Path(__file__).resolve().parents[3] / "config" / "classes.yaml"


def det(label, score, box):
    return Detection(label=label, score=score, box=np.array(box, float))


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
