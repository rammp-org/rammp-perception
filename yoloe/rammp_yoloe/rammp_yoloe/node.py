"""YOLOE instance segmentation, on the GPU or a DLA core, as an always-on detector node.

Parameters (besides rammp_detection_common's image_topic, rate_hz, classes_file, publish_overlay, mock):
  model       yoloe-26s-seg | yoloe-26m-seg      26m has the better masks; 26s is the one that fits a DLA core
  device      gpu | dla0 | dla1                  dla runs most layers on a DLA core and frees the GPU
  imgsz       640
  model_dir   /models/yoloe                       weights and text encoder, baked into the image
  engine_dir  /cache/engines                      TensorRT engines; mount a volume so they survive restarts
An engine missing from engine_dir is built at startup (a few minutes), then reused.
"""

from __future__ import annotations

import os

import numpy as np

from rammp_detection_common.classes import ClassMap, Detection
from rammp_detection_common.node import run

from . import engine


class YoloeBackend:
    def __init__(self, node, classes: ClassMap):
        from ultralytics import YOLO

        p = lambda name, default: node.declare_parameter(name, default).value  # noqa: E731
        model = p("model", "yoloe-26s-seg")
        device = p("device", "gpu")
        self.imgsz = int(p("imgsz", 640))
        model_dir = p("model_dir", "/models/yoloe")
        engine_dir = p("engine_dir", "/cache/engines")

        self.classes = classes
        prompts = classes.prompts
        path = engine.engine_path(engine_dir, model, prompts, self.imgsz, device)
        if not os.path.exists(path):
            path = engine.build(
                model_dir,
                engine_dir,
                model,
                prompts,
                self.imgsz,
                device,
                log=node.get_logger().info,
            )
        self.model = YOLO(path, task="segment")
        names = list(self.model.names.values())
        if names != prompts:  # the hash in the file name should make this impossible
            raise RuntimeError(
                f"{path} was built for prompts {names}, expected {prompts}"
            )
        self.model.predict(
            np.zeros((480, 640, 3), np.uint8), imgsz=self.imgsz, verbose=False
        )  # warm up
        self.description = (
            f"{model} on {device}, {self.imgsz}px, engine {os.path.basename(path)}"
        )

    def infer(self, rgb: np.ndarray) -> list[Detection]:
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        r = self.model.predict(
            bgr, imgsz=self.imgsz, conf=self.classes.lowest_min_score, verbose=False
        )[0]
        if not len(r.boxes):
            return []
        polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
        return [
            Detection(
                label=self.classes.prompt_to_label[r.names[int(c)]],
                score=float(s),
                box=b,
                polygon=poly,
            )
            for c, s, b, poly in zip(
                r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.xyxy.cpu().numpy(),
                polys,
            )
        ]


def main(args=None) -> None:
    run(YoloeBackend, args)
