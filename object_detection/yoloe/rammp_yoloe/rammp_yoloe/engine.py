"""Build and locate YOLOE TensorRT engines.

YOLOE bakes its class prompts into the detection head at export, and a TensorRT engine is specific to the
device it was built for (GPU, DLA core 0 or 1), the input size, and the TensorRT version. The engine file name
carries all of that, so changing any of it builds a new engine instead of loading a stale one.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile

DEVICES = {"gpu": 0, "dla0": "dla:0", "dla1": "dla:1"}
TEXT_ENCODER = "mobileclip2_b.ts"  # YOLOE-26 turns prompts into embeddings with this, at export only


def engine_path(
    engine_dir: str, model: str, prompts: list[str], imgsz: int, device: str
) -> str:
    import tensorrt
    import ultralytics

    key = json.dumps(
        {
            "model": model,
            "prompts": prompts,
            "imgsz": imgsz,
            "device": device,
            "tensorrt": tensorrt.__version__,
            "ultralytics": ultralytics.__version__,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return os.path.join(engine_dir, f"{model}-{device}-{imgsz}-fp16-{digest}.engine")


def build(
    model_dir: str,
    engine_dir: str,
    model: str,
    prompts: list[str],
    imgsz: int,
    device: str,
    log=print,
) -> str:
    """Export model with prompts baked in to a TensorRT FP16 engine for device. Returns the engine path."""
    from ultralytics import YOLOE

    if device not in DEVICES:
        raise ValueError(f"device must be one of {sorted(DEVICES)}, got {device!r}")
    out = engine_path(engine_dir, model, prompts, imgsz, device)
    os.makedirs(engine_dir, exist_ok=True)
    # Ultralytics looks for the text encoder in the working directory and writes exports next to the .pt.
    work = tempfile.mkdtemp(prefix="build-", dir=engine_dir)
    cwd = os.getcwd()
    try:
        for f in (f"{model}.pt", TEXT_ENCODER):
            os.symlink(os.path.join(model_dir, f), os.path.join(work, f))
        os.chdir(work)
        log(
            f"building {os.path.basename(out)}: {len(prompts)} prompts, {imgsz}px, {device}. Takes a few minutes."
        )
        m = YOLOE(f"{model}.pt")
        m.set_classes(prompts)
        m.save("vocab.pt")
        exported = YOLOE("vocab.pt").export(
            format="engine", imgsz=imgsz, half=True, device=DEVICES[device]
        )
        os.replace(exported, out)
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
    log(f"built {out}")
    return out


def main() -> None:
    """Build the engine a node with the given parameters would use, e.g. ahead of a demo:
    ros2 run rammp_yoloe build_engine --model yoloe-26s-seg --device dla0"""
    import argparse

    from rammp_detection_common.classes import ClassMap

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--model", default="yoloe-26s-seg")
    ap.add_argument("--device", default="gpu", choices=sorted(DEVICES))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes-file", default="/etc/rammp/perception/classes.yaml")
    ap.add_argument("--model-dir", default="/models/yoloe")
    ap.add_argument("--engine-dir", default="/cache/engines")
    a = ap.parse_args()
    prompts = ClassMap.load(a.classes_file).prompts
    path = engine_path(a.engine_dir, a.model, prompts, a.imgsz, a.device)
    if os.path.exists(path):
        print(f"already built: {path}")
        return
    build(a.model_dir, a.engine_dir, a.model, prompts, a.imgsz, a.device)
