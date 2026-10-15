"""Build the TensorRT encoder engines on this Jetson. Run once per machine (and after a TensorRT upgrade):

    ros2 run rammp_sam3 build_engines --model-path /models/sam3 --engine-dir /cache/engines

Takes roughly 15 minutes for the 1008 px detector and 5 for the 336 px tracker on an AGX Orin.
"""

import argparse
import gc
import os

import torch

from . import trt
from .engine import load_detector, load_tracker


def _build(kind: str, res: int, model_path: str, engine_dir: str, force: bool) -> None:
    out = trt.engine_path(engine_dir, kind, res)
    ok, why = trt.engine_usable(out)
    if ok and not force:
        print(f"{kind} {res}: engine already built and current, skipping ({out})")
        return
    print(f"{kind} {res}: exporting ONNX ({why})", flush=True)
    if kind == "detector":
        model, _ = load_detector(model_path, res, dtype=torch.float32)

        class Enc(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, pv):
                o = self.m.get_vision_features(pixel_values=pv)
                return tuple(o.fpn_hidden_states), o.last_hidden_state
    else:
        model, _ = load_tracker(model_path, res, dtype=torch.float32)

        class Enc(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, pv):
                return tuple(
                    self.m.vision_encoder(pv, return_dict=True).fpn_hidden_states
                ), None

    onnx = os.path.join(engine_dir, f"{kind}_encoder_{res}.onnx")
    trt.export_onnx(Enc(model).eval(), res, onnx)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{kind} {res}: building FP16 engine (this is the slow part)", flush=True)
    trt.build_engine(onnx, out)
    for f in os.listdir(engine_dir):
        if f.startswith(os.path.basename(onnx)):
            os.remove(os.path.join(engine_dir, f))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", default="/models/sam3")
    ap.add_argument("--engine-dir", default="/cache/engines")
    ap.add_argument("--detector-res", type=int, default=1008)
    ap.add_argument("--tracker-res", type=int, default=336)
    ap.add_argument("--only", choices=["detector", "tracker"])
    ap.add_argument(
        "--force", action="store_true", help="rebuild even if a current engine exists"
    )
    a = ap.parse_args()
    os.makedirs(a.engine_dir, exist_ok=True)
    for kind, res in (("tracker", a.tracker_res), ("detector", a.detector_res)):
        if a.only in (None, kind):
            _build(kind, res, a.model_path, a.engine_dir, a.force)


if __name__ == "__main__":
    main()
