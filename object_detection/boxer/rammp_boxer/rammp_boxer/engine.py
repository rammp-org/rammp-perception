"""Build and locate BoxerNet's TensorRT engines.

BoxerNet runs as two engines: an encoder (image + per-patch depth + rays -> 3600 tokens) that depends only on the
frame, and a decoder (2D box queries + tokens -> per-box features) with a dynamic box count. The small head that
turns those features into boxes stays in PyTorch. Engines are specific to the checkpoint, the TensorRT and torch
versions and the precision recipe; the file name carries all of them, so changing any builds new engines.

Three things the stock model needs before TensorRT 10.3 builds it correctly (each found by comparing engine output
with PyTorch on real frames):

1. DINOv3's RoPE exports an If node whose branches differ in shape, which TensorRT rejects. Its sin/cos tables
   depend only on the fixed patch grid, so they are frozen.
2. TensorRT 10.3 miscompiles DINOv3's packed q/k/v split (reshape to (B, N, 3, H, D) then unbind): engine boxes
   came out about 1 m off. Slicing q, k and v out of the projection is the same math and compiles correctly.
3. DINOv3 block 0's attention logits reach ~1.2e6, past fp16's range (later blocks stay < 5e3), so a plain fp16
   build returns broken boxes. The engines are traced under fp16 autocast with block 0's attention in bf16 (fp32
   range, still a fused kernel) and built strongly typed, so TensorRT keeps exactly those dtypes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile

BOXER_ROOT = os.environ.get("BOXER_ROOT", "/opt/boxer")
BOXER_COMMIT = os.environ.get("BOXER_COMMIT", "unknown")
CHECKPOINT = "boxernet_hw960in2x6d768-c88128f8.ckpt"
MAX_BOXES = 128  # decoder engine's largest box count; more boxes in one frame are lifted in chunks
RECIPE = "fp16-amp-block0-bf16-attn"


def import_boxer() -> None:
    """Make the Boxer repository importable (it is not a package)."""
    if BOXER_ROOT not in sys.path:
        sys.path.insert(0, BOXER_ROOT)


def load_boxernet(device: str = "cuda"):
    import_boxer()
    from boxernet.boxernet import BoxerNet
    from utils.demo_utils import CKPT_PATH

    return BoxerNet.load_from_checkpoint(
        os.path.join(CKPT_PATH, CHECKPOINT), device=device
    )


def engine_paths(engine_dir: str) -> tuple[str, str]:
    import tensorrt
    import torch

    key = json.dumps(
        {
            "checkpoint": CHECKPOINT,
            "boxer": BOXER_COMMIT,
            "recipe": RECIPE,
            "max_boxes": MAX_BOXES,
            "tensorrt": tensorrt.__version__,
            "torch": torch.__version__,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return tuple(
        os.path.join(engine_dir, f"boxernet-{part}-{digest}.engine")
        for part in ("encoder", "decoder")
    )


def modules(bn):
    """The encoder and decoder as export-friendly modules sharing bn's weights."""
    import torch
    from torch import nn

    class Encoder(nn.Module):
        """Image + per-patch depth + rays -> input tokens: everything in BoxerNet.encode that depends on the frame."""

        def __init__(self):
            super().__init__()
            self.bn = bn
            self.register_buffer(
                "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

        def forward(self, img, sdp_patch, ray_enc):
            x = (img - self.mean) / self.std
            feat = self.bn.dino.model.get_intermediate_layers(
                x, reshape=True, norm=True
            )[0]
            B, C, fH, fW = feat.shape
            feat = feat.reshape(B, C, fH * fW).permute(0, 2, 1)
            sdp = sdp_patch.reshape(B, 1, fH * fW).permute(0, 2, 1)
            x = torch.cat([feat, sdp] + ([ray_enc] if self.bn.with_ray else []), dim=-1)
            enc = self.bn.input2emb(x)
            return self.bn.self_attn(enc) if self.bn.in_depth > 0 else enc

    class Decoder(nn.Module):
        """Normalized 2D box queries + input tokens -> per-box features for BoxerNet's head."""

        def __init__(self):
            super().__init__()
            self.bn = bn

        def forward(self, query_tokens, input_enc):
            return self.bn.cross_attn(self.bn.query2emb(query_tokens), input_enc)

    return Encoder().cuda().eval(), Decoder().cuda().eval()


def _attention_sliced(self, qkv, attn_bias=None, rope=None):
    """DINOv3 SelfAttention.compute_attention with q, k, v sliced out instead of reshape + unbind (fix 2), and the
    flagged block's attention in bf16 when tracing under fp16 autocast (fix 3)."""
    import torch

    B, N, C3 = qkv.shape
    C = C3 // 3
    q, k, v = (
        qkv[..., i * C : (i + 1) * C]
        .reshape(B, N, self.num_heads, C // self.num_heads)
        .transpose(1, 2)
        for i in range(3)
    )
    if rope is not None:
        q, k = self.apply_rope(q, k, rope)
    if getattr(self, "bf16_attn", False) and torch.is_autocast_enabled("cuda"):
        with torch.autocast("cuda", enabled=False):
            x = torch.nn.functional.scaled_dot_product_attention(
                q.bfloat16(), k.bfloat16(), v.bfloat16()
            )
        x = x.half()
    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return x.transpose(1, 2).reshape([B, N, C])


def export_onnx(bn, out_dir: str) -> tuple[str, str]:
    import torch
    from boxernet.dinov3_wrapper import SelfAttention

    fhw = bn.hw // 16
    rope = bn.dino.model.rope_embed
    with torch.no_grad():
        sincos = rope(H=fhw, W=fhw)
    rope.forward = lambda *, H, W: sincos  # fix 1
    SelfAttention.compute_attention = _attention_sliced  # fix 2
    bn.dino.model.blocks[0].attn.bf16_attn = True  # fix 3

    enc, dec = modules(bn)
    img = torch.rand(1, 3, bn.hw, bn.hw, device="cuda")
    sdp = torch.rand(1, 1, fhw, fhw, device="cuda")
    ray = torch.rand(1, fhw * fhw, 6, device="cuda")
    q = torch.rand(1, 8, 4, device="cuda")
    paths = (
        os.path.join(out_dir, "encoder.onnx"),
        os.path.join(out_dir, "decoder.onnx"),
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        tokens = enc(img, sdp, ray)
        torch.onnx.export(
            enc,
            (img, sdp, ray),
            paths[0],
            opset_version=17,
            dynamo=False,
            input_names=["img", "sdp_patch", "ray_enc"],
            output_names=["input_enc"],
        )
        torch.onnx.export(
            dec,
            (q, tokens),
            paths[1],
            opset_version=17,
            dynamo=False,
            input_names=["query_tokens", "input_enc"],
            output_names=["query"],
            dynamic_axes={"query_tokens": {1: "M"}, "query": {1: "M"}},
        )
    return paths


def build_engine(onnx_path: str, plan_path: str, query_profile: bool = False) -> None:
    """Strongly typed TensorRT build: the ONNX's dtypes, as traced under autocast, are kept as they are."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError(
            f"cannot parse {onnx_path}: "
            + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        )
    config = builder.create_builder_config()
    if query_profile:
        profile = builder.create_optimization_profile()
        profile.set_shape("query_tokens", (1, 1, 4), (1, 16, 4), (1, MAX_BOXES, 4))
        config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT failed to build {plan_path}")
    tmp = plan_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(plan)
    os.replace(tmp, plan_path)


def build(engine_dir: str, bn=None, log=print) -> tuple[str, str]:
    """Export and build both engines into engine_dir. Takes a few minutes on an AGX Orin."""
    enc_path, dec_path = engine_paths(engine_dir)
    os.makedirs(engine_dir, exist_ok=True)
    bn = bn or load_boxernet()
    work = tempfile.mkdtemp(prefix="build-boxernet-", dir=engine_dir)
    try:
        log(
            f"building {os.path.basename(enc_path)} and {os.path.basename(dec_path)}. Takes a few minutes."
        )
        enc_onnx, dec_onnx = export_onnx(bn, work)
        build_engine(enc_onnx, enc_path)
        build_engine(dec_onnx, dec_path, query_profile=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    log(f"built {enc_path} and {dec_path}")
    return enc_path, dec_path


def main() -> None:
    """Build the engines the node would use, e.g. ahead of a demo: ros2 run rammp_boxer build_engines"""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--engine-dir", default="/cache/engines")
    a = ap.parse_args()
    paths = engine_paths(a.engine_dir)
    if all(os.path.exists(p) for p in paths):
        print(f"already built: {paths[0]}, {paths[1]}")
        return
    build(a.engine_dir)
