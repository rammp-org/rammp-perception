"""OWL-ViT open-vocabulary detection the way NanoOWL runs it, with the image side in one TensorRT FP16 engine.

NanoOWL (NVIDIA-AI-IOT/nanoowl) runs OWL-ViT's vision encoder and its box and class heads as a TensorRT engine and
embeds the prompts once, so a frame costs one engine call and a small matrix product. This does the same with
transformers' OWL-ViT and the TensorRT Python API, so the image needs neither nanoowl, torch2trt nor trtexec.

The prompts are embedded on the CPU at startup and the PyTorch model is then dropped: only the engine and the
prompt embeddings stay on the GPU. An engine is specific to the model, the GPU and the TensorRT version, and its
file name records them; a node that finds none builds one (a few minutes) and reuses it after.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import torch

from .boxes import fit

MODELS = {
    "owlvit-b32": "google/owlvit-base-patch32",
    "owlvit-b16": "google/owlvit-base-patch16",
}
SIZE = 768  # both models' input side; their position embeddings are fixed to it
# CLIP's normalization, which OWL-ViT's processor uses.
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
OUTPUTS = ("boxes", "class_embeds", "logit_shift", "logit_scale")


def load_model(model_dir: str, model: str):
    from transformers import OwlViTForObjectDetection

    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")
    return OwlViTForObjectDetection.from_pretrained(
        os.path.join(model_dir, model)
    ).eval()


def engine_path(engine_dir: str, model: str) -> str:
    import tensorrt
    import transformers

    key = json.dumps(
        {
            "model": model,
            "size": SIZE,
            "precision": "fp16, layer norms fp32",
            "device": torch.cuda.get_device_name(0),
            "tensorrt": tensorrt.__version__,
            "transformers": transformers.__version__,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return os.path.join(engine_dir, f"{model}-{SIZE}-fp16-{digest}.engine")


class ImageHead(torch.nn.Module):
    """Pixels to every patch's box and normalized class embedding, plus the logit shift and scale its class head
    applies to a prompt's similarity. Everything in OwlViTForObjectDetection that does not depend on the prompts."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, pixel_values):
        feature_map = self.m.image_embedder(pixel_values=pixel_values)[0]
        b, h, w, d = feature_map.shape
        feats = feature_map.reshape(b, h * w, d)
        boxes = self.m.box_predictor(feats, feature_map)  # cx, cy, w, h in [0, 1]
        head = self.m.class_head
        embeds = head.dense0(feats)
        embeds = embeds / (torch.linalg.norm(embeds, dim=-1, keepdim=True) + 1e-6)
        return (
            boxes,
            embeds,
            head.logit_shift(feats),
            head.elu(head.logit_scale(feats)) + 1,
        )


def build(
    model_dir: str, engine_dir: str, model: str, workspace_gib: int = 4, log=print
) -> str:
    """Export the image side of model to ONNX and build a TensorRT FP16 engine of it. Returns the engine path."""
    import tensorrt as trt

    out = engine_path(engine_dir, model)
    os.makedirs(engine_dir, exist_ok=True)
    onnx_path = out[: -len(".engine")] + ".onnx"
    t = time.time()
    log(f"exporting {model} to ONNX")
    torch.onnx.export(
        ImageHead(load_model(model_dir, model)),
        (torch.zeros(1, 3, SIZE, SIZE),),
        onnx_path,
        input_names=["pixel_values"],
        output_names=list(OUTPUTS),
        opset_version=17,
        dynamo=False,
    )
    log(f"building the TensorRT FP16 engine for {model} (a few minutes)")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}: {errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    # Layer norms in fp32: it halves the largest score error against PyTorch fp32 (0.066 to 0.035 over the bench
    # recording) and moves boxes less, at no measurable cost. Pinning more (heads, MLPs, the last blocks) did not
    # help more than rebuilding does; the rest of the gap is fp16 itself, about 3% of detections within 0.035 of
    # their threshold flipping.
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type == trt.LayerType.NORMALIZATION:
            layer.precision = trt.float32
            layer.set_output_type(0, trt.float32)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    with open(out + ".tmp", "wb") as f:
        f.write(serialized)
    os.replace(out + ".tmp", out)
    os.remove(onnx_path)
    log(f"built {out} in {time.time() - t:.0f} s")
    return out


class Engine:
    """A built ImageHead engine: fp32 pixel_values in, the OUTPUTS tensors out, in buffers reused every call."""

    def __init__(self, path: str):
        import tensorrt as trt

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.outputs = {
            n: torch.empty(
                tuple(self.engine.get_tensor_shape(n)),
                dtype=torch.float32,
                device="cuda",
            )
            for n in OUTPUTS
        }
        for n, t in self.outputs.items():
            self.context.set_tensor_address(n, t.data_ptr())

    def __call__(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        self.context.set_tensor_address("pixel_values", pixel_values.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.outputs


@torch.inference_mode()
def embed_prompts(model, prompts: list[str]) -> torch.Tensor:
    """Each prompt's normalized text embedding, as OWL-ViT's class head compares them: [prompts, dim]."""
    from transformers import OwlViTProcessor

    processor = OwlViTProcessor.from_pretrained(model.name_or_path)
    tokens = processor(
        text=prompts, return_tensors="pt", padding="max_length", truncation=True
    )
    q = model.owlvit.get_text_features(
        input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
    )
    q = getattr(
        q, "pooler_output", q
    )  # transformers returns the projected embeddings, wrapped in some versions
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-6)


class OwlDetector:
    """Scores every patch of a frame against groups of prompts (one group per label).

    A patch's score for a group is its best prompt's; each patch goes to its best group. Patches scoring under
    their group's threshold are dropped on the GPU, so only candidates come back to the CPU.
    """

    def __init__(
        self,
        model_dir: str,
        engine_dir: str,
        model: str,
        groups: list[list[str]],
        log=print,
    ):
        path = engine_path(engine_dir, model)
        if not os.path.exists(path):
            path = build(model_dir, engine_dir, model, log=log)
        self.engine = Engine(path)
        self.engine_file = os.path.basename(path)

        prompts = [p for g in groups for p in g]
        self.queries = embed_prompts(
            load_model(model_dir, model), prompts
        ).cuda()  # [prompts, dim]
        # Prompt i belongs to group prompt_group[i].
        self.prompt_group = torch.tensor(
            [gi for gi, g in enumerate(groups) for _ in g], device="cuda"
        )
        self.num_groups = len(groups)
        self.mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1) * 255
        self.std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1) * 255
        self.pixels = torch.zeros(1, 3, SIZE, SIZE, device="cuda")
        # TensorRT synchronizes extra on the default stream, so the whole frame runs on its own.
        self.stream = torch.cuda.Stream()

    @torch.inference_mode()
    def __call__(
        self, rgb: np.ndarray, thresholds: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Candidates in rgb (H x W x 3 uint8) scoring at least their group's threshold ([groups] on the GPU).

        Returns (cx, cy, w, h) boxes as fractions of the square input, scores, and group indices, as numpy.
        """
        with torch.cuda.stream(self.stream):
            return self._detect(rgb, thresholds)

    def _detect(
        self, rgb: np.ndarray, thresholds: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = fit(rgb.shape[0], rgb.shape[1], SIZE)
        x = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float()
        x = torch.nn.functional.interpolate(
            x, size=(h, w), mode="bilinear", antialias=True, align_corners=False
        )
        self.pixels.zero_()  # padding: 0 after normalization is the mean colour
        self.pixels[:, :, :h, :w] = (x - self.mean) / self.std
        out = self.engine(self.pixels)

        logits = out["class_embeds"][0] @ self.queries.T  # [patches, prompts]
        probs = torch.sigmoid((logits + out["logit_shift"][0]) * out["logit_scale"][0])
        per_group = torch.zeros(
            probs.shape[0], self.num_groups, device="cuda"
        ).scatter_reduce_(1, self.prompt_group.expand_as(probs), probs, "amax")
        scores, group = per_group.max(dim=1)
        keep = scores >= thresholds[group]
        return (
            out["boxes"][0][keep].cpu().numpy(),
            scores[keep].cpu().numpy(),
            group[keep].cpu().numpy(),
        )


def main() -> None:
    """build_engine: build an engine ahead of time, so the node's first start does not wait for it."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--model", default="owlvit-b32", choices=sorted(MODELS))
    ap.add_argument("--model-dir", default="/models/owlvit")
    ap.add_argument("--engine-dir", default="/cache/engines")
    a = ap.parse_args()
    path = engine_path(a.engine_dir, a.model)
    print(path if os.path.exists(path) else build(a.model_dir, a.engine_dir, a.model))
