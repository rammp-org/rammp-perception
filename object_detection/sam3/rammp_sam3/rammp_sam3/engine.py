"""SAM3 detection and tracking on one GPU, with no ROS in sight.

Detection runs the full model at `detector_res` (1008 by default): this is the expert query, judged on quality.
Tracking runs SAM3's video tracker at `tracker_res` (336 by default), seeded with the detector's masks.
Both image encoders use cached TensorRT FP16 engines when they exist and match this machine, else PyTorch bf16.

Measured on an AGX Orin 64GB (JetPack 6.2.1), 1008 detector / 336 tracker, TensorRT encoders:
  detect, 1 prompt   ~380 ms     detect, 10 prompts  ~810 ms
  track, 3 objects    ~74 ms     track, 7 objects    ~104 ms
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from . import batched_tracker, trt
from .masks import Detection, mask_box

DTYPE = torch.bfloat16
# Tracker state older than this many frames is dropped: it is the longer of the memory bank (7 frames) and the
# object-pointer window (16), so nothing the model can still read is removed.
KEEP_FRAMES = 16


def _detector_config(model_path: str, res: int):
    from transformers import Sam3Model

    cfg = Sam3Model.config_class.from_pretrained(model_path)
    vc = (
        cfg.detector_config.vision_config
        if hasattr(cfg, "detector_config")
        else cfg.vision_config
    )
    g = res // 14
    vc.backbone_config.image_size = res
    vc.backbone_feature_sizes = [[g * 4, g * 4], [g * 2, g * 2], [g, g]]
    return cfg


def _tracker_config(model_path: str, res: int):
    from transformers import Sam3TrackerVideoModel

    cfg = Sam3TrackerVideoModel.config_class.from_pretrained(model_path)
    g = res // 14
    cfg.image_size = res
    cfg.prompt_encoder_config.image_size = res
    cfg.vision_config.backbone_config.image_size = res
    cfg.vision_config.backbone_feature_sizes = [[g * 4, g * 4], [g * 2, g * 2], [g, g]]
    cfg.memory_attention_rope_feat_sizes = [g, g]
    return cfg


def _check_res(name: str, res: int) -> None:
    # The ViT attends in 24x24-patch windows of 14 px; other sizes pad and waste compute or break outright.
    if res % 336:
        raise ValueError(
            f"{name} must be a multiple of 336 (336, 672, 1008); got {res}"
        )


def load_detector(model_path: str, res: int, dtype=DTYPE):
    from transformers import Sam3Model, Sam3Processor

    model = (
        Sam3Model.from_pretrained(
            model_path, config=_detector_config(model_path, res), dtype=dtype
        )
        .to("cuda")
        .eval()
    )
    proc = Sam3Processor.from_pretrained(model_path)
    proc.image_processor.size = {"height": res, "width": res}
    return model, proc


def load_tracker(model_path: str, res: int, dtype=DTYPE):
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

    model = (
        Sam3TrackerVideoModel.from_pretrained(
            model_path, config=_tracker_config(model_path, res), dtype=dtype
        )
        .to("cuda")
        .eval()
    )
    proc = Sam3TrackerVideoProcessor.from_pretrained(model_path)
    proc.image_processor.size = {"height": res, "width": res}
    if getattr(proc, "video_processor", None) is not None:
        proc.video_processor.size = {"height": res, "width": res}
    return model, proc


@dataclass
class TrackState:
    session: object
    dets: list[Detection]  # id, label, score per tracked object, in session order
    frame_idx: int = 0
    started: float = field(default_factory=time.monotonic)


class Sam3Engine:
    def __init__(
        self,
        model_path: str,
        detector_res: int = 1008,
        tracker_res: int = 336,
        engine_dir: str | None = None,
        use_tensorrt: bool = True,
        log=print,
    ):
        _check_res("detector_res", detector_res)
        _check_res("tracker_res", tracker_res)
        self.log = log
        self.detector_res, self.tracker_res = detector_res, tracker_res
        t = time.monotonic()
        self.detector, self.det_proc = load_detector(model_path, detector_res)
        self.tracker, self.trk_proc = load_tracker(model_path, tracker_res)
        self.batch_stats = batched_tracker.install(self.tracker)
        if self.batch_stats is None:
            log(
                "transformers is not the version the batched tracker was verified on; tracking objects one at a time"
            )
        self.backend = {"detector": "pytorch", "tracker": "pytorch"}
        self._det_encoder = None
        if use_tensorrt and engine_dir:
            self._load_engines(engine_dir)
        self._text_cache: collections.OrderedDict = collections.OrderedDict()
        self.load_s = time.monotonic() - t
        log(
            f"SAM3 loaded in {self.load_s:.1f} s: detector {detector_res} px ({self.backend['detector']}), "
            f"tracker {tracker_res} px ({self.backend['tracker']})"
        )

    # ---- TensorRT -------------------------------------------------------------------------------------
    def _load_engines(self, engine_dir: str) -> None:
        path = trt.engine_path(engine_dir, "detector", self.detector_res)
        ok, why = trt.engine_usable(path)
        if ok:
            self._det_encoder = trt.TrtEncoder(path)
            with torch.inference_mode():  # positional encodings are constant per resolution; take PyTorch's once
                self._det_template = self.detector.get_vision_features(
                    pixel_values=torch.zeros(
                        1,
                        3,
                        self.detector_res,
                        self.detector_res,
                        device="cuda",
                        dtype=DTYPE,
                    )
                )
            self.backend["detector"] = "tensorrt"
        else:
            self.log(f"detector TensorRT engine unusable ({why}): {path}")
        path = trt.engine_path(engine_dir, "tracker", self.tracker_res)
        ok, why = trt.engine_usable(path)
        if ok:
            enc = trt.TrtEncoder(path)
            from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
                Sam3TrackerVideoVisionEncoderOutput,
            )

            with torch.inference_mode():
                pos = self.tracker.vision_encoder(
                    torch.zeros(
                        1,
                        3,
                        self.tracker_res,
                        self.tracker_res,
                        device="cuda",
                        dtype=DTYPE,
                    ),
                    return_dict=True,
                ).fpn_position_encoding

            class TrtVision(torch.nn.Module):
                def forward(self, pixel_values, return_dict=True, **kw):
                    out = enc(pixel_values)
                    fpn = tuple(out[f"fpn_{i}"].to(DTYPE) for i in range(len(pos)))
                    return Sam3TrackerVideoVisionEncoderOutput(
                        last_hidden_state=None,
                        fpn_hidden_states=fpn,
                        fpn_position_encoding=pos,
                    )

            self.tracker.vision_encoder = TrtVision()
            self.backend["tracker"] = "tensorrt"
        else:
            self.log(f"tracker TensorRT engine unusable ({why}): {path}")

    def _det_vision(self, pixel_values):
        if self._det_encoder is None:
            return self.detector.get_vision_features(pixel_values=pixel_values)
        out = self._det_encoder(pixel_values)
        vis = self._det_template.__class__(**dict(self._det_template.items()))
        vis["fpn_hidden_states"] = tuple(
            out[f"fpn_{i}"].to(DTYPE) for i in range(self._det_encoder.num_fpn)
        )
        if "last_hidden" in out:
            vis["last_hidden_state"] = out["last_hidden"].to(DTYPE)
        return vis

    # ---- detection ----------------------------------------------------------------------------------------
    def _text(self, prompts: tuple[str, ...]):
        hit = self._text_cache.get(prompts)
        if hit is None:
            ti = self.det_proc(text=list(prompts), return_tensors="pt").to("cuda")
            out = self.detector.text_encoder(
                input_ids=ti["input_ids"],
                attention_mask=ti["attention_mask"],
                return_dict=True,
            )
            hit = (
                self.detector.text_projection(out.last_hidden_state),
                ti["attention_mask"],
            )
            self._text_cache[prompts] = hit
            while len(self._text_cache) > 32:
                self._text_cache.popitem(last=False)
        else:
            self._text_cache.move_to_end(prompts)
        return hit

    @torch.inference_mode()
    def detect(
        self, rgb: np.ndarray, prompts: list[str], threshold: float
    ) -> list[Detection]:
        """Every instance of every prompt in one RGB image (H x W x 3, uint8). All prompts share one head pass."""
        prompts = tuple(prompts)
        n = len(prompts)
        x = self.det_proc(images=Image.fromarray(rgb), return_tensors="pt").to("cuda")
        vis = self._det_vision(x["pixel_values"].to(DTYPE))
        if n > 1:
            vis = vis.__class__(
                **{
                    k: (
                        v.expand(n, *v.shape[1:])
                        if torch.is_tensor(v)
                        else tuple(t.expand(n, *t.shape[1:]) for t in v)
                        if isinstance(v, (tuple, list))
                        else v
                    )
                    for k, v in vis.items()
                }
            )
        text, attn = self._text(prompts)
        out = self.detector(
            vision_embeds=vis,
            text_embeds=SimpleNamespace(pooler_output=text),
            attention_mask=attn,
        )
        results = self.det_proc.post_process_instance_segmentation(
            out,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=x["original_sizes"].tolist() * n,
        )
        dets = []
        for label, r in zip(prompts, results):
            for m, s, b in zip(r["masks"], r["scores"].float(), r["boxes"].float()):
                mask = m.bool().cpu().numpy()
                if mask.any():
                    dets.append(
                        Detection(
                            label=label,
                            score=float(s),
                            mask=mask,
                            box=[float(v) for v in b.cpu()],
                        )
                    )
        dets.sort(key=lambda d: -d.score)
        return dets

    # ---- tracking -----------------------------------------------------------------------------------------
    @torch.inference_mode()
    def start_track(self, rgb: np.ndarray, seeds: list[Detection]) -> TrackState:
        """Open a tracker session on this frame, seeded with `seeds` (each must already carry its id)."""
        sess = self.trk_proc.init_video_session(
            inference_device="cuda",
            processing_device="cuda",
            video_storage_device="cuda",
            dtype=DTYPE,
        )
        x = self.trk_proc(
            images=Image.fromarray(rgb), device="cuda", return_tensors="pt"
        )
        self.trk_proc.add_inputs_to_inference_session(
            inference_session=sess,
            frame_idx=0,
            obj_ids=[d.id for d in seeds],
            input_masks=[d.mask for d in seeds],
            original_size=x["original_sizes"][0],
        )
        self.tracker(inference_session=sess, frame=x["pixel_values"][0].to(DTYPE))
        state = TrackState(
            session=sess,
            dets=[
                Detection(d.label, d.score, d.mask, id=d.id, box=list(d.box))
                for d in seeds
            ],
        )
        self._prune(state)
        return state

    @torch.inference_mode()
    def track(self, state: TrackState, rgb: np.ndarray) -> list[Detection]:
        """Advance the session by one frame. Every seeded object comes back; `visible` is False when it is absent."""
        x = self.trk_proc(
            images=Image.fromarray(rgb), device="cuda", return_tensors="pt"
        )
        out = self.tracker(
            inference_session=state.session, frame=x["pixel_values"][0].to(DTYPE)
        )
        state.frame_idx = out.frame_idx
        masks = self.trk_proc.post_process_masks(
            [out.pred_masks], original_sizes=x["original_sizes"], binarize=True
        )[0]
        masks = (masks[:, 0] if masks.ndim == 4 else masks).cpu().numpy()
        logits = out.object_score_logits.float().flatten().cpu().numpy()
        by_id = {d.id: d for d in state.dets}
        result = []
        for obj_id, mask, logit in zip(out.object_ids, masks, logits):
            d = by_id[obj_id]
            visible = bool(logit > 0 and mask.any())
            d.mask = mask if visible else np.zeros_like(mask)
            d.visible = visible
            d.box = mask_box(d.mask)
            result.append(d)
        self._prune(state)
        return result

    @staticmethod
    def _prune(state: TrackState) -> None:
        """Drop session state the model can no longer read, so a long session does not grow without bound."""
        sess, f = state.session, state.frame_idx
        if sess.processed_frames:
            for k in [
                k for k, v in sess.processed_frames.items() if k < f and v is not None
            ]:
                sess.processed_frames[k] = (
                    None  # keep the key: the next frame index is len(processed_frames)
                )
        cutoff = f - KEEP_FRAMES
        for obj_idx in list(sess.output_dict_per_obj):
            old = sess.output_dict_per_obj[obj_idx]["non_cond_frame_outputs"]
            for k in [k for k in old if k < cutoff]:
                del old[k]
            tracked = sess.frames_tracked_per_obj.get(obj_idx, {})
            for k in [k for k in tracked if k < cutoff]:
                del tracked[k]

    def warmup(self) -> None:
        """Run each path once so the first real request does not pay first-call costs."""
        rgb = np.zeros((480, 640, 3), np.uint8)
        rgb[100:300, 200:400] = 200
        self.detect(rgb, ["object"], 0.5)
        seed = Detection("object", 1.0, np.zeros((480, 640), bool), id=1)
        seed.mask[100:300, 200:400] = True
        st = self.start_track(rgb, [seed])
        self.track(st, rgb)
        torch.cuda.synchronize()
