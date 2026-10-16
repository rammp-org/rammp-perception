"""Run every tracked object through SAM3's memory attention and mask decoder in one batched pass.

Stock transformers runs the tracker once per object with batch size 1. This patch batches them on plain
streaming frames (no new prompts this frame, every object's memory the same length) and falls back to the
stock path otherwise. Verified against transformers 5.19.0 only: it calls private methods, so `install`
refuses any other version rather than risk silently wrong masks.

Two stock behaviours it works around for batch > 1:
- `_prepare_vision_features(..., batch_size>1)` calls `.expand` on a list, so features are expanded here.
- best-mask selection in `_single_frame_forward` broadcasts to (n, n, ...); object i's result is at [i, i].
"""

import types

import torch
import transformers
from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
    Sam3TrackerVideoSegmentationOutput,
)

VERIFIED_TRANSFORMERS = "5.19.0"


def _object_memory(model, sess, obj_idx, frame_idx, num_channels, device):
    tp = model._gather_memory_frame_outputs(sess, obj_idx, frame_idx, False)
    mems, pos = model._build_memory_attention_inputs(tp, device)
    offsets, ptrs, max_ptrs = model._get_object_pointers(
        sess, obj_idx, frame_idx, sess.num_frames, device, False, True
    )
    n_ptr = 0
    if ptrs:
        p, ppos = model._process_object_pointers(
            offsets, ptrs, max_ptrs, 1, num_channels, device
        )
        if p is not None:
            mems.append(p)
            pos.append(ppos)
            n_ptr = p.shape[0]
    return torch.cat(mems, 0).to(sess.dtype), torch.cat(pos, 0), n_ptr


def install(model) -> dict | None:
    """Patch `model.forward` in place. Returns counters of batched/fallback frames, or None if not installed."""
    if transformers.__version__ != VERIFIED_TRANSFORMERS:
        return None
    stats = {"batched": 0, "fallback": 0}
    stock_forward = model.forward

    def forward(
        self,
        inference_session,
        frame_idx=None,
        frame=None,
        reverse=False,
        run_mem_encoder=True,
        **kw,
    ):
        sess = inference_session
        n = sess.get_obj_num()
        if (
            frame is None
            or reverse
            or n < 2
            or sess.obj_with_new_inputs
            or self.num_maskmem == 0
        ):
            stats["fallback"] += 1
            return stock_forward(
                inference_session=sess,
                frame_idx=frame_idx,
                frame=frame,
                reverse=reverse,
                run_mem_encoder=run_mem_encoder,
                **kw,
            )
        frame_idx = sess.add_new_frame(frame, frame_idx)
        feats, pos_embeds = self._prepare_vision_features(
            sess, frame_idx, 1
        )  # (seq, 1, C) per level
        device, C = feats[-1].device, self.hidden_dim
        H, W = self.backbone_feature_sizes[-1]
        mems = [_object_memory(self, sess, i, frame_idx, C, device) for i in range(n)]
        if len({(m.shape[0], k) for m, _, k in mems}) != 1:
            # Objects added at different times have different memory lengths; the frame is already in the
            # session and its features cached, so the stock path can take it from here.
            stats["fallback"] += 1
            return stock_forward(
                inference_session=sess,
                frame_idx=frame_idx,
                frame=None,
                reverse=reverse,
                run_mem_encoder=run_mem_encoder,
                **kw,
            )
        feats = [f.expand(f.shape[0], n, *f.shape[2:]) for f in feats]
        pos_embeds = [p.expand(p.shape[0], n, *p.shape[2:]) for p in pos_embeds]
        cond = self.memory_attention(
            current_vision_features=feats[-1],
            current_vision_position_embeddings=pos_embeds[-1],
            memory=torch.cat([m for m, _, _ in mems], 1),
            memory_posision_embeddings=torch.cat([p for _, p, _ in mems], 1),
            num_object_pointer_tokens=mems[0][2],
        )
        pix_feat = cond.reshape(n, H * W, C).transpose(1, 2).reshape(n, C, H, W)
        high_res = [
            x.permute(1, 2, 0).reshape(x.size(1), x.size(2), *s)
            for x, s in zip(feats[:-1], self.backbone_feature_sizes[:-1])
        ]
        sam = self._single_frame_forward(
            pixel_values=None,
            input_points=None,
            input_labels=None,
            input_masks=None,
            image_embeddings=high_res + [pix_feat],
            multimask_output=self._use_multimask(False, None),
        )

        def diag(t):
            return (
                torch.stack([t[i, i] for i in range(n)])[:, None]
                if t.dim() >= 2 and t.shape[1] == n
                else t
            )

        pred, ptr, high = (
            diag(sam.pred_masks),
            diag(sam.object_pointer),
            diag(sam.high_res_masks),
        )
        score = sam.object_score_logits  # (n, 1, 1)
        for i in range(n):
            sess.store_output(
                i,
                frame_idx,
                output_value={
                    "pred_masks": pred[i : i + 1],
                    "object_pointer": ptr[i : i + 1],
                    "high_res_masks": high[i : i + 1],
                    "object_score_logits": score[i : i + 1],
                },
                is_conditioning_frame=False,
            )
            sess.frames_tracked_per_obj[i][frame_idx] = {"reverse": reverse}
        if run_mem_encoder:
            self._batch_encode_memories(
                inference_session=sess,
                frame_idx=frame_idx,
                objects_needing_memory_encoding=list(range(n)),
                high_res_masks_for_memory=[high[i : i + 1] for i in range(n)],
                object_score_logits_for_memory=[score[i : i + 1] for i in range(n)],
                is_mask_from_pts_per_obj=[False] * n,
            )
        stats["batched"] += 1
        return Sam3TrackerVideoSegmentationOutput(
            object_ids=sess.obj_ids.copy(),
            pred_masks=pred,
            object_score_logits=score.squeeze(-1),
            frame_idx=frame_idx,
        )

    model.forward = types.MethodType(forward, model)
    return stats
