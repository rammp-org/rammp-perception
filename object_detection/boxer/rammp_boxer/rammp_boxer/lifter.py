"""Lift 2D boxes to 3D boxes on one RGB-D frame with BoxerNet's TensorRT engines.

The non-network steps (per-patch depth, camera rays, gravity alignment, the box head) are Boxer's own code, so the
numbers match the reference implementation; only the encoder and decoder run as engines (see engine.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from . import engine
from .geometry import box_in_camera, boxes_to_boxer, points_from_depth, scale_intrinsics


@dataclass
class Box3D:
    prob: float  # BoxerNet's confidence in the 3D box, [0, 1]
    center: np.ndarray  # (3,) in the camera optical frame, meters
    rotation: (
        np.ndarray
    )  # (3, 3) box axes in the camera optical frame; the box's z axis is gravity-up
    size: np.ndarray  # (3,) full extent along the box's x, y, z axes, meters


class TrtModule:
    """A TensorRT engine called on torch CUDA tensors, with dynamic input shapes."""

    def __init__(self, path: str):
        import tensorrt as trt

        self.trt = trt
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.inputs = [
            n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        self.outputs = [n for n in names if n not in self.inputs]

    def _dtype(self, name):
        return {self.trt.float32: torch.float32, self.trt.float16: torch.float16}[
            self.engine.get_tensor_dtype(name)
        ]

    def __call__(self, *args):
        keep = []
        for name, t in zip(self.inputs, args):
            t = t.to(self._dtype(name)).contiguous()
            keep.append(t)
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
        (out_name,) = self.outputs
        out = torch.empty(
            tuple(self.context.get_tensor_shape(out_name)),
            dtype=self._dtype(out_name),
            device="cuda",
        )
        self.context.set_tensor_address(out_name, out.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return out.float()


class Lifter:
    def __init__(self, engine_dir: str, log=print):
        self.bn = engine.load_boxernet("cuda")
        self.hw = self.bn.hw
        paths = engine.engine_paths(engine_dir)
        if not all(os.path.exists(p) for p in paths):
            engine.build(engine_dir, bn=self.bn, log=log)
        self.encoder, self.decoder = (TrtModule(p) for p in paths)
        # The engines carry the backbone and attention; keep only what the glue and head use.
        for name in ("dino", "self_attn", "cross_attn", "input2emb", "query2emb"):
            setattr(self.bn, name, None)
        torch.cuda.empty_cache()

        from boxernet.boxernet import (
            generate_plucker_encoding,
            gravity_align_T_world_cam,
            sdp_to_patches,
        )
        from utils.tw.camera import CameraTW
        from utils.tw.pose import PoseTW

        self._plucker, self._gravity_align, self._sdp_to_patches = (
            generate_plucker_encoding,
            gravity_align_T_world_cam,
            sdp_to_patches,
        )
        self._CameraTW, self._PoseTW = CameraTW, PoseTW

    def _camera(self, fx, fy, cx, cy):
        """A pinhole CameraTW for the hw x hw input, as Boxer's loaders build one (BaseLoader.pinhole_from_K)."""
        data = [
            self.hw,
            self.hw,
            fx,
            fy,
            cx,
            cy,
            -1,
            1e-3,
            99999.0,
            99999.0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
        ]
        return self._CameraTW(torch.tensor(data, dtype=torch.float32)[None]).cuda()

    @torch.no_grad()
    def lift(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
        R_wc: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> list[Box3D]:
        """3D boxes for 2D boxes on one frame.

        rgb: (H, W, 3) uint8 RGB. depth_m: (H, W) meters registered to rgb, 0 where invalid. K: rgb's 3x3
        intrinsics (undistorted pinhole). R_wc: rotation gravity frame (+z up) <- camera optical frame.
        boxes_xyxy: (N, 4) [x_min, y_min, x_max, y_max] in rgb pixels.
        """
        if len(boxes_xyxy) == 0:
            return []
        H, W = rgb.shape[:2]
        hw = self.hw
        fx, fy, cx, cy = scale_intrinsics(K, W, H, hw)
        R_wc = np.asarray(R_wc, dtype=np.float32)

        img = torch.from_numpy(
            cv2.resize(rgb, (hw, hw), interpolation=cv2.INTER_LINEAR)
        ).cuda()
        img = img.permute(2, 0, 1)[None].float() / 255.0
        depth = cv2.resize(
            depth_m.astype(np.float32), (hw, hw), interpolation=cv2.INTER_NEAREST
        )
        points = torch.from_numpy(
            points_from_depth(depth, fx, fy, cx, cy, R_wc)
        ).cuda()[None]

        cam = self._camera(fx, fy, cx, cy)
        T_wc = self._PoseTW(torch.tensor([*R_wc.flatten(), 0.0, 0.0, 0.0])[None]).cuda()
        T_wv = self._gravity_align(T_wc, z_grav=True)
        sdp = self._sdp_to_patches(points, cam, T_wc, hw, hw, 16)
        ray = self._plucker(1, hw // 16, hw // 16, 16, cam, T_wv.inverse() @ T_wc)
        tokens = self.encoder(img, sdp, ray)

        q = torch.from_numpy(boxes_to_boxer(boxes_xyxy, W, H, hw)).cuda()[None]
        q[..., :2] = (q[..., :2] + 0.5) / hw
        q[..., 2:] = (q[..., 2:] + 0.5) / hw
        query = torch.cat(
            [self.decoder(c, tokens) for c in q.split(engine.MAX_BOXES, dim=1)], dim=1
        )
        batch = {
            "T_world_voxel0": T_wv,
            "obbs_valid": torch.ones(
                q.shape[:2] + (1,), dtype=torch.bool, device="cuda"
            ),
        }
        _, out = self.bn.head(batch, {"query": query})
        obbs = out["obbs_pr_w"][0]

        prob = obbs.prob.reshape(-1).cpu().numpy()
        centers = obbs.bb3_center_world.reshape(-1, 3).cpu().numpy()
        rotations = obbs.T_world_object.R.reshape(-1, 3, 3).cpu().numpy()
        ext = obbs.bb3_object.reshape(-1, 6).cpu().numpy()
        boxes = []
        for i in range(len(prob)):
            center, rotation = box_in_camera(centers[i], rotations[i], R_wc)
            size = ext[i, 1::2] - ext[i, 0::2]
            boxes.append(Box3D(float(prob[i]), center, rotation, size))
        return boxes
