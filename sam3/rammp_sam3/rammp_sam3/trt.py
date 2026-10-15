"""TensorRT image encoders for SAM3: export to ONNX, build an FP16 engine, and run it in place of the PyTorch encoder.

Engines are specific to the GPU and the TensorRT version, so they are built on the Jetson that runs them and
cached on disk next to a metadata file; an engine whose metadata does not match the running TensorRT is ignored.
"""

from __future__ import annotations

import json
import os
import time

import torch

META_SUFFIX = ".json"


def engine_path(engine_dir: str, kind: str, res: int) -> str:
    return os.path.join(engine_dir, f"{kind}_encoder_{res}_fp16.engine")


def _identity() -> dict:
    import tensorrt as trt

    return {
        "tensorrt": trt.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def engine_usable(path: str) -> tuple[bool, str]:
    if not os.path.exists(path):
        return False, "not built"
    try:
        meta = json.load(open(path + META_SUFFIX))
    except (OSError, ValueError):
        return False, "metadata missing"
    want = _identity()
    for k in ("tensorrt", "device", "capability"):
        if meta.get(k) != want[k]:
            return False, f"built for {k}={meta.get(k)}, running {want[k]}"
    return True, "ok"


class TrtEncoder:
    """Runs a serialized encoder engine: fp32 `pixel_values` in, fp32 `fpn_*` (and optional `last_hidden`) out."""

    def __init__(self, path: str):
        import tensorrt as trt

        self._trt = trt
        self.engine = trt.Runtime(
            trt.Logger(trt.Logger.WARNING)
        ).deserialize_cuda_engine(open(path, "rb").read())
        self.context = self.engine.create_execution_context()
        names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.outputs = {}
        for n in names:
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                dtype = (
                    torch.float16
                    if self.engine.get_tensor_dtype(n) == trt.DataType.HALF
                    else torch.float32
                )
                self.outputs[n] = torch.empty(
                    tuple(self.engine.get_tensor_shape(n)), dtype=dtype, device="cuda"
                )
        self.input_dtype = (
            torch.float16
            if self.engine.get_tensor_dtype("pixel_values") == trt.DataType.HALF
            else torch.float32
        )
        self.num_fpn = sum(1 for n in self.outputs if n.startswith("fpn_"))
        self.stream = torch.cuda.Stream()

    def __call__(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        x = pixel_values.to(self.input_dtype).contiguous()
        self.stream.wait_stream(torch.cuda.current_stream())
        self.context.set_tensor_address("pixel_values", x.data_ptr())
        for n, t in self.outputs.items():
            self.context.set_tensor_address(n, t.data_ptr())
        self.context.execute_async_v3(self.stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(self.stream)
        return self.outputs


def export_onnx(module: torch.nn.Module, res: int, path: str) -> None:
    x = torch.randn(1, 3, res, res, device="cuda")
    with torch.inference_mode():
        outs = module(x)
    names = [f"fpn_{i}" for i in range(len(outs[0]))] + (
        ["last_hidden"] if outs[1] is not None else []
    )

    class Flat(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            fpn, last = self.m(pixel_values)
            return tuple(fpn) + ((last,) if last is not None else ())

    torch.onnx.export(
        Flat(module).eval(),
        (x,),
        path,
        input_names=["pixel_values"],
        output_names=names,
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )


def build_engine(
    onnx_path: str, out_path: str, workspace_gib: int = 8, log=print
) -> None:
    import tensorrt as trt

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
    t = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(serialized)
    os.replace(tmp, out_path)
    json.dump(
        {
            **_identity(),
            "source": os.path.basename(onnx_path),
            "built_s": round(time.time() - t),
        },
        open(out_path + META_SUFFIX, "w"),
    )
    log(f"built {out_path} in {time.time() - t:.0f} s")
