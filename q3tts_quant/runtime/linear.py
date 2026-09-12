from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import lru_cache

import torch
from torch import nn


_ACTIVE_QUANT_BACKEND: ContextVar[str | None] = ContextVar("q3tts_quant_backend", default=None)


@lru_cache(maxsize=1)
def _comfy_kitchen():
    try:
        import comfy_kitchen
    except ImportError as exc:
        raise RuntimeError("comfy-kitchen is required for W8A8 ConvRot inference") from exc
    return comfy_kitchen


@contextmanager
def quant_backend_scope(backend: str):
    """Select Comfy Kitchen once for a complete warmup or generation.

    Entering ``use_backend`` for every linear adds Python and registry overhead
    hundreds of times per acoustic frame.  The engine owns one outer scope;
    individual modules retain a safe standalone fallback.
    """

    kitchen = _comfy_kitchen()
    context = kitchen.use_backend(backend) if backend != "auto" else nullcontext()
    token = _ACTIVE_QUANT_BACKEND.set(backend)
    try:
        with context:
            yield
    finally:
        _ACTIVE_QUANT_BACKEND.reset(token)


class W8A8ConvRotLinear(nn.Module):
    """Packed INT8 linear that never materializes a persistent FP16 weight."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        dtype: torch.dtype,
        device: str | torch.device = "meta",
        group_size: int = 256,
        kernel_backend: str = "cuda",
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.kernel_backend = kernel_backend
        self.out_dtype = dtype
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("weight_scale", torch.empty(out_features, 1, dtype=torch.float32, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        comfy_kitchen = _comfy_kitchen()
        if _ACTIVE_QUANT_BACKEND.get() is not None:
            return comfy_kitchen.int8_linear(
                input,
                self.weight,
                self.weight_scale,
                self.bias,
                out_dtype=self.out_dtype,
                convrot=True,
                convrot_groupsize=self.group_size,
            )
        backend = self.kernel_backend
        context = comfy_kitchen.use_backend(backend) if backend != "auto" else nullcontext()
        with context:
            return comfy_kitchen.int8_linear(
                input,
                self.weight,
                self.weight_scale,
                self.bias,
                out_dtype=self.out_dtype,
                convrot=True,
                convrot_groupsize=self.group_size,
            )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, format=w8a8_convrot, group_size={self.group_size}"
        )


class CPUGatherEmbedding(nn.Module):
    """Keep a large prompt-only table in RAM and transfer only selected rows."""

    def __init__(self, num_embeddings: int, embedding_dim: int, output_device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.output_device = torch.device(output_device)
        self.output_dtype = dtype
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="meta"), requires_grad=False
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        cpu_indices = indices.to(device="cpu", non_blocking=False)
        output = torch.nn.functional.embedding(cpu_indices, self.weight)
        return output.to(device=self.output_device, dtype=self.output_dtype, non_blocking=False)

    def _apply(self, fn, recurse=True):
        # ComfyUI lifecycle moves must not promote the 593.5 MiB prompt table to VRAM.
        return self
