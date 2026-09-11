from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, asdict
from typing import Any

from ..errors import BackendUnavailableError


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass
class BackendStatus:
    torch: str | None
    cuda_runtime: str | None
    gpu: str | None
    compute_capability: str | None
    comfy_kitchen: str | None
    kitchen_backends: dict[str, Any]
    flash_attention: str | None
    faster_qwen3_tts: str | None
    selected_kernel_backend: str
    selected_attention_backend: str
    cuda_graphs_requested: str
    cuda_graphs_active: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_backends(settings) -> BackendStatus:
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError("PyTorch is unavailable in this Python environment") from exc
    gpu = capability = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        capability = f"{major}.{minor}"
    kitchen_backends: dict[str, Any] = {}
    try:
        import comfy_kitchen
        kitchen_backends = comfy_kitchen.list_backends()
    except Exception as exc:
        kitchen_backends = {"error": str(exc)}
    return BackendStatus(
        torch=torch.__version__,
        cuda_runtime=getattr(torch.version, "cuda", None),
        gpu=gpu,
        compute_capability=capability,
        comfy_kitchen=package_version("comfy-kitchen"),
        kitchen_backends=kitchen_backends,
        flash_attention=package_version("flash-attn"),
        faster_qwen3_tts=package_version("faster-qwen3-tts"),
        selected_kernel_backend=settings.kernel_backend,
        selected_attention_backend=settings.attention_backend,
        cuda_graphs_requested=settings.cuda_graphs,
    )


def require_cuda_int8_backend(status: BackendStatus) -> None:
    if status.gpu is None:
        raise BackendUnavailableError("the selected runtime requires an NVIDIA CUDA device")
    if status.compute_capability is None or tuple(map(int, status.compute_capability.split("."))) < (7, 5):
        raise BackendUnavailableError("Comfy Kitchen INT8 requires compute capability 7.5 or newer")
    info = status.kitchen_backends.get("cuda", {})
    if not info.get("available", False) or info.get("disabled", False):
        reason = info.get("unavailable_reason") or "CUDA backend is not registered"
        raise BackendUnavailableError(f"Comfy Kitchen CUDA backend unavailable: {reason}")
