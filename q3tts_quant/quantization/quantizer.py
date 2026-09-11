from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from ..adapters import Qwen3TTSVoiceDesignAdapter
from ..errors import BackendUnavailableError, CheckpointError
from .metadata import QuantizationMetadata, QuantizedLayer, encode_comfy_marker
from .source import load_config, resolve_model_source
from .writer import SafetensorsShardWriter

log = logging.getLogger("q3tts.quantizer")


def _prepare_output(output: Path, force: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    generated = list(output.glob("model-*.safetensors"))
    generated += [output / "model.safetensors.index.json", output / "quantization_manifest.json"]
    existing = [path for path in generated if path.exists()]
    if existing and not force:
        raise CheckpointError(f"output already contains a generated checkpoint: {output}; use --force")
    for path in existing:
        path.unlink()


def _copy_assets(source_root: Path, config_root: Path, output: Path) -> None:
    roots = []
    for candidate in (config_root, source_root):
        if candidate not in roots:
            roots.append(candidate)
    for root in roots:
        for path in root.iterdir():
            if path.is_file() and not path.name.endswith((".safetensors", ".safetensors.index.json")):
                destination = output / path.name
                if not destination.exists():
                    shutil.copy2(path, destination)
    speech_dir = next((root / "speech_tokenizer" for root in roots if (root / "speech_tokenizer").is_dir()), None)
    if speech_dir is None:
        raise CheckpointError("speech_tokenizer directory is missing from the source snapshot")
    destination = output / "speech_tokenizer"
    shutil.copytree(speech_dir, destination, symlinks=False, dirs_exist_ok=True)


def _quantize_weight(tensor: Any, device: str, kernel_backend: str, group_size: int):
    try:
        import torch
        import comfy_kitchen
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    except ImportError as exc:
        raise BackendUnavailableError(
            "Comfy Kitchen and PyTorch are required. Run the quantizer inside the ComfyUI Python environment."
        ) from exc

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise BackendUnavailableError("--device cuda requested, but torch.cuda.is_available() is false")
    dense = tensor.to(device=target)
    context = comfy_kitchen.use_backend(kernel_backend) if kernel_backend != "auto" else None
    try:
        if context is None:
            qweight, params = TensorWiseINT8Layout.quantize(
                dense, is_weight=True, per_channel=True, convrot=True,
                convrot_groupsize=group_size, stochastic_rounding=0,
            )
        else:
            with context:
                qweight, params = TensorWiseINT8Layout.quantize(
                    dense, is_weight=True, per_channel=True, convrot=True,
                    convrot_groupsize=group_size, stochastic_rounding=0,
                )
    except Exception as exc:
        raise BackendUnavailableError(
            f"Comfy Kitchen {kernel_backend!r} backend could not quantize shape {tuple(tensor.shape)}: {exc}"
        ) from exc
    return qweight.cpu(), params.scale.float().cpu()


def quantize_model(
    model: str,
    output: str,
    *,
    revision: str = "main",
    config_dir: str | None = None,
    device: str = "cuda",
    kernel_backend: str = "cuda",
    group_size: int = 256,
    max_shard_mib: int = 768,
    force: bool = False,
) -> dict[str, Any]:
    """Stream the source checkpoint and emit heterogeneous W8A8 ConvRot shards."""
    import torch
    from safetensors import safe_open

    if group_size != 256:
        raise ValueError("the selected ConvRot policy requires group_size=256")
    started = time.perf_counter()
    source = resolve_model_source(model, revision=revision, config_dir=config_dir)
    config = load_config(source)
    specs = list(source.iter_tensor_specs())
    adapter = Qwen3TTSVoiceDesignAdapter()
    adapter.validate_checkpoint(config, (name for name, _, _ in specs))

    decisions = {name: adapter.classify_tensor(name, shape) for name, shape, _ in specs}
    output_path = Path(output).expanduser().resolve()
    _prepare_output(output_path, force)

    metadata = QuantizationMetadata(
        original_model=source.model_id,
        source_revision=source.revision,
        source_fingerprint=source.fingerprint(),
        group_size=group_size,
    )
    spec_map = {name: (shape, dtype) for name, shape, dtype in specs}
    for name, decision in decisions.items():
        shape, dtype = spec_map[name]
        if decision.quantize:
            metadata.layers[decision.module_name] = QuantizedLayer(
                source_dtype=dtype,
                shape=list(shape),
                component=decision.component,
                layer_kind=decision.layer_kind,
                convrot_groupsize=group_size,
            )
        elif name.endswith(".weight"):
            metadata.excluded_tensors[name] = decision.reason

    writer = SafetensorsShardWriter(
        output_path,
        metadata.safetensors_metadata(),
        max_shard_bytes=max_shard_mib * 1024 * 1024,
    )
    marker = torch.tensor(list(encode_comfy_marker(group_size)), dtype=torch.uint8)
    dense_bytes = 0
    stored_bytes = 0
    quantized_count = 0
    seen: set[str] = set()
    with torch.inference_mode():
        for file in source.weight_files:
            log.info("reading source shard %s", file.name)
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in seen:
                        raise CheckpointError(f"duplicate tensor {name!r} across source shards")
                    seen.add(name)
                    tensor = handle.get_tensor(name)
                    dense_bytes += tensor.numel() * tensor.element_size()
                    decision = decisions[name]
                    if decision.quantize:
                        qweight, scale = _quantize_weight(tensor, device, kernel_backend, group_size)
                        module = decision.module_name
                        writer.add(name, qweight)
                        writer.add(f"{module}.weight_scale", scale)
                        writer.add(f"{module}.comfy_quant", marker)
                        stored_bytes += qweight.numel() * qweight.element_size()
                        stored_bytes += scale.numel() * scale.element_size() + marker.numel()
                        quantized_count += 1
                        log.info("quantized %-72s %s", module, tuple(tensor.shape))
                    else:
                        writer.add(name, tensor)
                        stored_bytes += tensor.numel() * tensor.element_size()
                    del tensor
                    if (
                        decision.quantize
                        and quantized_count % 16 == 0
                        and device.startswith("cuda")
                        and torch.cuda.is_available()
                    ):
                        torch.cuda.empty_cache()

    write_summary = writer.finish()
    _copy_assets(source.root, source.config_root, output_path)
    result = {
        "status": "succeeded",
        "format": "w8a8_convrot",
        "architecture": adapter.adapter_id,
        "source": source.model_id,
        "revision": source.revision,
        "output": str(output_path),
        "quantized_layers": quantized_count,
        "source_bytes": dense_bytes,
        "stored_tensor_bytes": stored_bytes,
        "compression_ratio": dense_bytes / max(stored_bytes, 1),
        "elapsed_seconds": time.perf_counter() - started,
        **write_summary,
        "metadata": metadata.to_dict(),
    }
    (output_path / "quantization_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
