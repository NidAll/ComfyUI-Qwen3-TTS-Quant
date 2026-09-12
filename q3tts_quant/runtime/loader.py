from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from ..config import RuntimeSettings
from ..adapters import ARCHITECTURE_IDS
from ..errors import CheckpointError, CompatibilityError
from ..quantization.metadata import METADATA_KEY, QuantizationMetadata
from .linear import CPUGatherEmbedding, W8A8ConvRotLinear

log = logging.getLogger("q3tts.loader")


def _torch_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _checkpoint_files(root: Path) -> tuple[list[Path], dict[str, str], int]:
    from safetensors import safe_open

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        files = [root / name for name in sorted(set(weight_map.values()))]
        total_size = int(index.get("metadata", {}).get("total_size", 0))
    else:
        files = sorted(root.glob("*.safetensors"))
        weight_map = {}
        total_size = sum(path.stat().st_size for path in files)
    if not files or any(not path.is_file() for path in files):
        raise CheckpointError(f"incomplete quantized checkpoint under {root}")
    with safe_open(str(files[0]), framework="pt", device="cpu") as handle:
        raw_metadata = handle.metadata() or {}
    if METADATA_KEY not in raw_metadata:
        raise CheckpointError("checkpoint lacks q3tts.metadata; quantize it with this toolkit")
    return files, raw_metadata, total_size


def read_quantized_metadata(root: str | Path) -> tuple[QuantizationMetadata, int]:
    path = Path(root).expanduser().resolve()
    files, raw, total = _checkpoint_files(path)
    try:
        metadata = QuantizationMetadata.from_json(raw[METADATA_KEY])
    except Exception as exc:
        raise CheckpointError("invalid q3tts.metadata JSON") from exc
    if metadata.quantization_format != "w8a8_convrot":
        raise CheckpointError(f"unsupported quantization format: {metadata.quantization_format}")
    return metadata, total


def _module_at(root: Any, path: str):
    current = root
    for part in path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def _parent_and_name(root: Any, path: str):
    parent_path, name = path.rsplit(".", 1)
    return _module_at(root, parent_path), name


def _replace_quantized_linears(model, metadata: QuantizationMetadata, settings: RuntimeSettings, device):
    import torch
    from torch import nn

    dtype = _torch_dtype(settings.compute_dtype)
    for module_name, layer in metadata.layers.items():
        parent, leaf = _parent_and_name(model, module_name)
        original = getattr(parent, leaf)
        if not isinstance(original, nn.Linear):
            raise CompatibilityError(f"{module_name} is {type(original).__name__}, expected Linear")
        replacement = W8A8ConvRotLinear(
            original.in_features,
            original.out_features,
            original.bias is not None,
            dtype=dtype,
            device="meta",
            group_size=layer.convrot_groupsize,
            kernel_backend=settings.kernel_backend,
        )
        setattr(parent, leaf, replacement)

    text_parent, text_leaf = _parent_and_name(model, "talker.model.text_embedding")
    text_embedding = getattr(text_parent, text_leaf)
    setattr(
        text_parent,
        text_leaf,
        CPUGatherEmbedding(text_embedding.num_embeddings, text_embedding.embedding_dim, device, dtype),
    )


def _assign_tensor(model, name: str, value, device, compute_dtype):
    from accelerate.utils import set_module_tensor_to_device

    if name.endswith(".comfy_quant"):
        return
    if value.is_floating_point() and not name.endswith("weight_scale"):
        value = value.to(dtype=compute_dtype)
    target = "cpu" if name == "talker.model.text_embedding.weight" else device
    try:
        set_module_tensor_to_device(model, name, target, value=value)
    except Exception as exc:
        raise CheckpointError(f"could not load tensor {name!r}: {exc}") from exc


def _assert_fully_materialized(model) -> None:
    unresolved = []
    for name, parameter in model.named_parameters():
        if parameter.device.type == "meta":
            unresolved.append(name)
    for name, buffer in model.named_buffers():
        if buffer.device.type == "meta":
            unresolved.append(name)
    if unresolved:
        preview = ", ".join(unresolved[:8])
        raise CheckpointError(f"checkpoint did not materialize {len(unresolved)} tensors: {preview}")


def _enforce_compute_dtype(model, compute_dtype) -> int:
    """Make every floating tensor in the Talker path use the runtime dtype.

    Quantized linears intentionally keep their packed ``int8`` weights and
    ``float32`` scales.  The remaining Qwen modules are regular PyTorch
    modules, and ``init_empty_weights`` can leave them at the constructor's
    default FP32 dtype even when the checkpoint values were converted to BF16
    or FP16 during assignment.  That creates a late failure at
    ``text_projection`` or inside ``StaticCache.index_copy_`` when the first
    prompt is built.  Cast only floating parameters and buffers, preserving
    quantized storage and scale precision.
    """
    changed = 0
    for parameter in model.parameters():
        if parameter.is_floating_point() and parameter.dtype != compute_dtype:
            parameter.data = parameter.data.to(dtype=compute_dtype)
            changed += 1
    for name, buffer in model.named_buffers():
        if name.endswith("weight_scale"):
            continue
        if buffer.is_floating_point() and buffer.dtype != compute_dtype:
            buffer.data = buffer.data.to(dtype=compute_dtype)
            changed += 1
    return changed


def load_quantized_base_model(root: str | Path, settings: RuntimeSettings, device):
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
    from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

    root = Path(root).expanduser().resolve()
    files, raw_metadata, total_size = _checkpoint_files(root)
    metadata = QuantizationMetadata.from_json(raw_metadata[METADATA_KEY])
    if metadata.architecture not in set(ARCHITECTURE_IDS.values()):
        raise CheckpointError(f"unsupported architecture metadata: {metadata.architecture}")

    config = Qwen3TTSConfig.from_pretrained(str(root), local_files_only=True)
    config_variant = str(getattr(config, "tts_model_type", "")).lower()
    metadata_variant = getattr(metadata, "model_variant", "voice_design")
    if ARCHITECTURE_IDS.get(config_variant) != metadata.architecture or metadata_variant != config_variant:
        raise CheckpointError(
            "checkpoint configuration and quantization metadata disagree: "
            f"config={config_variant!r}, metadata={metadata_variant!r}"
        )
    config._attn_implementation = settings.attention_backend
    config.talker_config._attn_implementation = settings.attention_backend
    config.talker_config.code_predictor_config._attn_implementation = settings.attention_backend
    with init_empty_weights(include_buffers=False):
        model = Qwen3TTSForConditionalGeneration(config)
    _replace_quantized_linears(model, metadata, settings, device)

    compute_dtype = _torch_dtype(settings.compute_dtype)
    loaded_names: set[str] = set()
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in loaded_names:
                    raise CheckpointError(f"duplicate tensor in quantized shards: {name}")
                loaded_names.add(name)
                _assign_tensor(model, name, handle.get_tensor(name), device, compute_dtype)
    _assert_fully_materialized(model)
    # init_empty_weights(include_buffers=False) leaves non-persistent RoPE buffers
    # on CPU. Move the model by device only. CPUGatherEmbedding deliberately blocks
    # this traversal so the large text table stays on host, while FP32 INT8 scales
    # retain their dtype.
    model.to(device=device)
    changed = _enforce_compute_dtype(model, compute_dtype)
    if changed:
        log.info("normalized %d floating Talker tensors to %s", changed, settings.compute_dtype)
    model.eval()

    processor = Qwen3TTSProcessor.from_pretrained(str(root), local_files_only=True, fix_mistral_regex=True)
    codec_device = device
    codec_dtype = _torch_dtype(settings.codec_dtype)
    speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(root / "speech_tokenizer"),
        local_files_only=True,
        device_map=str(codec_device),
        dtype=codec_dtype,
        attn_implementation="sdpa",
    )
    model.load_speech_tokenizer(speech_tokenizer)
    generation_path = root / "generation_config.json"
    if not generation_path.is_file():
        raise CheckpointError("generation_config.json is missing")
    generation_config = json.loads(generation_path.read_text(encoding="utf-8"))
    model.load_generate_config(generation_config)

    wrapper = Qwen3TTSModel(model=model, processor=processor, generate_defaults=generation_config)
    wrapper.device = torch.device(device)
    return wrapper, metadata, total_size
