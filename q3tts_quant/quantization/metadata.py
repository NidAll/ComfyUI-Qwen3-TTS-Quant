from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

FORMAT_VERSION = "1.1"
QUANTIZER_VERSION = "0.2.0"
METADATA_KEY = "q3tts.metadata"
COMFY_METADATA_KEY = "_quantization_metadata"


@dataclass
class QuantizedLayer:
    format: str = "int8_tensorwise"
    weight_scale: str = "per_output_channel"
    convrot: bool = True
    convrot_groupsize: int = 256
    source_dtype: str = "bfloat16"
    shape: list[int] = field(default_factory=list)
    component: str = "talker"
    layer_kind: str = "linear"


@dataclass
class QuantizationMetadata:
    format_version: str = FORMAT_VERSION
    quantizer_version: str = QUANTIZER_VERSION
    architecture: str = "qwen3_tts_voice_design_v1"
    model_variant: str = "voice_design"
    quantization_format: str = "w8a8_convrot"
    original_model: str = ""
    source_revision: str = ""
    source_fingerprint: str = ""
    compute_dtype: str = "bfloat16"
    recommended_runtime_backend: str = "comfy_kitchen_cuda"
    attention_backend: str = "sdpa"
    cuda_graphs: str = "auto"
    group_size: int = 256
    layers: dict[str, QuantizedLayer] = field(default_factory=dict)
    excluded_tensors: dict[str, str] = field(default_factory=dict)
    tensor_layout: str = "TensorWiseINT8Layout"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "QuantizationMetadata":
        data = dict(raw)
        data["layers"] = {
            name: value if isinstance(value, QuantizedLayer) else QuantizedLayer(**value)
            for name, value in data.get("layers", {}).items()
        }
        return cls(**data)

    @classmethod
    def from_json(cls, raw: str) -> "QuantizationMetadata":
        return cls.from_dict(json.loads(raw))

    def safetensors_metadata(self) -> dict[str, str]:
        comfy_layers = {name: {"format": layer.format} for name, layer in self.layers.items()}
        return {
            METADATA_KEY: self.to_json(),
            COMFY_METADATA_KEY: json.dumps(
                {"format_version": FORMAT_VERSION, "layers": comfy_layers},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "q3tts.quantization_format": self.quantization_format,
            "q3tts.architecture": self.architecture,
            "q3tts.model_variant": self.model_variant,
            "q3tts.original_model": self.original_model,
            "q3tts.source_revision": self.source_revision,
        }


def encode_comfy_marker(group_size: int = 256) -> bytes:
    marker = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": int(group_size),
        "params": {"convrot": True, "convrot_groupsize": int(group_size)},
    }
    return json.dumps(marker, sort_keys=True, separators=(",", ":")).encode("utf-8")
