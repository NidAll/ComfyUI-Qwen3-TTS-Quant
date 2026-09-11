from __future__ import annotations

from typing import Any, Iterable, Mapping

from .base import ArchitectureAdapter, LayerDecision


class Qwen3TTSVoiceDesignAdapter(ArchitectureAdapter):
    """Shape-aware policy for the 1.7B Qwen3-TTS VoiceDesign family."""

    adapter_id = "qwen3_tts_voice_design_v1"
    _CORE_PREFIXES = (
        "talker.model.layers.",
        "talker.code_predictor.model.layers.",
    )
    _LINEAR_SUFFIXES = (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    )
    _BRIDGE_SUFFIXES = (
        "talker.code_predictor.small_to_mtp_projection.weight",
    )

    def detect(self, config: Mapping[str, Any]) -> bool:
        model_type = str(config.get("model_type", "")).lower()
        tts_type = str(config.get("tts_model_type", "")).lower()
        talker = config.get("talker_config", {}) or {}
        return (
            model_type == "qwen3_tts"
            and tts_type == "voice_design"
            and int(talker.get("num_code_groups", 0)) == 16
        )

    def validate_checkpoint(self, config: Mapping[str, Any], tensor_names: Iterable[str]) -> None:
        if not self.detect(config):
            raise ValueError("checkpoint is not a Qwen3-TTS 12 Hz VoiceDesign model")
        names = set(tensor_names)
        required = {
            "talker.model.text_embedding.weight",
            "talker.model.codec_embedding.weight",
            "talker.codec_head.weight",
        }
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"checkpoint is missing required tensors: {missing}")

    @staticmethod
    def _component(name: str) -> str:
        if "code_predictor" in name:
            return "predictor"
        if name.startswith("talker."):
            return "talker"
        if name.startswith("speech_tokenizer."):
            return "codec"
        return "other"

    @staticmethod
    def _kind(name: str) -> str:
        if ".self_attn." in name:
            if ".q_proj." in name:
                return "attention_q"
            if ".k_proj." in name:
                return "attention_k"
            if ".v_proj." in name:
                return "attention_v"
            return "attention_output"
        if ".mlp.gate_proj." in name:
            return "mlp_gate"
        if ".mlp.up_proj." in name:
            return "mlp_up"
        if ".mlp.down_proj." in name:
            return "mlp_down"
        if "embedding" in name:
            return "embedding"
        if "norm" in name:
            return "normalization"
        if "head" in name:
            return "output_head"
        if "projection" in name:
            return "projection"
        return "other"

    def classify_tensor(self, name: str, shape: tuple[int, ...]) -> LayerDecision:
        module_name = name[:-7] if name.endswith(".weight") else name
        component = self._component(name)
        kind = self._kind(name)
        is_core = name.startswith(self._CORE_PREFIXES) and name.endswith(self._LINEAR_SUFFIXES)
        is_bridge = name.endswith(self._BRIDGE_SUFFIXES)
        aligned = len(shape) == 2 and shape[1] % 256 == 0
        quantize = bool((is_core or is_bridge) and aligned)
        if quantize:
            reason = "recurrent batch-1 linear with ConvRot-compatible K dimension"
        elif is_core or is_bridge:
            reason = "linear shape is incompatible with the selected group size"
        elif kind in {"embedding", "normalization", "output_head"}:
            reason = "quality-sensitive layer retained in source precision"
        else:
            reason = "outside the latency-critical recurrent policy"
        return LayerDecision(name, module_name, component, kind, quantize, reason)

    def get_exclusion_policy(self) -> dict[str, str]:
        return {
            "talker.model.text_embedding": "CPU BF16 selective row gather",
            "*.embedding": "BF16",
            "*.norm": "BF16 with FP32 reduction",
            "talker.codec_head": "BF16",
            "talker.code_predictor.lm_head.*": "BF16",
            "talker.text_projection": "BF16",
            "speech_tokenizer": "FP16 scheduled codec",
        }

