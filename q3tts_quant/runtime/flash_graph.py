from __future__ import annotations

"""FlashAttention 2 adapters for the fixed-shape CUDA graph decode path.

Transformers' generic FlashAttention adapter treats a 4D attention mask as a
padding mask and sends it through ``unpad_input``.  The graph implementation in
faster-qwen3-tts uses a 4D additive StaticCache mask, so combining the two paths
can produce invalid unpadding indices.  This module leaves dynamic prefill on
the upstream implementation and routes only StaticCache decode through
FlashAttention's purpose-built KV-cache kernel.
"""

import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable


@dataclass(frozen=True)
class FlashKVCacheStatus:
    available: bool
    reason: str | None = None


def _load_flash_kvcache() -> Callable[..., Any]:
    try:
        from flash_attn import flash_attn_with_kvcache
    except (ImportError, AttributeError):
        try:
            from flash_attn.flash_attn_interface import flash_attn_with_kvcache
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "the installed flash-attn package does not expose flash_attn_with_kvcache"
            ) from exc

    try:
        parameters = inspect.signature(flash_attn_with_kvcache).parameters
    except (TypeError, ValueError):
        # Some compiled callables do not expose a Python signature.  The actual
        # invocation remains the authoritative compatibility check.
        return flash_attn_with_kvcache

    required = {
        "q",
        "k_cache",
        "v_cache",
        "k",
        "v",
        "cache_seqlens",
        "softmax_scale",
        "causal",
        "window_size",
    }
    missing = sorted(required.difference(parameters))
    if missing:
        raise RuntimeError(
            "flash_attn_with_kvcache has an incompatible signature; missing " + ", ".join(missing)
        )
    return flash_attn_with_kvcache


def inspect_flash_kvcache() -> FlashKVCacheStatus:
    try:
        _load_flash_kvcache()
    except Exception as exc:
        return FlashKVCacheStatus(False, str(exc))
    return FlashKVCacheStatus(True)


def _window_size(sliding_window: int | None) -> tuple[int, int]:
    # Qwen allows keys with key_position > query_position - window.  FA2's
    # inclusive left window therefore uses window - 1.
    return (-1, -1) if sliding_window is None else (max(0, int(sliding_window) - 1), 0)


@lru_cache(maxsize=1)
def _static_cache_class():
    try:
        from transformers import StaticCache

        return StaticCache
    except ImportError:
        return None


def _is_static_cache(cache: Any) -> bool:
    if cache is None:
        return False
    static_cache = _static_cache_class()
    return (
        isinstance(cache, static_cache)
        if static_cache is not None
        else cache.__class__.__name__ == "StaticCache"
    )


class FlashKVCacheAttention:
    """Wrap one Qwen attention module without changing global registrations."""

    def __new__(cls, inner, *, multimodal_rope: bool, flash_fn: Callable[..., Any]):
        import torch.nn as nn

        class _Adapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = inner
                self.multimodal_rope = multimodal_rope
                self.flash_fn = flash_fn
                self._q3tts_flash_kvcache_adapter = True

            @property
            def layer_idx(self):
                return self.inner.layer_idx

            def forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                cache_position=None,
                **kwargs,
            ):
                if not _is_static_cache(past_key_values):
                    return self.inner(
                        hidden_states=hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        **kwargs,
                    )
                return self._static_forward(
                    hidden_states, position_embeddings, past_key_values, cache_position
                )

            def _static_forward(self, hidden_states, position_embeddings, cache, cache_position):
                if cache_position is None:
                    raise RuntimeError("Flash KV-cache decode requires cache_position")
                if hidden_states.shape[0] != 1:
                    raise RuntimeError("Flash CUDA-graph decode currently supports batch size 1")

                import torch
                from qwen_tts.core.models.modeling_qwen3_tts import (
                    apply_multimodal_rotary_pos_emb,
                    apply_rotary_pos_emb,
                )

                attn = self.inner
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn.head_dim)
                query = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                key = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                value = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                if self.multimodal_rope:
                    rope = attn.rope_scaling
                    query, key = apply_multimodal_rotary_pos_emb(
                        query,
                        key,
                        cos,
                        sin,
                        rope["mrope_section"],
                        rope["interleaved"],
                    )
                else:
                    query, key = apply_rotary_pos_emb(query, key, cos, sin)

                layer = cache.layers[attn.layer_idx]
                if not getattr(layer, "is_initialized", False):
                    raise RuntimeError("StaticCache layer was not initialized before graph capture")

                # flash_attn_with_kvcache expects [batch, sequence, heads, dim].
                # The transposed StaticCache view retains a contiguous last
                # dimension and avoids a cache-sized copy on every layer.
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                key_cache = layer.keys.transpose(1, 2)
                value_cache = layer.values.transpose(1, 2)

                # cache_position is the old cache length.  Passing k/v lets FA2
                # fuse the fixed-cache write with attention.  Keep sequence
                # length on CUDA so replay does not need a host scalar.
                cache_seqlens = cache_position[:1].to(dtype=torch.int32)
                output = self.flash_fn(
                    query,
                    key_cache,
                    value_cache,
                    k=key,
                    v=value,
                    cache_seqlens=cache_seqlens,
                    softmax_scale=attn.scaling,
                    causal=True,
                    window_size=_window_size(getattr(attn, "sliding_window", None)),
                )
                output = output.reshape(*input_shape, -1).contiguous()
                return attn.o_proj(output), None

        return _Adapter()


def install_flash_kvcache_attention(talker_model, predictor_model) -> int:
    """Install instance-local StaticCache adapters and return the layer count."""

    flash_fn = _load_flash_kvcache()
    count = 0
    for layer in talker_model.layers:
        layer.self_attn = FlashKVCacheAttention(
            layer.self_attn, multimodal_rope=True, flash_fn=flash_fn
        )
        count += 1
    for layer in predictor_model.layers:
        layer.self_attn = FlashKVCacheAttention(
            layer.self_attn, multimodal_rope=False, flash_fn=flash_fn
        )
        count += 1
    return count


def uninstall_flash_kvcache_attention(talker_model, predictor_model) -> int:
    """Restore original modules after an auto-mode graph fallback."""

    count = 0
    for model in (talker_model, predictor_model):
        for layer in model.layers:
            attention = layer.self_attn
            if getattr(attention, "_q3tts_flash_kvcache_adapter", False):
                layer.self_attn = attention.inner
                count += 1
    return count


def graph_classes():
    """Create graph subclasses lazily so importing the node pack stays cheap."""

    import torch
    from faster_qwen3_tts.predictor_graph import PredictorGraph
    from faster_qwen3_tts.talker_graph import TalkerGraph

    class FlashPredictorGraph(PredictorGraph):
        def _build_attention_masks(self):
            # The StaticCache adapter derives causality from cache_position and
            # FA2 cache lengths.  A dict bypasses Transformers mask synthesis.
            self.prefill_attn = {"full_attention": None, "sliding_attention": None}
            self.decode_attn = [
                {"full_attention": None, "sliding_attention": None}
                for _ in self.decode_cache_positions
            ]

        @torch.inference_mode()
        def run(self, pred_input):
            self.input_buf.copy_(pred_input)
            self.static_cache.reset()
            self.graph.replay()
            # The caller consumes these tokens before the next replay.  Avoid a
            # device allocation and copy for every acoustic frame.
            return self.output_tokens

    class FlashTalkerGraph(TalkerGraph):
        def _build_attention_masks(self, attention_mask=None):
            # A single placeholder preserves the Qwen model-forward contract.
            # The per-layer adapter does not consume it, so there is no
            # max_seq_len table and no per-token mask copy.
            self.attn_mask_table = None
            if self.attn_mask is None:
                self.attn_mask = torch.zeros(
                    1, 1, 1, self.max_seq_len, dtype=self.dtype, device=self.device
                )
            else:
                self.attn_mask.zero_()

        def _set_attention_mask(self, position: int):
            return None

        def set_generation_state(self, attention_mask, rope_deltas):
            if attention_mask is not None and not bool(torch.all(attention_mask != 0).item()):
                raise RuntimeError(
                    "Flash CUDA-graph decode requires a padding-free batch-1 prompt; "
                    "use SDPA for padded inputs"
                )
            if self.attn_mask is None:
                self._build_attention_masks()
            if rope_deltas is None:
                self.rope_deltas.zero_()
            else:
                if rope_deltas.dim() == 1:
                    rope_deltas = rope_deltas.unsqueeze(1)
                self.rope_deltas.copy_(
                    rope_deltas.to(self.rope_deltas.device, dtype=self.rope_deltas.dtype)
                )

    return FlashPredictorGraph, FlashTalkerGraph
