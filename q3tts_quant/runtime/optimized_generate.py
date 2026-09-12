from __future__ import annotations

"""Allocation-conscious VoiceDesign loop for the W8A8 ConvRot graph runtime.

The control flow follows faster-qwen3-tts' graph API, while keeping codec
history in a preallocated tensor.  This removes the growing per-step stack and
the clone of the Talker graph output without changing sampling semantics.
See THIRD_PARTY_NOTICES.md.
"""

import logging
import time
from dataclasses import dataclass

log = logging.getLogger("q3tts.generate")


@dataclass
class DecodeWorkspace:
    codec_ids: object
    pred_input: object
    suppress_mask: object
    capacity: int
    vocab_size: int
    num_code_groups: int


def _workspace(talker, *, capacity: int, vocab_size: int, num_code_groups: int, hidden_size: int, device, dtype):
    import torch

    current = getattr(talker, "_q3tts_decode_workspace", None)
    compatible = (
        current is not None
        and current.capacity >= capacity
        and current.vocab_size == vocab_size
        and current.num_code_groups == num_code_groups
        and current.codec_ids.device == device
        and current.pred_input.dtype == dtype
    )
    if compatible:
        return current

    suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    suppress_start = max(0, vocab_size - 1024)
    suppress_mask[suppress_start:] = True
    workspace = DecodeWorkspace(
        codec_ids=torch.empty(capacity, num_code_groups, dtype=torch.long, device=device),
        pred_input=torch.empty(1, 2, hidden_size, dtype=dtype, device=device),
        suppress_mask=suppress_mask,
        capacity=capacity,
        vocab_size=vocab_size,
        num_code_groups=num_code_groups,
    )
    setattr(talker, "_q3tts_decode_workspace", workspace)
    return workspace


def fast_generate_w8a8(
    *,
    talker,
    talker_input_embeds,
    attention_mask,
    trailing_text_hiddens,
    tts_pad_embed,
    config,
    predictor_graph,
    talker_graph,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float,
):
    import torch
    from faster_qwen3_tts.sampling import apply_repetition_penalty, sample_logits

    eos_id = int(config.codec_eos_token_id)
    num_code_groups = int(config.num_code_groups)
    vocab_size = int(config.vocab_size)
    device = talker_input_embeds.device
    predictor = talker.code_predictor
    talker_codec_embed = talker.get_input_embeddings()
    talker_codec_head = talker.codec_head
    predictor_codec_embeds = predictor.get_input_embeddings()
    workspace = _workspace(
        talker,
        capacity=max_new_tokens,
        vocab_size=vocab_size,
        num_code_groups=num_code_groups,
        hidden_size=talker_input_embeds.shape[-1],
        device=device,
        dtype=talker_input_embeds.dtype,
    )
    workspace.suppress_mask[eos_id] = False

    started = time.perf_counter()
    output = talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        generation_step=None,
        past_hidden=None,
        past_key_values=None,
    )
    past_hidden = output.past_hidden
    generation_step = output.generation_step
    logits = output.logits[:, -1, :]
    token = sample_logits(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        suppress_mask=workspace.suppress_mask,
        suppress_tokens=[eos_id] if min_new_tokens > 0 else None,
    )

    prefill_len = talker_graph.prefill_kv(output.past_key_values)
    talker_graph.set_generation_state(attention_mask, getattr(talker, "rope_deltas", None))
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started

    decode_started = time.perf_counter()
    steps = 0
    for step_idx in range(max_new_tokens):
        # EOS is a data-dependent exit and therefore remains the one required
        # host synchronization in each outer acoustic-frame iteration.
        if token.item() == eos_id:
            break

        last_id_hidden = talker_codec_embed(token.unsqueeze(1))
        workspace.pred_input[:, :1].copy_(past_hidden)
        workspace.pred_input[:, 1:].copy_(last_id_hidden)
        codebook_tokens = predictor_graph.run(workspace.pred_input)

        workspace.codec_ids[step_idx, 0].copy_(token[0])
        workspace.codec_ids[step_idx, 1:].copy_(codebook_tokens)
        steps = step_idx + 1

        # Preserve the upstream cat + reduction calculation.  Changing this to
        # serialized in-place adds is a separate kernel-level benchmark.
        codec_hiddens = [last_id_hidden]
        for index in range(num_code_groups - 1):
            codec_hiddens.append(
                predictor_codec_embeds[index](codebook_tokens[index].reshape(1, 1))
            )
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)

        if generation_step < trailing_text_hiddens.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hiddens[:, generation_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed

        current_position = prefill_len + step_idx
        if current_position >= talker_graph.max_seq_len - 1:
            break
        hidden_states = talker_graph.run(inputs_embeds, position=current_position)
        logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)

        if repetition_penalty != 1.0:
            # A view over the preallocated first-codebook column replaces an
            # O(sequence) torch.stack allocation on every decode iteration.
            history = workspace.codec_ids[:steps, 0]
            logits = apply_repetition_penalty(logits, history, repetition_penalty)

        token = sample_logits(
            logits.squeeze(0),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            suppress_mask=workspace.suppress_mask,
            suppress_tokens=[eos_id] if steps < min_new_tokens else None,
        )
        # PredictorGraph copies this view before TalkerGraph overwrites its
        # output buffer on the following iteration, so a clone is unnecessary.
        past_hidden = hidden_states[:, -1:, :]
        generation_step += 1

    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    timing = {
        "prefill_ms": prefill_seconds * 1000,
        "decode_s": decode_seconds,
        "steps": steps,
        "ms_per_step": (decode_seconds / steps * 1000) if steps else 0.0,
        "steps_per_s": (steps / decode_seconds) if decode_seconds else 0.0,
    }
    if not steps:
        return None, timing
    return workspace.codec_ids[:steps], timing


def optimized_wrapper_class():
    """Build lazily so importing ComfyUI nodes does not require the runtime."""

    import numpy as np
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    class OptimizedFasterQwen3TTS(FasterQwen3TTS):
        @torch.inference_mode()
        def generate_voice_design(
            self,
            text: str,
            instruct: str,
            language: str,
            non_streaming_mode=None,
            max_new_tokens: int = 2048,
            min_new_tokens: int = 2,
            temperature: float = 0.9,
            top_k: int = 50,
            top_p: float = 1.0,
            do_sample: bool = True,
            repetition_penalty: float = 1.05,
        ):
            if self.model.model.tts_model_type != "voice_design":
                raise ValueError("Loaded model does not support voice design generation")
            self.model._validate_languages([language])
            non_streaming_mode = self._resolve_non_streaming_mode(non_streaming_mode, default=True)
            model, talker, config, embeds, mask, trailing, pad = self._prepare_generation_custom(
                text=text,
                language=language,
                speaker=None,
                instruct=instruct,
                non_streaming_mode=non_streaming_mode,
            )
            codec_ids, timing = fast_generate_w8a8(
                talker=talker,
                talker_input_embeds=embeds,
                attention_mask=mask,
                trailing_text_hiddens=trailing,
                tts_pad_embed=pad,
                config=config,
                predictor_graph=self.predictor_graph,
                talker_graph=self.talker_graph,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
            )
            if codec_ids is None:
                log.warning("generation returned no acoustic tokens")
                self._last_timing = timing
                return [np.zeros(1, dtype=np.float32)], self.sample_rate

            codec_started = time.perf_counter()
            audio_list, sample_rate = model.speech_tokenizer.decode(
                {"audio_codes": codec_ids.unsqueeze(0)}
            )
            torch.cuda.synchronize()
            timing["codec_s"] = time.perf_counter() - codec_started
            self._last_timing = timing
            audio_arrays = []
            for audio in audio_list:
                if hasattr(audio, "cpu"):
                    audio_arrays.append(audio.flatten().cpu().numpy())
                else:
                    audio_arrays.append(audio.flatten() if hasattr(audio, "flatten") else audio)
            audio_duration = timing["steps"] / 12.0
            total = timing["prefill_ms"] / 1000 + timing["decode_s"]
            log.info(
                "generated %.2fs audio in %.2fs (%.1fms/frame, %.2fx realtime)",
                audio_duration,
                total,
                timing["ms_per_step"],
                audio_duration / total if total else 0.0,
            )
            return audio_arrays, sample_rate

    return OptimizedFasterQwen3TTS
