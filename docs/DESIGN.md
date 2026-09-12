# Runtime design and quantization decision

## Selected first format

The target format is heterogeneous **W8A8 ConvRot**:

| Component | Storage / compute | Residency |
|---|---|---|
| Talker attention and MLP linears | INT8 weights, dynamically quantized INT8 activations, BF16 output, ConvRot 256 | GPU resident |
| Code Predictor attention and MLP linears | INT8 weights, dynamically quantized INT8 activations, BF16 output, ConvRot 256 | GPU resident |
| Norms, embeddings, codec heads, predictor heads | Source precision, BF16 at runtime | GPU, except text table |
| Text embedding | BF16 selective CPU gather | CPU |
| 12 Hz speech tokenizer / decoder | FP16 | GPU resident |
| Talker KV cache | FP16/BF16 static cache | GPU |

This selection follows the actual batch-1 decode shapes. Current Comfy Kitchen CUDA has a fused M=1 ConvRot INT8 path for K dimensions divisible by 256. The VoiceDesign recurrent linears use K=1024, 2048, 3072, and 6144, which all satisfy that path. Plain W8A8 does not cover the same set through its most specialized M=1 route. W4A16 remains a strong future challenger, but it needs an on-device comparison at these exact shapes before replacing this default.

## Status by optimization

| Optimization | Status | Treatment |
|---|---|---|
| Comfy Kitchen CUDA W8A8 ConvRot | Confirmed available upstream | Required runtime path |
| Packed INT8 weight residency | Confirmed available | No persistent dense weight copy |
| FP16 codec | Confirmed available | Selected |
| CPU text-embedding gather | Likely viable | Implemented to reclaim about 594 MiB of VRAM |
| Static Talker KV cache | Confirmed available in faster-qwen3-tts | Selected with fixed maximum length |
| Talker and Predictor CUDA graphs | Confirmed available upstream | Auto capture with observable fallback; `on` is strict |
| FlashAttention 2 KV-cache graphs | Likely viable | Instance-local adapter uses fused cache update; requires an on-device validation |
| torch.compile | Requires benchmark | Excluded from the first path due graph overlap and compile memory |
| W4A16 AWQ | Requires benchmark | Deferred |
| W4A8 / W4A4 | Requires quality and speed benchmark | Deferred |
| SageAttention | Not suitable as a default | No demonstrated advantage for this short-query autoregressive path |
| NVMe weight streaming | Not suitable | RAM plus persistent GPU decode weights is faster for this model size |

## Compatibility boundary

The node never edits or patches ComfyUI core. `q3tts_quant.comfy_integration.compat` is the only module that calls ComfyUI APIs. It uses `comfy.model_management.free_memory` before loading and `soft_empty_cache` after explicit unload. The packed TTS engine is intentionally retained as one workflow object because diffusion-style layer-by-layer patching would evict the Talker during decoding.

The implementation was aligned to these inspected revisions:

- QwenLM/Qwen3-TTS: `022e286b98fbec7e1e916cb940cdf532cd9f488e`
- Comfy-Org/ComfyUI: `c40c94e1f5142b236274c01381b0bd6ba7ca8f85`
- Comfy-Org/comfy-kitchen: `21003fa97bf3b180393446d729ae630ceb6c2a52`
- faster-qwen3-tts: `e2a215f61984c0e72a242f8dd72333338e7672f4`

Feature detection and explicit errors protect the load path when these projects evolve.

## FlashAttention graph design

Dynamic prompt prefill remains on Qwen/Transformers. For graph decode, the runtime wraps only the
33 attention instances used by the 28-layer Talker and 5-layer Code Predictor. A StaticCache call
uses FlashAttention 2's `flash_attn_with_kvcache` interface directly. Qwen applies its own RoPE
before the call; FA2 receives the new K/V tensors, the preallocated cache, and an INT32 CUDA
sequence-length tensor. This fuses the cache write with attention and avoids the incompatible
variable-length unpadding path.

The graph subclasses also eliminate the Talker's table of one full attention mask per possible
sequence position and the per-frame mask copy. The Predictor passes explicit no-mask entries because
causality and sliding-window bounds are supplied to the KV-cache kernel. Padded batch input is
rejected explicitly; the optimized VoiceDesign node targets batch size 1.

## W8A8 decode-loop tuning

Comfy Kitchen's backend override is entered once per warmup or generation rather than once per
quantized Linear call. The VoiceDesign loop keeps a persistent codec output/history buffer, reuses
the two-token Predictor input, returns the Predictor's captured output buffer without cloning, and
uses a history view for repetition penalty. Sampling retains the one necessary per-frame EOS host
check, so generated length and deterministic seed behavior remain unchanged.
