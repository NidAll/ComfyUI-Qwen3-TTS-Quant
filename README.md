# ComfyUI-Qwen3-TTS-Quant

A native ComfyUI runtime and streaming quantization toolkit for
[`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign).

The project runs the latency-critical Talker and Code Predictor through **W8A8 ConvRot Comfy
Kitchen CUDA kernels**, keeps quantized weights packed during inference, and supports captured
FlashAttention 2 KV-cache decode. The speech decoder remains FP16 to preserve reconstruction
quality.

> **v0.1.6:** FlashAttention 2 and CUDA graphs now work together through an instance-local
> `flash_attn_with_kvcache` adapter. Local validation reached **2.79 to 2.84× realtime** during
> acoustic generation and approximately **2.52× realtime end to end** after graph warmup.

## Validated performance

The following measurements come from repeated VoiceDesign generations with W8A8 ConvRot, BF16
transformer compute, an FP16 codec, FlashAttention 2, and captured CUDA graphs.

| Run | Audio produced | Acoustic generation | Acoustic speed | Full ComfyUI prompt | End-to-end speed | End-to-end RTF |
|---|---:|---:|---:|---:|---:|---:|
| Initial capture | 15.92 s | 5.61 s | 2.84× | 10.24 s | 1.55× | 0.643 |
| Warm run | 26.00 s | 9.15 s | 2.84× | 10.32 s | 2.52× | 0.397 |
| Warm run | 16.42 s | 5.88 s | 2.79× | 6.52 s | 2.52× | 0.397 |

The initial run includes CUDA graph capture and startup work. Warm runs reuse the model, graph
pools, static KV cache, and persistent decode buffers. Results vary with prompt length, generated
duration, software versions, GPU, and power limits.

The fastest validated configuration in the current development run is:

```text
Quantization       W8A8 ConvRot
Transformer dtype  BF16
Codec dtype        FP16
Attention          FlashAttention 2 KV-cache
CUDA graphs        Enabled
Memory profile     Speed
Batch size         1
```

## Current scope

| Capability | Status |
|---|---|
| Qwen3-TTS 1.7B VoiceDesign | Supported |
| W8A8 ConvRot checkpoints | Supported |
| Comfy Kitchen CUDA execution | Required |
| FlashAttention 2 graph decode | Supported with feature detection |
| SDPA graph decode | Supported fallback |
| Batch-1 generation | Optimized path |
| Other Qwen3-TTS 1.7B variants | Planned |
| Additional quantization formats | Planned and benchmark-gated |

## Highlights

- Streaming/sharded quantization from a Hugging Face ID, snapshot directory, safetensors file, safetensors index, or sharded safetensors directory
- Shape-aware `Qwen3TTSVoiceDesignAdapter`
- Safetensors-native format and architecture metadata
- Packed Comfy Kitchen INT8 linear modules
- CPU selective-gather text embedding
- Static Talker KV cache with SDPA and FlashAttention 2 CUDA graph paths
- Fused FlashAttention KV-cache updates for fixed-shape Talker and Predictor decode
- Persistent codec history and Predictor input buffers
- Generation-wide Comfy Kitchen backend dispatch
- Explicit ComfyUI VRAM-pressure integration without ComfyUI core changes
- Loader, Voice Design, Runtime Settings, Diagnostics, one-shot Benchmark, and Unload nodes
- Deterministic seed and generation telemetry

Read [docs/DESIGN.md](docs/DESIGN.md) for the quantization rationale, execution design, and
compatibility boundary.

## Installation

Use the Python executable that launches your ComfyUI installation. The commands below assume the
repository is installed directly under `custom_nodes`.

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/NidAll/ComfyUI-Qwen3-TTS-Quant.git
cd ComfyUI-Qwen3-TTS-Quant

/path/to/ComfyUI/venv/bin/python -m pip install \
  --upgrade-strategy only-if-needed -r requirements.txt

/path/to/ComfyUI/venv/bin/python -m pip install --no-deps -e .
```

To update an existing installation:

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-Qwen3-TTS-Quant
git pull --ff-only
/path/to/ComfyUI/venv/bin/python -m pip install --no-deps -e .
```

The requirements intentionally do not pin Torch, torchvision, torchaudio, CUDA, or Transformers. Confirm that pip did not propose replacing Torch before accepting any environment change. `faster-qwen3-tts 0.4.x` requires Transformers 5.15.1 or newer.

Comfy Kitchen must already be importable from the same Python environment:

```bash
/path/to/venv/bin/python -c "import torch, comfy_kitchen; print(torch.__version__, torch.version.cuda); print(comfy_kitchen.list_backends())"
```

Restart ComfyUI after installation. No ComfyUI core file is edited.

## Quantize the model

Run conversion inside the ComfyUI environment so it uses the installed Comfy Kitchen CUDA backend:

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-Qwen3-TTS-Quant
/path/to/venv/bin/python -m qwen3_tts_quantizer \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --output /path/to/ComfyUI/models/qwen3_tts_quant/Qwen3-TTS-12Hz-1.7B-VoiceDesign-W8A8-ConvRot \
  --format auto \
  --device cuda \
  --kernel-backend cuda \
  --calibration auto \
  --max-shard-mib 768 \
  --verbose
```

`--format auto` currently resolves to the one selected target, `w8a8-convrot`. It does not claim that an unmeasured format matrix was run. This format uses dynamic activation quantization, so no calibration corpus is required.

For a local source:

```bash
/path/to/venv/bin/python -m qwen3_tts_quantizer \
  --model /models/Qwen3-TTS-VoiceDesign \
  --output /path/to/ComfyUI/models/qwen3_tts_quant/VoiceDesign-W8A8-ConvRot \
  --format w8a8-convrot --device cuda --kernel-backend cuda
```

For one safetensors file or an index whose config assets live elsewhere, add `--config-dir /path/to/model-config-directory`.

Conversion reads one source tensor at a time. It does not load the original model into VRAM as a complete dense model. Essential recognition metadata is embedded into every output safetensors shard; `quantization_manifest.json` is diagnostic.

The writer clones serialized tensors before saving. This is required because the repeated Comfy quantization marker intentionally has identical contents, while safetensors rejects aliased storage.

## ComfyUI workflow

1. Copy or quantize the output directory under `ComfyUI/models/qwen3_tts_quant/`.
2. Restart ComfyUI so the loader list refreshes.
3. Add `Qwen3 TTS Runtime Settings` and retain the defaults.
4. Connect it to `Qwen3 TTS Quantized Model Loader`.
5. Select the converted model directory.
6. Connect `Qwen3 TTS Voice Design`, then connect its `AUDIO` output to ComfyUI’s `SaveAudio` node.

An importable starting workflow is in [examples/voice_design_workflow.json](examples/voice_design_workflow.json).

## Recommended starting settings

| Setting | Value | Reason |
|---|---|---|
| Compute dtype | `bfloat16` | Stable transformer interface on supported CUDA devices |
| Codec dtype | `float16` | Lower decoder VRAM and strong CUDA throughput |
| Attention | `flash_attention_2` | Fastest path in the current validated W8A8 ConvRot runs |
| Kernel backend | `cuda` | Selects Comfy Kitchen compiled CUDA kernels explicitly |
| CUDA graphs | `auto` | Uses graphs when capture succeeds and records any fallback |
| Memory profile | `speed` | Keeps Talker, Predictor, KV, and codec GPU resident |
| Maximum sequence | `2048` | About 224 MiB for the 28-layer Talker KV cache |
| Safety headroom | `320 MiB` | Avoids operating at the physical VRAM edge |

`cuda_graphs=on` is strict and fails if capture is incompatible. `auto` records `cuda_graph_error` in the Diagnostics node before using the upstream path. The runtime never reports graphs as active after a failed capture.

For the compatibility path, use `attention_backend=sdpa`. For the fastest validated path, use
`flash_attention_2` with `cuda_graphs=auto` first. After one successful generation,
the Diagnostics node must show both:

```json
{
  "attention_execution_path": "flash_kvcache_cuda_graph",
  "cuda_graphs_active": true
}
```

Then `cuda_graphs=on` can be used as a strict setting. Dynamic prefill continues to use the
Transformers FlashAttention 2 integration. Fixed-shape Talker and Code Predictor decode use
`flash_attn_with_kvcache`, with the cache update fused into attention and the sequence length kept
in a CUDA tensor for graph replay. The adapter is installed only on this model instance and does
not alter Transformers or ComfyUI globally.

## How the v0.1.6 FlashAttention graph path works

The earlier graph integration supplied a fixed 4D additive StaticCache mask to Transformers'
generic FlashAttention adapter. That adapter interpreted the tensor as a padding mask and entered
the variable-length unpadding path, which could produce an out-of-bounds CUDA index during graph
warmup.

Version 0.1.6 separates the two attention workloads:

- Variable-length prompt prefill stays on the normal Transformers FlashAttention 2 implementation.
- Fixed-shape Talker and Code Predictor graph decode call `flash_attn_with_kvcache` directly.
- New K/V values are written into the preallocated cache inside the attention kernel.
- Cache length remains an INT32 CUDA tensor, allowing it to advance during graph replay.
- RoPE remains inside Qwen's model implementation, preserving its multimodal position semantics.
- Attention wrappers belong only to the loaded model instance. ComfyUI and Transformers remain untouched globally.

The graph runtime also removes the Talker's per-position mask table and per-frame mask copy. This
reduces setup memory and recurring launch work.

You can validate the installed FlashAttention kernel and the exact transposed StaticCache layout
without loading the TTS checkpoint:

```bash
/path/to/venv/bin/python benchmarks/flash_kvcache_smoke.py --dtype bfloat16
```

Both query-length cases should report `"passed": true`. Query length 1 represents Talker decode;
query length 2 represents Predictor prefill inside its captured 15-step loop.

The Predictor graph captures its sampling parameters on first generation. Reload the model to change temperature, top-k, or top-p after capture, or select graphs `off`.

## Telemetry and one-shot benchmark

The Voice Design node returns JSON alongside audio:

```json
{
  "audio_seconds": 8.25,
  "total_seconds": 5.10,
  "rtf": 0.618,
  "realtime_multiplier": 1.618,
  "peak_vram_mib": 0.0,
  "steady_vram_mib": 0.0,
  "graph_active": true,
  "acoustic_steps": 99,
  "prefill_seconds": 0.12,
  "decode_seconds": 3.70,
  "codec_seconds": 0.84,
  "acoustic_frames_per_second": 26.76
}
```

The zero values in this schema example are populated with measurements at runtime. Use `Qwen3 TTS One-Shot Benchmark` for one warmed generation. `RTF = generation seconds / audio seconds`; lower is faster. The reciprocal is reported as `realtime_multiplier`.

To compare only the two practical graph paths with the same prompt and seed:

```bash
/path/to/venv/bin/python benchmarks/one_shot.py \
  --checkpoint /path/to/VoiceDesign-W8A8-ConvRot \
  --attention sdpa --cuda-graphs on --max-new-tokens 512

/path/to/venv/bin/python benchmarks/one_shot.py \
  --checkpoint /path/to/VoiceDesign-W8A8-ConvRot \
  --attention flash_attention_2 --cuda-graphs on --max-new-tokens 512
```

Compare `rtf`, `acoustic_frames_per_second`, `prefill_seconds`, `codec_seconds`, and peak VRAM.
This targeted comparison isolates the two practical attention paths without running a complete
quantization matrix. Runtime performance still decides the result because single-query attention
can represent a modest share of total W8A8 autoregressive time.

## Memory behavior

Before load, the compatibility adapter calls ComfyUI’s public `free_memory` function to release competing models. It does not modify core files or install global monkey patches. The autoregressive transformer stays resident for the lifetime of the loader output. The Unload node releases model references, graph pools, and CUDA cache explicitly.

This first speed release intentionally implements only the `speed` profile. Codec scheduling and partial residency remain deferred because moving the codec while retaining graph pools can increase peak memory or introduce synchronization, which conflicts with the requested fastest RTF.

## Troubleshooting

**Prompt validation reports `top_p` > 1, `temperature` > 2, or `max_new_tokens` < 2:**
This is a stale ComfyUI widget-order issue, not a model-download or CUDA error. ComfyUI stores
custom-node widget values positionally in a workflow. If a workflow was saved with an earlier
revision of this node, values can be presented under the wrong names (for example, repetition
penalty `1.05` being read as `top_p`, or top-k `50` being read as `temperature`). Stop ComfyUI,
install the current pack, restart it, then delete and re-add the **Qwen3 TTS Voice Design** node
before reconnecting the model. Do not reuse the old node instance. Use these valid starter values:

```text
seed                 42
max_new_tokens       1024
temperature          0.90
top_k                50
top_p                1.00
repetition_penalty   1.05
```

The supplied `examples/voice_design_workflow.json` contains this current ordering. Save the
workflow again after recreating the node. This does not require changing any ComfyUI core file.

**`sox: command not found`:** Install the SoX system utility with your operating system's package manager, then verify it is on `PATH`:

```bash
command -v sox
```

The VoiceDesign path does not use reference-audio extraction, so SoX is not on the critical
generation path. Installing it removes the warning and keeps future voice-clone/audio-preprocess
paths functional.

**`mat1 and mat2 must have the same dtype` or CUDA graph warmup reports a BF16/Float mismatch:**
Install the current pack revision and restart ComfyUI. The loader now normalizes all floating
Talker parameters and buffers to the selected compute dtype while preserving INT8 packed weights
and FP32 quantization scales. This covers the unquantized `text_projection` layers and the static
KV-cache capture path.

**Sample-rate fallback warning:** Qwen3-TTS `12Hz` is the codec frame rate; generated waveforms
are normally 24 kHz. The current loader reads `output_sample_rate` from the speech-tokenizer
configuration before constructing the fast wrapper.

**No checkpoint appears:** Place the entire converted directory, including `config.json`, shards, index, processor assets, and `speech_tokenizer/`, under `models/qwen3_tts_quant/`, then restart ComfyUI.

**Comfy Kitchen CUDA backend unavailable:** Run the import check above. The Diagnostics node reports the installed package version, GPU capability, and backend registry. This runtime refuses to label an eager or Triton fallback as CUDA.

**Out of memory before generation:** Close other GPU applications, unload diffusion models, reduce maximum sequence length to 1024, or increase available VRAM. A 1024-token Talker cache is about 112 MiB. The runtime rejects an unsafe estimated plan before generation.

**CUDA graph capture fails:** Keep `cuda_graphs=auto` for an explicit recorded fallback, or set it to `off`. Use `on` only after one successful auto capture.

**FlashAttention 2 with CUDA graphs:** Version 0.1.6 no longer feeds the graph's 4D additive mask
to FlashAttention's variable-length unpadding code. Static-cache decode instead uses
`flash_attn_with_kvcache`. This path currently requires batch size 1, a padding-free prompt,
FP16/BF16 compute, and a compatible FlashAttention 2 build. If the function is absent, `auto`
uses dynamic FA2 and records the reason; `on` fails before capture. If a previous version caused a
CUDA device-side assertion, fully restart ComfyUI before testing this version because that CUDA
context cannot be recovered safely inside the same process.

**FlashAttention unavailable:** Use `sdpa`. Selecting `flash_attention_2` delegates to Transformers and fails visibly when FA2 is unavailable or incompatible.

**Transformers reports that FlashAttention was selected without a dtype:** The loader constructs
the architecture before materializing checkpoint tensors, so this warning can appear during model
creation. The runtime subsequently normalizes floating Talker tensors to the selected compute dtype.
Confirm the final value through `compute_dtype` in the Diagnostics node.

**Transformers reports a `qwen3_tts` model type mismatch with an empty target type:** This message
comes from speech-tokenizer configuration loading in the current upstream stack. Successful loading
of all tokenizer weights followed by normal waveform generation confirms the compatible path.

**Model was unloaded:** Execute the loader again. The explicit Unload node invalidates that handle by design.

## Tests

CPU-only policy and metadata tests:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

Policy, metadata, architecture, and feature-detection tests run without CUDA. The INT8 kernel test
requires Comfy Kitchen CUDA. FlashAttention KV-cache graph capture and full VoiceDesign generation
have also been validated through repeated local runs; the performance table above reports those
measurements directly.

## Updating safely

Update ComfyUI and Comfy Kitchen normally, without copying files from this project into `ComfyUI/comfy/`. After an update, run the backend import check and one short Diagnostics/Benchmark workflow. The custom checkpoint metadata contains its format version and source fingerprint; incompatible formats fail explicitly.

## License

Apache-2.0. Upstream Qwen3-TTS, ComfyUI, Comfy Kitchen, and faster-qwen3-tts retain their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
