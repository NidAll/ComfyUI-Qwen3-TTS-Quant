# ComfyUI-Qwen3-TTS-Quant

A native ComfyUI node pack and streaming checkpoint converter for `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`, designed for fast batch-1 generation on CUDA devices with constrained VRAM.

The selected first configuration is **W8A8 ConvRot through Comfy Kitchen CUDA**, with a BF16 Talker interface, FP16 speech decoder, CPU-resident text embedding, static KV cache, and optional CUDA graphs. Quantized weights remain INT8 throughout loading and inference.

## What is included

- Streaming/sharded quantization from a Hugging Face ID, snapshot directory, safetensors file, safetensors index, or sharded safetensors directory
- Shape-aware `Qwen3TTSVoiceDesignAdapter`
- Safetensors-native format and architecture metadata
- Packed Comfy Kitchen INT8 linear modules
- CPU selective-gather text embedding
- Static Talker KV cache and graph wrappers from `faster-qwen3-tts`
- Explicit ComfyUI VRAM-pressure integration without ComfyUI core changes
- Loader, Voice Design, Runtime Settings, Diagnostics, one-shot Benchmark, and Unload nodes
- Deterministic seed and generation telemetry

Read [docs/DESIGN.md](docs/DESIGN.md) for the technical selection rationale and compatibility boundary.

## Installation

Use the Python executable that launches your native ComfyUI installation. The commands below assume the repository is installed directly under `custom_nodes`.

```bash
cd /path/to/ComfyUI/custom_nodes
unzip /path/to/ComfyUI-Qwen3-TTS-Quant.zip
cd ComfyUI-Qwen3-TTS-Quant
/path/to/ComfyUI/python -m pip install --upgrade-strategy only-if-needed -r requirements.txt
/path/to/ComfyUI/python -m pip install --no-deps -e .
```

For a normal native virtual environment, `/path/to/ComfyUI/python` is usually `/path/to/venv/bin/python`.

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
| Attention | `sdpa` | Reliable default for short-query decode; test FA2 later if desired |
| Kernel backend | `cuda` | Selects Comfy Kitchen compiled CUDA kernels explicitly |
| CUDA graphs | `auto` | Uses graphs when capture succeeds and records any fallback |
| Memory profile | `speed` | Keeps Talker, Predictor, KV, and codec GPU resident |
| Maximum sequence | `2048` | About 224 MiB for the 28-layer Talker KV cache |
| Safety headroom | `320 MiB` | Avoids operating at the physical VRAM edge |

`cuda_graphs=on` is strict and fails if capture is incompatible. `auto` records `cuda_graph_error` in the Diagnostics node before using the upstream path. The runtime never reports graphs as active after a failed capture.

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
  "graph_active": true
}
```

The zero values in this schema example are populated with measurements at runtime. Use `Qwen3 TTS One-Shot Benchmark` for one warmed generation. `RTF = generation seconds / audio seconds`; lower is faster. The reciprocal is reported as `realtime_multiplier`.

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

**CUDA graph capture fails:** Keep `cuda_graphs=auto` for an explicit recorded fallback, or set it to `off`. Use `on` only while diagnosing the capture path.

**FlashAttention 2 with CUDA graphs:** The current `faster-qwen3-tts` graph path uses a fixed
4D StaticCache mask. FlashAttention 2's unpadding implementation expects a different mask layout
and can trigger an asynchronous CUDA device-side index assertion during predictor warmup. The
loader now detects this combination before graph capture. With `attention_backend=flash_attention_2`
and `cuda_graphs=auto`, it runs the real Transformers FlashAttention path with dynamic caching and
reports `cuda_graphs_active=false`. `cuda_graphs=on` fails immediately with an actionable error.
For the lowest RTF on a constrained CUDA device, start with `attention_backend=sdpa` plus
`cuda_graphs=auto`; that is the stable static-cache path. Confirm the choice with the one-shot
benchmark on your own device.

**FlashAttention unavailable:** Use `sdpa`. Selecting `flash_attention_2` delegates to Transformers and fails visibly when FA2 is unavailable or incompatible.

**Model was unloaded:** Execute the loader again. The explicit Unload node invalidates that handle by design.

## Tests

CPU-only policy and metadata tests:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

The actual INT8 kernel, graph-capture, VRAM, deterministic audio, and RTF tests require the target ComfyUI CUDA environment and the converted checkpoint. The current artifact has not fabricated results for hardware it could not access.

## Updating safely

Update ComfyUI and Comfy Kitchen normally, without copying files from this project into `ComfyUI/comfy/`. After an update, run the backend import check and one short Diagnostics/Benchmark workflow. The custom checkpoint metadata contains its format version and source fingerprint; incompatible formats fail explicitly.

## License

Apache-2.0. Upstream Qwen3-TTS, ComfyUI, Comfy Kitchen, and faster-qwen3-tts retain their own licenses.
