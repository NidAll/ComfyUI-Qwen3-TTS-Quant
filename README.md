# ComfyUI-Qwen3-TTS-Quant

**Fast, low-VRAM Qwen3-TTS VoiceDesign inference for ComfyUI.**

A native ComfyUI node pack and streaming quantization toolkit for:

```text
Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

The project is built around a simple goal:

> Keep the latency-critical Qwen3-TTS autoregressive path quantized, GPU-resident, and CUDA-optimized while minimizing VRAM pressure.

The current optimized runtime uses **W8A8 ConvRot through Comfy Kitchen CUDA**, static KV caching, optional CUDA Graphs, selective CPU residency, and explicit memory planning.

Quantized transformer weights remain **INT8 during loading and inference**. The runtime does not reconstruct a persistent dense FP16/BF16 copy of the model.

---

## Highlights

- **W8A8 ConvRot** Talker and Code Predictor
- **Comfy Kitchen CUDA kernels**
- Quantized weights remain **INT8 in VRAM**
- **CUDA Graphs** for repetitive autoregressive decoding
- Static Talker and Predictor KV caches
- FP16 speech decoder
- BF16 transformer interface
- CPU-resident text embedding with selective GPU gather
- Explicit low-VRAM memory planning
- Streaming/sharded model conversion
- Safetensors-native quantized checkpoints
- No ComfyUI core modifications
- Runtime diagnostics and telemetry
- Built-in one-shot benchmarking
- Explicit model unloading and CUDA cleanup

---

# Why this project exists

Qwen3-TTS VoiceDesign is a capable speech-generation model, but the standard dense runtime is difficult to use efficiently on constrained GPUs.

Simply moving parts of the model between CPU and GPU can reduce VRAM usage while severely hurting autoregressive performance.

This project takes a different approach.

The latency-critical path is kept resident:

```text
Text / Voice Instruction
          │
          ▼
   CPU Text Embedding
   selective row gather
          │
          ▼
 ┌───────────────────────┐
 │       Talker          │
 │   W8A8 ConvRot CUDA   │
 │                       │
 │     Static KV Cache   │
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐
 │    Code Predictor     │
 │   W8A8 ConvRot CUDA   │
 │                       │
 │      CUDA Graph       │
 └──────────┬────────────┘
            │
            ▼
      Acoustic Codes
            │
            ▼
 ┌───────────────────────┐
 │  12 Hz Speech Codec   │
 │        FP16           │
 └──────────┬────────────┘
            │
            ▼
           AUDIO
```

The runtime is optimized for **batch-1 autoregressive inference**, where the kernel requirements are very different from large-batch Transformer workloads.

---

# Current optimization strategy

The first production path uses heterogeneous precision:

| Component | Runtime representation |
|---|---|
| Talker attention linears | W8A8 ConvRot |
| Talker MLP linears | W8A8 ConvRot |
| Code Predictor linears | W8A8 ConvRot |
| Transformer output | BF16 |
| Norms / sensitive tensors | source precision / BF16 |
| Text embedding | BF16 on CPU |
| KV cache | FP16 / BF16 static |
| Speech tokenizer / decoder | FP16 |

The recurrent Transformer linear dimensions are compatible with Comfy Kitchen's specialized ConvRot path.

The implementation deliberately avoids assuming that one quantization format is universally optimal.

Future formats will only replace the default after real end-to-end benchmarks.

---

# CUDA acceleration

The runtime can combine:

```text
W8A8 ConvRot
      +
Comfy Kitchen CUDA
      +
Static KV Cache
      +
CUDA Graphs
      +
Persistent GPU Residency
```

CUDA Graphs are particularly useful for Qwen3-TTS because Talker and Code Predictor decoding repeatedly execute small, similar workloads.

Reducing Python dispatch and kernel-launch overhead can therefore matter as much as accelerating an individual matrix multiplication.

---

# Attention backends

Available runtime choices:

```text
auto
sdpa
flash_attention_2
eager
```

## Recommended starting point

```text
SDPA + CUDA Graphs
```

This is currently the most reliable graph-compatible path.

FlashAttention 2 is supported as an optional execution backend, but the current upstream graph implementation uses a fixed 4D `StaticCache` mask while FlashAttention 2's unpadding path expects a different layout.

Because of this:

```text
FlashAttention 2 + CUDA Graphs
```

is detected before graph capture.

With:

```text
cuda_graphs = auto
```

the runtime falls back explicitly and reports that graphs are inactive.

With:

```text
cuda_graphs = on
```

the loader fails immediately with an actionable error instead of silently changing execution behavior.

No backend is reported as active unless it is actually being used.

---

# Installation

Clone directly into your ComfyUI custom nodes directory:

```bash
cd /path/to/ComfyUI/custom_nodes

git clone https://github.com/NidAll/ComfyUI-Qwen3-TTS-Quant.git

cd ComfyUI-Qwen3-TTS-Quant
```

Use the same Python environment that launches ComfyUI:

```bash
/path/to/venv/bin/python -m pip install \
  --upgrade-strategy only-if-needed \
  -r requirements.txt
```

Then install the package itself:

```bash
/path/to/venv/bin/python -m pip install --no-deps -e .
```

Restart ComfyUI.

---

## Dependency policy

This project intentionally avoids forcing replacements for foundational packages such as:

```text
torch
torchvision
torchaudio
CUDA
transformers
```

The goal is to integrate with an existing modern ComfyUI environment rather than rebuild it around the node pack.

Before accepting large dependency changes, inspect what pip intends to modify.

---

# Verify Comfy Kitchen

Comfy Kitchen must be importable from the same Python environment as ComfyUI.

Run:

```bash
/path/to/venv/bin/python - <<'PY'
import torch
import comfy_kitchen

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Comfy Kitchen backends:", comfy_kitchen.list_backends())
PY
```

For NVIDIA CUDA inference, the optimized CUDA backend should be available.

---

# Quantizing Qwen3-TTS VoiceDesign

The project includes a streaming checkpoint converter.

Quantize directly from Hugging Face:

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-Qwen3-TTS-Quant

/path/to/venv/bin/python -m qwen3_tts_quantizer \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --output /path/to/ComfyUI/models/qwen3_tts_quant/Qwen3-TTS-VoiceDesign-W8A8-ConvRot \
  --format auto \
  --device cuda \
  --kernel-backend cuda \
  --calibration auto \
  --max-shard-mib 768 \
  --verbose
```

Currently:

```text
--format auto
```

selects:

```text
w8a8-convrot
```

It does not pretend to have benchmarked formats that have not yet been tested.

---

## Local checkpoints

A local Hugging Face model directory can also be used:

```bash
/path/to/venv/bin/python -m qwen3_tts_quantizer \
  --model /models/Qwen3-TTS-VoiceDesign \
  --output /path/to/ComfyUI/models/qwen3_tts_quant/VoiceDesign-W8A8-ConvRot \
  --format w8a8-convrot \
  --device cuda \
  --kernel-backend cuda
```

Supported source layouts include:

```text
Hugging Face model ID
local Hugging Face snapshot
single safetensors file
safetensors index
sharded safetensors directory
```

For standalone safetensors whose configuration assets live elsewhere:

```bash
--config-dir /path/to/model/config
```

---

# Streaming conversion

The original dense checkpoint is never required to reside entirely in VRAM during conversion.

The quantizer processes source tensors incrementally and writes sharded safetensors output.

This makes conversion practical on systems where the original dense model cannot comfortably fit on the GPU.

---

# Quantized checkpoint format

Essential recognition metadata is embedded directly into every output safetensors shard.

Metadata includes information such as:

```text
quantization format
format version
source architecture
source checkpoint
kernel requirements
ConvRot group size
runtime compute dtype
excluded tensors
source fingerprint
```

A separate:

```text
quantization_manifest.json
```

is generated for diagnostics and inspection.

The loader validates checkpoint metadata before inference.

Incompatible formats fail explicitly.

---

# ComfyUI nodes

The pack currently provides:

```text
Qwen3 TTS Runtime Settings

Qwen3 TTS Quantized Model Loader

Qwen3 TTS Voice Design

Qwen3 TTS Diagnostics

Qwen3 TTS One-Shot Benchmark

Qwen3 TTS Unload Model
```

---

# Basic workflow

Place the converted model under:

```text
ComfyUI/models/qwen3_tts_quant/
```

Restart ComfyUI.

Then create:

```text
Qwen3 TTS Runtime Settings
              │
              ▼
Qwen3 TTS Quantized Model Loader
              │
              ▼
      Qwen3 TTS Voice Design
              │
              ▼
            AUDIO
              │
              ▼
          SaveAudio
```

An importable example is included:

```text
examples/voice_design_workflow.json
```

---

# Recommended starting settings

| Setting | Recommended |
|---|---|
| Compute dtype | `bfloat16` |
| Codec dtype | `float16` |
| Attention | `sdpa` |
| Kernel backend | `cuda` |
| CUDA graphs | `auto` |
| Memory profile | `speed` |
| Maximum sequence | `2048` |
| Safety headroom | `320 MiB` |

These are starting values.

The included benchmark node should be used to determine the best configuration for your own system.

---

# Memory strategy

The runtime treats the autoregressive path as latency-critical.

The current `speed` profile prioritizes keeping:

```text
Talker
Code Predictor
KV cache
speech decoder
```

resident on the GPU whenever the memory plan allows it.

The large text embedding table is different.

It is only required during prompt processing, so it remains in system RAM.

Only the requested embedding rows are gathered and transferred to the GPU.

This can reclaim a substantial amount of VRAM without repeatedly moving the autoregressive model.

---

# ComfyUI integration

The project does **not** modify ComfyUI core.

It does not replace:

```text
ComfyUI/comfy/*
ComfyUI/nodes.py
ComfyUI/execution.py
ComfyUI/main.py
```

The compatibility layer uses ComfyUI's memory-management interfaces to release competing models before loading the TTS engine and to clean up resources after explicit unload.

The TTS engine is intentionally retained as a persistent workflow object.

Diffusion-style layer-by-layer offloading would be counterproductive for autoregressive decoding.

---

# Benchmarking

The project includes a one-shot benchmark node intended to measure the complete generation pipeline.

Reported metrics include:

```text
audio duration
generation duration
RTF
realtime multiplier
peak VRAM
steady-state VRAM
attention backend
kernel backend
CUDA Graph state
```

The standard definition used by this project is:

```text
RTF = generation_seconds / audio_seconds
```

Therefore:

```text
RTF < 1
```

means faster-than-realtime generation.

The reciprocal is also reported:

```text
realtime_multiplier = audio_seconds / generation_seconds
```

Higher is better.

For example:

```text
realtime_multiplier = 2.0
```

means two seconds of audio are generated per second of inference time.

---

# Performance philosophy

This project intentionally avoids publishing universal "fastest" claims.

TTS performance can change significantly with:

```text
GPU architecture
CUDA version
kernel backend
prompt length
generated sequence length
attention backend
sampling settings
CUDA Graph compatibility
available VRAM
```

The included benchmark tooling exists so users can measure the actual runtime on their own system.

Optimizations are promoted to defaults only after they outperform the existing path in end-to-end generation while maintaining acceptable audio quality.

---

# Diagnostics

The Diagnostics node reports runtime information including:

```text
GPU capability
PyTorch version
CUDA version
Comfy Kitchen version
available kernel backends
selected kernel backend
attention implementation
quantization format
CUDA Graph status
memory plan
```

The runtime avoids misleading fallback behavior.

If CUDA kernels, FlashAttention, or CUDA Graph capture are unavailable, the reason is surfaced explicitly.

---

# Runtime telemetry

Voice generation returns telemetry alongside the audio.

Example:

```json
{
  "audio_seconds": 8.25,
  "total_seconds": 5.10,
  "rtf": 0.618,
  "realtime_multiplier": 1.618,
  "peak_vram_mib": 2670.0,
  "steady_vram_mib": 2190.0,
  "graph_active": true,
  "attention_backend": "sdpa",
  "kernel_backend": "cuda"
}
```

The numbers above are only an example of the telemetry schema.

They are not published benchmark results.

---

# CUDA Graph behavior

CUDA Graphs can be configured as:

```text
auto
on
off
```

### `auto`

Attempts capture.

If capture fails, the runtime records the error and uses the uncaptured execution path.

### `on`

Strict mode.

Graph capture must succeed or model loading fails.

### `off`

Disables graph capture entirely.

This is useful for debugging and backend comparisons.

---

# Sampling and CUDA Graphs

The Predictor CUDA Graph captures sampling-related state on the first generation.

After graph capture, changing:

```text
temperature
top_k
top_p
```

requires reloading the model or disabling CUDA Graphs.

This behavior is explicit rather than silently reusing stale captured parameters.

---

# Troubleshooting

## Model does not appear

The entire converted model directory must be under:

```text
ComfyUI/models/qwen3_tts_quant/
```

It should include the expected:

```text
config.json
safetensors shards
safetensors index
processor assets
speech_tokenizer/
```

Restart ComfyUI after adding a new model.

---

## Comfy Kitchen CUDA backend unavailable

Run:

```bash
/path/to/venv/bin/python - <<'PY'
import comfy_kitchen

print(comfy_kitchen.list_backends())
PY
```

The Diagnostics node also reports which backend is actually active.

The runtime does not label a fallback path as CUDA.

---

## Out of memory

Try:

```text
close other GPU applications
unload other ComfyUI models
reduce maximum sequence length
increase available VRAM headroom
```

Reducing the maximum sequence length decreases static KV-cache allocation.

The runtime estimates memory requirements before generation and can reject unsafe plans.

---

## CUDA Graph capture fails

Start with:

```text
cuda_graphs = auto
```

The failure reason will be recorded.

Use:

```text
cuda_graphs = on
```

only when diagnosing graph capture directly.

---

## FlashAttention 2 + CUDA Graphs

The current upstream graph path uses a fixed four-dimensional StaticCache mask.

FlashAttention 2's unpadding path expects a different representation and may fail during graph warmup.

The loader detects this combination.

Recommended starting path:

```text
attention_backend = sdpa
cuda_graphs = auto
```

FlashAttention 2 can still be benchmarked independently with CUDA Graphs inactive.

---

## dtype mismatch during graph warmup

Errors such as:

```text
mat1 and mat2 must have the same dtype
```

typically indicate an unnormalized floating tensor in the Talker.

The current loader normalizes floating Talker parameters and buffers to the selected compute dtype while preserving:

```text
INT8 packed weights
FP32 quantization scales
```

Restart ComfyUI after updating the node pack.

---

## Stale ComfyUI workflow values

ComfyUI stores custom-node widget values positionally.

If the Voice Design node changes between versions, an old workflow can map saved values to the wrong widgets.

Symptoms include validation errors such as:

```text
top_p > 1
temperature > 2
max_new_tokens < 2
```

Delete the old Voice Design node from the workflow and recreate it.

Recommended starter values:

```text
seed                 42
max_new_tokens       1024
temperature          0.90
top_k                50
top_p                1.00
repetition_penalty   1.05
```

---

# Explicit unloading

The project provides:

```text
Qwen3 TTS Unload Model
```

This releases:

```text
model references
CUDA Graph pools
runtime state
CUDA cache
```

Once unloaded, the previous model handle is intentionally invalid.

Execute the loader again to resume generation.

---

# Tests

Run CPU-compatible policy and metadata tests with:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

CUDA-specific behavior requires a compatible GPU environment.

This includes tests involving:

```text
Comfy Kitchen INT8 kernels
CUDA Graph capture
VRAM measurements
deterministic audio generation
end-to-end RTF
```

The project does not fabricate results for hardware it has not executed on.

---

# Project structure

```text
ComfyUI-Qwen3-TTS-Quant/
├── nodes/
│   ├── loader.py
│   ├── voice_design.py
│   ├── settings.py
│   ├── benchmark.py
│   ├── diagnostics.py
│   └── unload.py
│
├── q3tts_quant/
│   ├── adapters/
│   ├── comfy_integration/
│   ├── quantization/
│   ├── runtime/
│   ├── config.py
│   └── errors.py
│
├── qwen3_tts_quantizer/
├── benchmarks/
├── examples/
├── profiles/
├── docs/
├── tests/
├── pyproject.toml
└── requirements.txt
```

The inference engine and ComfyUI-specific integration are deliberately separated.

---

# Design documentation

For the technical architecture and quantization rationale, see:

```text
docs/DESIGN.md
```

The design documentation tracks the upstream revisions used during implementation and describes compatibility boundaries with:

```text
Qwen3-TTS
ComfyUI
Comfy Kitchen
faster-qwen3-tts
```

---

# Roadmap

Future work is benchmark-driven.

Potential directions include:

- W4A16 AWQ for batch-1 GEMV
- W8A16
- alternative W8A8 layouts
- additional Comfy Kitchen kernels
- improved codec scheduling
- progressive audio decoding
- automatic per-GPU runtime profiles
- broader Qwen3-TTS architecture support
- additional VoiceDesign-compatible checkpoints

Formats will not become defaults solely because they use fewer bits.

They must demonstrate superior end-to-end performance and acceptable speech quality.

---

# Updating safely

Update ComfyUI and Comfy Kitchen normally.

Never copy files from this project into:

```text
ComfyUI/comfy/
```

After an update:

```text
1. verify Comfy Kitchen backends
2. run Diagnostics
3. perform one short benchmark
4. confirm CUDA Graph behavior
```

Quantized checkpoints include format metadata and source fingerprints.

Unsupported formats fail explicitly.

---

# License

Apache-2.0.

Qwen3-TTS, ComfyUI, Comfy Kitchen, faster-qwen3-tts, and other upstream dependencies retain their respective licenses.

---

# Acknowledgements

Built on the work of the open-source projects behind:

- Qwen3-TTS
- ComfyUI
- Comfy Kitchen
- faster-qwen3-tts
- PyTorch
- Hugging Face

This project focuses on bringing those components together into a quantized, CUDA-optimized Qwen3-TTS runtime designed specifically for efficient ComfyUI inference.