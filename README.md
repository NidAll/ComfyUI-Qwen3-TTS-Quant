# ComfyUI-Qwen3-TTS-Quant

A native ComfyUI node pack and streaming quantization toolkit for the complete
**Qwen3-TTS 12 Hz 1.7B family**.

One runtime now supports every official 1.7B generation mode:

| Model | Capability | ComfyUI node |
|---|---|---|
| [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | Create a voice from a natural-language description | Qwen3 TTS Voice Design |
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Clone a voice from reference audio | Qwen3 TTS Voice Clone |
| [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | Generate speech with official named speakers and style instructions | Qwen3 TTS Custom Voice |

The latency-critical Talker and Code Predictor use packed **W8A8 ConvRot**
weights through Comfy Kitchen. Quantized weights remain packed during inference.
The speech tokenizer stays in FP16 by default to protect waveform reconstruction
quality.

## Features

- Full support for VoiceDesign, Base voice cloning, and CustomVoice
- Streaming conversion from Hugging Face or local checkpoints
- Architecture-aware W8A8 ConvRot quantization
- Single-file `model.safetensors` layout as the recommended default
- Packed INT8 weights throughout model loading and inference
- Comfy Kitchen CUDA backend integration
- FlashAttention 2 and SDPA attention modes
- Static KV caches for the Talker and Code Predictor
- CUDA graph acceleration with explicit feature detection and fallback reporting
- Persistent decode buffers for batch-1 autoregressive generation
- CPU selective-gather text embeddings
- Reusable voice-clone prompt cache
- Deterministic sampling controls and generation telemetry
- Explicit model unloading and CUDA resource cleanup
- No ComfyUI core modifications

## Supported checkpoint layout

The recommended layout keeps one root safetensors file:

```text
Qwen3-TTS-12Hz-1.7B-Base-W8A8-ConvRot/
├── model.safetensors
├── config.json
├── generation_config.json
├── tokenizer assets
├── processor assets
├── quantization_manifest.json
└── speech_tokenizer/
```

Sharded safetensors checkpoints remain supported for systems that need lower
CPU memory use during conversion.

## Requirements

- A modern ComfyUI installation
- A CUDA-capable NVIDIA GPU
- A ComfyUI Python environment with PyTorch and CUDA already working
- Comfy Kitchen with its CUDA backend
- `qwen-tts-hf`
- `faster-qwen3-tts`
- `safetensors`, `accelerate`, and `huggingface-hub`
- FlashAttention 2 only when selecting the FlashAttention backend

The project does not pin or replace PyTorch, torchvision, torchaudio, CUDA, or
Transformers. Installation uses the environment that already runs ComfyUI.

### Supported Transformers combination

`qwen-tts` 0.1.1 pins `transformers` 4.57.3, while `qwen-tts-hf` and
`faster-qwen3-tts` currently require `transformers` 5.15.1 or later. The two
sides cannot be resolved in one environment, and each side fails differently
(see issue #1).

The supported combination for this node is:

- `transformers` 4.57.3
- `qwen-tts` 0.1.1
- An environment that does not also install `qwen-tts-hf` or `faster-qwen3-tts`

Mixing those 5.x-only packages into the same environment is not supported
until upstream relaxes the pin. This will be revisited if upstream relaxes
the pin or if a compatibility shim becomes warranted.

## Installation

Clone the repository under ComfyUI's `custom_nodes` directory:

```text
ComfyUI/
└── custom_nodes/
    └── ComfyUI-Qwen3-TTS-Quant/
```

Then install the package using the Python executable associated with ComfyUI:

```console
<COMFYUI_PYTHON> -m pip install --upgrade-strategy only-if-needed -r requirements.txt
<COMFYUI_PYTHON> -m pip install --no-deps -e .
```

Replace `<COMFYUI_PYTHON>` with the correct executable for your installation.
Examples include a virtual-environment Python executable, a portable embedded
Python executable, or `python` after activating the environment.

Verify the required runtime:

```console
<COMFYUI_PYTHON> -c "import torch, comfy_kitchen; print(torch.__version__, torch.version.cuda); print(comfy_kitchen.list_backends())"
```

Restart ComfyUI after installation.

## Quantization

Run the quantizer from the repository directory with the same Python environment
used by ComfyUI.

### VoiceDesign

```console
<COMFYUI_PYTHON> -m qwen3_tts_quantizer --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --output <MODEL_DIRECTORY>/Qwen3-TTS-12Hz-1.7B-VoiceDesign-W8A8-ConvRot --format w8a8-convrot --device cuda --kernel-backend cuda --calibration auto --max-shard-mib 8192 --verbose
```

### Base voice-cloning model

```console
<COMFYUI_PYTHON> -m qwen3_tts_quantizer --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --output <MODEL_DIRECTORY>/Qwen3-TTS-12Hz-1.7B-Base-W8A8-ConvRot --format w8a8-convrot --device cuda --kernel-backend cuda --calibration auto --max-shard-mib 8192 --verbose
```

### CustomVoice

```console
<COMFYUI_PYTHON> -m qwen3_tts_quantizer --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --output <MODEL_DIRECTORY>/Qwen3-TTS-12Hz-1.7B-CustomVoice-W8A8-ConvRot --format w8a8-convrot --device cuda --kernel-backend cuda --calibration auto --max-shard-mib 8192 --verbose
```

`<MODEL_DIRECTORY>` should normally point to:

```text
ComfyUI/models/qwen3_tts_quant
```

The `8192 MiB` limit produces one physical shard for these packed 1.7B
checkpoints. The current writer names it
`model-00001-of-00001.safetensors` and creates a single-shard index.

For the recommended clean layout, rename:

```text
model-00001-of-00001.safetensors  ->  model.safetensors
```

Then remove `model.safetensors.index.json`.

On POSIX-compatible shells:

```console
mv <OUTPUT_DIRECTORY>/model-00001-of-00001.safetensors <OUTPUT_DIRECTORY>/model.safetensors
rm <OUTPUT_DIRECTORY>/model.safetensors.index.json
```

On PowerShell:

```powershell
Move-Item "<OUTPUT_DIRECTORY>/model-00001-of-00001.safetensors" "<OUTPUT_DIRECTORY>/model.safetensors"
Remove-Item "<OUTPUT_DIRECTORY>/model.safetensors.index.json"
```

Single-file output retains converted tensors in CPU memory until serialization.
Use a smaller value such as `--max-shard-mib 768` when conversion memory is
limited. Keep every shard and the generated index when using that mode.

### Local sources

`--model` also accepts:

- A local Hugging Face snapshot
- A directory containing safetensors shards
- A `model.safetensors.index.json` file
- One safetensors file

For a standalone safetensors file whose configuration lives elsewhere, add:

```console
--config-dir <CONFIG_DIRECTORY>
```

Conversion processes source tensors incrementally. The original dense model is
never loaded completely into GPU memory.

## Importable workflows

The `examples` directory contains:

| Workflow | File |
|---|---|
| Voice design | [voice_design_workflow.json](examples/voice_design_workflow.json) |
| Voice cloning | [voice_clone_workflow.json](examples/voice_clone_workflow.json) |
| CustomVoice | [custom_voice_workflow.json](examples/custom_voice_workflow.json) |

Import the appropriate JSON into ComfyUI, select the matching checkpoint in the
loader, and connect the output to `SaveAudio` or another compatible audio node.

The model and generation node must match:

- VoiceDesign checkpoint with `Qwen3 TTS Voice Design`
- Base checkpoint with `Qwen3 TTS Voice Clone`
- CustomVoice checkpoint with `Qwen3 TTS Custom Voice`

Mismatched combinations fail with an explicit compatibility error.

## Nodes

### Qwen3 TTS Runtime Settings

Configures:

- Transformer compute dtype
- Speech codec dtype
- Attention backend
- Comfy Kitchen kernel backend
- CUDA graph mode
- Maximum sequence length
- VRAM safety reserve

### Qwen3 TTS Quantized Model Loader

Discovers checkpoints under `models/qwen3_tts_quant`, reads their embedded
metadata, instantiates the correct architecture, and preserves packed weights.
Loaded models are reused across workflow executions.

### Qwen3 TTS Voice Design

Generates speech from:

- Target text
- Voice description
- Language
- Sampling controls

### Qwen3 TTS Voice Clone

Uses a ComfyUI `AUDIO` reference and provides two modes:

| Mode | Reference transcript | Behavior |
|---|---|---|
| `icl` | Required | Uses speaker features and reference acoustic context for stronger similarity |
| `speaker_embedding_only` | Optional | Uses only the speaker representation and can simplify cross-language cloning |

For `icl`, enter an accurate transcript of the reference recording. A short,
clean sample with one speaker gives the strongest foundation.

The latest extracted reference prompt is cached. Reusing the same audio,
transcript, and cloning mode avoids repeated prompt extraction. The cache is
bounded and cleared when the model is unloaded.

### Qwen3 TTS Custom Voice

Supports the official named speakers:

| Speaker | Native profile |
|---|---|
| Vivian | Chinese female |
| Serena | Chinese female |
| Uncle_Fu | Chinese male |
| Dylan | Beijing dialect male |
| Eric | Sichuan dialect male |
| Ryan | English male |
| Aiden | English male |
| Ono_Anna | Japanese female |
| Sohee | Korean female |

Each voice can generate any language supported by the checkpoint. Native
language pairings generally provide the most natural result.

### Qwen3 TTS Diagnostics

Reports the versions and execution paths actually active at runtime, including:

- PyTorch and CUDA runtime
- GPU compute capability
- Comfy Kitchen version and selected backend
- Attention implementation
- CUDA graph state
- Quantization format
- Model variant
- Memory plan

### Qwen3 TTS One-Shot Benchmark

Runs one deterministic VoiceDesign generation and returns structured timing and
memory telemetry. It provides a reproducible local measurement rather than a
theoretical throughput estimate.

### Qwen3 TTS Unload Model

Invalidates the model handle and releases model references, graph pools, prompt
caches, and CUDA cache resources.

## Recommended runtime settings

| Setting | Recommended start |
|---|---|
| Transformer dtype | `bfloat16` |
| Codec dtype | `float16` |
| Attention | `flash_attention_2` when compatible, otherwise `sdpa` |
| Kernel backend | `cuda` |
| CUDA graphs | `auto` |
| Memory profile | `speed` |
| Maximum sequence length | `2048` |

`cuda_graphs=auto` attempts graph initialization and records the reason when a
fallback occurs. `cuda_graphs=on` is strict. Use strict mode after confirming
that capture succeeds with the installed versions.

After a successful FlashAttention graph capture, Diagnostics reports:

```json
{
  "attention_execution_path": "flash_kvcache_cuda_graph",
  "cuda_graphs_active": true
}
```

Dynamic prompt prefill uses the selected Transformers attention integration.
Fixed-shape Talker and Code Predictor decode can use captured graphs and
`flash_attn_with_kvcache`.

## Quantization policy

The architecture adapter identifies recurrent attention and MLP linears by
module type, path, tensor shape, and configuration metadata. It applies W8A8
ConvRot to compatible Talker and Code Predictor projections.

Quality-sensitive components remain in higher precision:

- Text and codec embeddings
- RMSNorm parameters
- Output and sampling heads
- Text projections
- Speaker-processing components
- Speech-tokenizer reconstruction layers

Each safetensors output embeds:

- Quantization format and version
- Architecture and model variant
- Original model and source revision
- Tensor layout and group size
- ConvRot parameters
- Quantized-layer map
- Exclusion policy
- Compute dtype and recommended backend
- Source fingerprint

The loader validates this metadata before allocating the runtime.

## Attention and CUDA graphs

The graph runtime uses separate paths for variable-length prompt prefill and
fixed-shape autoregressive decode.

During FlashAttention graph decode, an instance-local adapter calls
`flash_attn_with_kvcache` for the Talker and Code Predictor. Cache updates occur
inside the attention operation, and sequence length stays in a CUDA tensor for
graph replay.

The adapter applies only to the loaded Qwen3-TTS model instance. It does not
modify Transformers or ComfyUI globally.

Use `sdpa` when FlashAttention 2 is unavailable or incompatible with the current
environment.

## Telemetry

Generation nodes return an audio value and a JSON telemetry string containing:

- Total generation time
- Generated audio duration
- Real-time factor
- Audio-seconds-per-generation-second
- Prefill time
- Decode time
- Codec time
- Acoustic frames per second
- Peak and steady CUDA memory
- Active attention and kernel backends
- CUDA graph status
- Seed

Real-time factor is calculated as:

```text
RTF = generation_seconds / audio_seconds
```

Values below `1.0` represent faster-than-realtime generation.

## Troubleshooting

### Checkpoint is missing from the loader

Place the complete checkpoint directory under:

```text
ComfyUI/models/qwen3_tts_quant/
```

A single-file checkpoint needs `model.safetensors` and its configuration,
processor, tokenizer, and `speech_tokenizer` assets. A sharded checkpoint needs
every shard plus `model.safetensors.index.json`. Restart ComfyUI after adding a
checkpoint.

### Prompt values appear under the wrong controls

ComfyUI stores widget values positionally. Delete stale generation nodes and
import the current workflow again. The supplied workflows include the seed
control widget required to keep the following values aligned:

```text
seed                 42
seed control         fixed
max_new_tokens       1024
temperature          0.90
top_k                50
top_p                1.00
repetition_penalty   1.05
```

### FlashAttention is unavailable

Select `sdpa`. FlashAttention availability depends on the installed PyTorch,
CUDA, GPU architecture, and FlashAttention build.

### CUDA graph capture fails

Use `cuda_graphs=auto` to retain an explicit fallback or `off` to disable
capture. If a CUDA device-side assertion has occurred, restart ComfyUI before
running another CUDA workload.

### Sampling settings changed after graph capture

Predictor sampling parameters are captured on the first graph generation.
Reload the model to capture different temperature, top-k, or top-p values, or
turn CUDA graphs off.

### Dtype mismatch

Install the latest node-pack revision and restart ComfyUI. The loader normalizes
floating Talker tensors to the selected compute dtype while retaining packed
INT8 weights and FP32 quantization scales.

### SoX warning

VoiceDesign and CustomVoice do not require reference-audio extraction. Voice
Clone passes the ComfyUI waveform and sample rate directly to the upstream
prompt builder. SoX remains useful for wider audio-format support but is not a
package-level requirement.

### Out of memory

Close competing GPU workloads, lower the maximum sequence length, and unload
other ComfyUI models. The memory planner checks available device memory before
generation and rejects unsafe plans explicitly.

## Tests

Run the CPU-compatible suite:

```console
<COMFYUI_PYTHON> -m unittest discover -s tests -v
```

CUDA kernel validation requires Comfy Kitchen and a compatible CUDA environment.
The FlashAttention KV-cache smoke test is available separately:

```console
<COMFYUI_PYTHON> benchmarks/flash_kvcache_smoke.py --dtype bfloat16
```

## Updating

Update the repository with Git, reinstall the editable package if required, and
restart ComfyUI:

```console
git pull --ff-only
<COMFYUI_PYTHON> -m pip install --no-deps -e .
```

No project file belongs inside `ComfyUI/comfy`. Keep the node pack isolated
under `custom_nodes` so ComfyUI updates remain safe.

## Project structure

```text
ComfyUI-Qwen3-TTS-Quant/
├── nodes/
├── q3tts_quant/
│   ├── adapters/
│   ├── comfy_integration/
│   ├── quantization/
│   └── runtime/
├── qwen3_tts_quantizer/
├── benchmarks/
├── examples/
├── tests/
└── docs/
```

## License

Apache-2.0. Qwen3-TTS, ComfyUI, Comfy Kitchen, FlashAttention, and
faster-qwen3-tts retain their respective licenses. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
