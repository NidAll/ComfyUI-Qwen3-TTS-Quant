from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from ..config import RuntimeSettings
from ..errors import BackendUnavailableError, CompatibilityError
from ..logging_utils import log_event
from .backends import inspect_backends, require_cuda_int8_backend
from .loader import load_quantized_base_model, read_quantized_metadata
from .memory import MemoryPlan, build_memory_plan
from .telemetry import GenerationTelemetry

log = logging.getLogger("q3tts.engine")


def _dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _set_static_cache_dtype(graph, config, max_cache_len: int, dtype) -> bool:
    """Use the graph compute dtype for Transformers' lazy StaticCache.

    Recent Transformers releases expose ``dtype`` and ``device`` on
    ``StaticCache``.  Older releases infer the cache dtype during lazy layer
    initialization, so retain the graph's original cache when those keywords
    are unavailable.  This adapter prevents a BF16/FP32 ``index_copy_`` at
    graph capture without tying the node pack to one Transformers signature.
    """
    try:
        from transformers import StaticCache

        graph.static_cache = StaticCache(
            config=config,
            max_cache_len=max_cache_len,
            device="cuda",
            dtype=dtype,
        )
        return True
    except TypeError:
        return False


def _populate_sample_rate(base_model) -> int | None:
    """Expose the codec's configured waveform rate to faster-qwen3-tts.

    Qwen3-TTS 12 Hz refers to codec frame rate, while the waveform is normally
    24 kHz. The tokenizer stores that value as ``output_sample_rate`` on its
    config/model rather than as ``sample_rate``. Copy it to the wrapper's
    conventional attribute so the external CUDA-graph wrapper does not emit a
    misleading fallback warning.
    """
    speech_tokenizer = getattr(getattr(base_model, "model", None), "speech_tokenizer", None)
    candidates = [
        getattr(speech_tokenizer, "sample_rate", None),
        getattr(speech_tokenizer, "output_sample_rate", None),
        getattr(getattr(speech_tokenizer, "config", None), "output_sample_rate", None),
        getattr(base_model, "sample_rate", None),
    ]
    for value in candidates:
        if value is not None:
            sample_rate = int(value)
            setattr(base_model, "sample_rate", sample_rate)
            return sample_rate
    return None


def _make_fast_wrapper(base_model, settings: RuntimeSettings):
    from faster_qwen3_tts import FasterQwen3TTS
    from faster_qwen3_tts.predictor_graph import PredictorGraph
    from faster_qwen3_tts.talker_graph import TalkerGraph

    _populate_sample_rate(base_model)
    talker = base_model.model.talker
    talker_config = base_model.model.config.talker_config
    predictor = talker.code_predictor
    pred_config = predictor.model.config
    predictor_graph = PredictorGraph(
        predictor,
        pred_config,
        talker_config.hidden_size,
        device="cuda",
        dtype=_dtype(settings.compute_dtype),
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
    )
    talker_graph = TalkerGraph(
        talker.model,
        talker_config,
        device="cuda",
        dtype=_dtype(settings.compute_dtype),
        max_seq_len=settings.max_sequence_length,
    )
    # The external graph package supports multiple Transformers generations.
    # Replace its default-inferred caches only when the installed cache API
    # accepts explicit dtype/device arguments.
    predictor_cache_configured = _set_static_cache_dtype(
        predictor_graph,
        pred_config,
        predictor_graph.max_seq,
        _dtype(settings.compute_dtype),
    )
    talker_cache_configured = _set_static_cache_dtype(
        talker_graph,
        talker_config,
        settings.max_sequence_length,
        _dtype(settings.compute_dtype),
    )
    if predictor_cache_configured or talker_cache_configured:
        log.info(
            "configured CUDA graph StaticCache dtype=%s predictor=%s talker=%s",
            settings.compute_dtype,
            predictor_cache_configured,
            talker_cache_configured,
        )
    return FasterQwen3TTS(
        base_model=base_model,
        predictor_graph=predictor_graph,
        talker_graph=talker_graph,
        device="cuda",
        dtype=_dtype(settings.compute_dtype),
        max_seq_len=settings.max_sequence_length,
    )


@dataclass
class Qwen3TTSEngine:
    checkpoint: Path
    settings: RuntimeSettings
    base_model: Any
    generator: Any
    memory_plan: MemoryPlan
    metadata: Any
    diagnostics: dict[str, Any]
    active_cuda_graphs: bool = False
    _lock: RLock = field(default_factory=RLock, repr=False)
    _unloaded: bool = False
    _graph_sampling_signature: tuple[float, int, float] | None = field(default=None, repr=False)

    @classmethod
    def load(cls, checkpoint: str | Path, settings: RuntimeSettings) -> "Qwen3TTSEngine":
        import torch

        if not torch.cuda.is_available():
            raise BackendUnavailableError("Qwen3-TTS optimized runtime requires an NVIDIA CUDA GPU")
        device = torch.device("cuda")
        checkpoint = Path(checkpoint).expanduser().resolve()
        metadata, checkpoint_bytes = read_quantized_metadata(checkpoint)
        status = inspect_backends(settings)
        require_cuda_int8_backend(status)

        free_bytes, _ = torch.cuda.mem_get_info(device)
        # The text embedding is held on CPU. Estimate it from the canonical model
        # config before asking ComfyUI to free competing models.
        import json

        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        talker_cfg = config.get("talker_config", config)
        text_vocab = int(talker_cfg.get("text_vocab_size", config.get("text_vocab_size", 151936)))
        text_hidden = int(talker_cfg.get("text_hidden_size", config.get("text_hidden_size", 2048)))
        text_bytes = text_vocab * text_hidden * 2
        gpu_main_bytes = max(0, checkpoint_bytes - text_bytes)
        flash_graph_incompatible = settings.attention_backend == "flash_attention_2"
        if flash_graph_incompatible and settings.cuda_graphs == "on":
            raise CompatibilityError(
                "FlashAttention 2 cannot be combined with the current faster-qwen3-tts "
                "static CUDA-graph masks. Select CUDA graphs auto/off; auto keeps real "
                "FlashAttention on the dynamic upstream path."
            )
        if flash_graph_incompatible and settings.cuda_graphs == "auto":
            log.warning(
                "CUDA graphs disabled for flash_attention_2: the predictor graph uses "
                "a fixed 4D StaticCache mask that FlashAttention 2 cannot unpad safely"
            )
        wants_graphs = settings.cuda_graphs in {"auto", "on"} and not flash_graph_incompatible
        plan = build_memory_plan(
            free_bytes,
            gpu_main_bytes,
            settings.max_sequence_length,
            settings.safety_headroom_mib,
            wants_graphs,
            settings.memory_profile,
        )

        base, metadata, _ = load_quantized_base_model(checkpoint, settings, device)
        generator = base
        active_graphs = False
        if wants_graphs:
            try:
                generator = _make_fast_wrapper(base, settings)
                active_graphs = True
            except Exception as exc:
                if settings.cuda_graphs == "on":
                    raise CompatibilityError(f"CUDA graph runtime initialization failed: {exc}") from exc
                log.warning("CUDA graphs unavailable in auto mode; using upstream static-cache path: %s", exc)

        diagnostics = status.to_dict()
        diagnostics.update(
            {
                "checkpoint": str(checkpoint),
                "quantization": metadata.quantization_format,
                "compute_dtype": settings.compute_dtype,
                "codec_dtype": settings.codec_dtype,
                "cuda_graphs_active": active_graphs,
                "memory_plan": plan.to_dict(),
            }
        )
        if flash_graph_incompatible:
            diagnostics["cuda_graphs_disabled_reason"] = (
                "flash_attention_2_static_cache_mask_incompatible"
            )
        log_event(
            "runtime_loaded",
            gpu=status.gpu,
            compute_capability=status.compute_capability,
            torch=status.torch,
            cuda=status.cuda_runtime,
            comfy_kitchen=status.comfy_kitchen,
            kernel_backend=settings.kernel_backend,
            attention=settings.attention_backend,
            quantization=metadata.quantization_format,
            compute_dtype=settings.compute_dtype,
            codec_dtype=settings.codec_dtype,
            cuda_graphs=active_graphs,
            memory_plan=plan.to_dict(),
        )
        return cls(checkpoint, settings, base, generator, plan, metadata, diagnostics, active_graphs)

    def warmup(
        self,
        prompt_tokens: int = 96,
        predictor_sampling: tuple[float, int, float] = (0.9, 50, 1.0),
    ) -> None:
        if self._unloaded:
            raise RuntimeError("model has been unloaded")
        if self.active_cuda_graphs and not getattr(self.generator, "_warmed_up", False):
            try:
                predictor = self.generator.predictor_graph
                predictor.temperature, predictor.top_k, predictor.top_p = predictor_sampling
                self._graph_sampling_signature = predictor_sampling
                self.generator.warmup(prefill_len=min(prompt_tokens, self.settings.max_sequence_length - 2))
            except Exception as exc:
                if self.settings.cuda_graphs == "on":
                    raise CompatibilityError(f"CUDA graph capture failed: {exc}") from exc
                log.warning("CUDA graph capture failed in auto mode; disabling graphs: %s", exc)
                self.generator = self.base_model
                self.active_cuda_graphs = False
                self._graph_sampling_signature = None
                gc.collect()
                import torch

                torch.cuda.empty_cache()
                self.diagnostics["cuda_graphs_active"] = False
                self.diagnostics["cuda_graph_error"] = str(exc)

    def generate_voice_design(
        self,
        *,
        text: str,
        instruction: str,
        language: str,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> tuple[dict[str, Any], GenerationTelemetry]:
        import torch

        if self._unloaded:
            raise RuntimeError("model has been unloaded")
        with self._lock, torch.inference_mode():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            signature = (temperature, top_k, top_p)
            if self.active_cuda_graphs and self._graph_sampling_signature not in {None, signature}:
                raise CompatibilityError(
                    "predictor CUDA graph sampling parameters changed after capture; reload the model "
                    "to recapture, or select CUDA graphs off"
                )
            self.warmup(predictor_sampling=signature)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            wavs, sample_rate = self.generator.generate_voice_design(
                text=text,
                instruct=instruction,
                language=language,
                max_new_tokens=min(max_new_tokens, self.settings.max_sequence_length),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            waveform = torch.as_tensor(wavs[0], dtype=torch.float32, device="cpu").flatten()
            audio_seconds = waveform.numel() / float(sample_rate)
            steady = torch.cuda.memory_allocated()
            telemetry = GenerationTelemetry(
                total_seconds=elapsed,
                audio_seconds=audio_seconds,
                rtf=elapsed / max(audio_seconds, 1e-9),
                realtime_multiplier=audio_seconds / max(elapsed, 1e-9),
                peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20,
                steady_vram_mib=steady / 2**20,
                graph_active=self.active_cuda_graphs,
                attention_backend=self.settings.attention_backend,
                kernel_backend=self.settings.kernel_backend,
                seed=seed,
            )
            return {"waveform": waveform[None, None, :], "sample_rate": int(sample_rate)}, telemetry

    def unload(self) -> None:
        if self._unloaded:
            return
        import torch

        with self._lock:
            self.generator = None
            self.base_model = None
            self._unloaded = True
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
