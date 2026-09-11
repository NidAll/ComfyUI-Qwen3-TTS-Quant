from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from ..config import RuntimeSettings
from ..runtime.engine import Qwen3TTSEngine
from .compat import prepare_vram, release_vram_cache


@dataclass
class Qwen3TTSHandle:
    checkpoint: Path
    settings: RuntimeSettings
    engine: Qwen3TTSEngine
    _lock: RLock

    @classmethod
    def load(cls, checkpoint: Path, settings: RuntimeSettings):
        prepare_vram()
        engine = Qwen3TTSEngine.load(checkpoint, settings)
        return cls(checkpoint, settings, engine, RLock())

    def unload(self):
        with self._lock:
            if self.engine is not None:
                self.engine.unload()
                self.engine = None
                release_vram_cache()

    def require_engine(self):
        if self.engine is None:
            raise RuntimeError("Qwen3-TTS model has been explicitly unloaded; run the loader again")
        return self.engine

    def get_models(self):
        # Comfy's prompt lifecycle recognizes this hook. The packed autoregressive
        # runtime manages residency as one indivisible unit, so no ModelPatcher is exposed.
        return []
