from __future__ import annotations

import time
from dataclasses import asdict, dataclass


@dataclass
class GenerationTelemetry:
    total_seconds: float = 0.0
    audio_seconds: float = 0.0
    rtf: float = 0.0
    realtime_multiplier: float = 0.0
    peak_vram_mib: float = 0.0
    steady_vram_mib: float = 0.0
    graph_active: bool = False
    attention_backend: str = ""
    kernel_backend: str = ""
    seed: int = 0
    acoustic_steps: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    codec_seconds: float = 0.0
    acoustic_frames_per_second: float = 0.0

    def to_dict(self):
        return asdict(self)


class WallTimer:
    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.started
