from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeSettings:
    compute_dtype: str = "bfloat16"
    codec_dtype: str = "float16"
    attention_backend: str = "sdpa"
    kernel_backend: str = "cuda"
    cuda_graphs: str = "auto"
    memory_profile: str = "speed"
    max_sequence_length: int = 2048
    safety_headroom_mib: int = 320
    telemetry: bool = False

    def __post_init__(self) -> None:
        if self.compute_dtype not in {"bfloat16", "float16"}:
            raise ValueError("compute_dtype must be bfloat16 or float16")
        if self.codec_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("codec_dtype must be bfloat16, float16, or float32")
        if self.attention_backend not in {"sdpa", "flash_attention_2", "eager"}:
            raise ValueError("unsupported attention backend")
        if self.kernel_backend not in {"cuda", "auto", "triton", "eager"}:
            raise ValueError("unsupported kernel backend")
        if self.cuda_graphs not in {"auto", "on", "off"}:
            raise ValueError("cuda_graphs must be auto, on, or off")
        if self.memory_profile != "speed":
            raise ValueError("this release implements the speed memory profile only")
        if not 128 <= self.max_sequence_length <= 8192:
            raise ValueError("max_sequence_length must be between 128 and 8192")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
