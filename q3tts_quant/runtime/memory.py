from __future__ import annotations

from dataclasses import asdict, dataclass

from ..errors import InsufficientVRAMError


@dataclass(frozen=True)
class MemoryPlan:
    free_mib: int
    autoregressive_weights_mib: int
    kv_cache_mib: int
    codec_weights_mib: int
    graph_workspace_mib: int
    safety_headroom_mib: int
    total_required_mib: int
    talker: str
    predictor: str
    codec: str
    text_embedding: str

    def to_dict(self):
        return asdict(self)


def kv_cache_mib(max_sequence_length: int, bytes_per_element: int = 2) -> int:
    # 28 layers * K/V * 8 KV heads * head dimension 128.
    return (28 * 2 * 8 * 128 * bytes_per_element * max_sequence_length + 2**20 - 1) // 2**20


def build_memory_plan(
    free_bytes: int,
    stored_main_bytes: int,
    max_sequence_length: int,
    safety_headroom_mib: int,
    graphs: bool,
    profile: str,
) -> MemoryPlan:
    free_mib = free_bytes // 2**20
    main_mib = (stored_main_bytes + 2**20 - 1) // 2**20
    kv_mib = kv_cache_mib(max_sequence_length)
    codec_mib = 340
    graph_mib = 360 if graphs else 96
    total = main_mib + kv_mib + codec_mib + graph_mib + safety_headroom_mib
    if total > free_mib:
        raise InsufficientVRAMError(
            f"estimated requirement {total} MiB exceeds current free VRAM {free_mib} MiB; "
            "reduce max sequence length or close other GPU workloads"
        )
    return MemoryPlan(
        free_mib=free_mib,
        autoregressive_weights_mib=main_mib,
        kv_cache_mib=kv_mib,
        codec_weights_mib=codec_mib,
        graph_workspace_mib=graph_mib,
        safety_headroom_mib=safety_headroom_mib,
        total_required_mib=total,
        talker="GPU resident",
        predictor="GPU resident",
        codec="FP16 GPU resident",
        text_embedding="BF16 CPU selective gather",
    )
