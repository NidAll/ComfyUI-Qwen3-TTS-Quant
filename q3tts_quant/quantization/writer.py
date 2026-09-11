from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SafetensorsShardWriter:
    def __init__(self, output_dir: Path, metadata: dict[str, str], max_shard_bytes: int):
        self.output_dir = output_dir
        self.metadata = metadata
        self.max_shard_bytes = max_shard_bytes
        self.pending: dict[str, Any] = {}
        self.pending_bytes = 0
        self.shards: list[tuple[str, list[str], int]] = []

    @staticmethod
    def _bytes(tensor: Any) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    def add(self, name: str, tensor: Any) -> None:
        # Safetensors rejects aliased/shared storage. Quantization metadata markers
        # are intentionally identical across layers, and source checkpoints can
        # also contain tied weights. Materialize an independent CPU tensor for
        # every serialized key so shard writes remain deterministic and portable.
        tensor = tensor.detach().to(device="cpu").contiguous().clone()
        size = self._bytes(tensor)
        if self.pending and self.pending_bytes + size > self.max_shard_bytes:
            self.flush()
        self.pending[name] = tensor
        self.pending_bytes += size

    def flush(self) -> None:
        if not self.pending:
            return
        from safetensors.torch import save_file

        index = len(self.shards) + 1
        temporary = f"model-{index:05d}.safetensors"
        save_file(self.pending, str(self.output_dir / temporary), metadata=self.metadata)
        self.shards.append((temporary, list(self.pending), self.pending_bytes))
        self.pending = {}
        self.pending_bytes = 0

    def finish(self) -> dict[str, Any]:
        self.flush()
        total = len(self.shards)
        weight_map: dict[str, str] = {}
        final_shards: list[str] = []
        for index, (old_name, keys, _) in enumerate(self.shards, start=1):
            new_name = f"model-{index:05d}-of-{total:05d}.safetensors"
            (self.output_dir / old_name).replace(self.output_dir / new_name)
            final_shards.append(new_name)
            weight_map.update({key: new_name for key in keys})
        index = {
            "metadata": {"total_size": sum(size for _, _, size in self.shards)},
            "weight_map": weight_map,
        }
        (self.output_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {"shards": final_shards, "tensor_count": len(weight_map), **index["metadata"]}
