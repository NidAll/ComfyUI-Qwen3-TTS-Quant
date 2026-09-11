from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..errors import CheckpointError


@dataclass(frozen=True)
class ModelSource:
    root: Path
    weight_files: tuple[Path, ...]
    model_id: str
    revision: str
    config_root: Path

    def iter_tensor_specs(self) -> Iterator[tuple[str, tuple[int, ...], str]]:
        from safetensors import safe_open

        for file in self.weight_files:
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    tensor = handle.get_slice(name)
                    yield name, tuple(tensor.get_shape()), str(tensor.get_dtype()).lower()

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for file in self.weight_files:
            digest.update(file.name.encode())
            digest.update(str(file.stat().st_size).encode())
            with file.open("rb") as stream:
                prefix = stream.read(8)
                digest.update(prefix)
                if len(prefix) == 8:
                    header_size = int.from_bytes(prefix, "little")
                    digest.update(stream.read(header_size))
        return digest.hexdigest()


def _resolve_local(path: Path, config_dir: str | None) -> ModelSource:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.name.endswith(".safetensors.index.json"):
            root = path.parent
            raw = json.loads(path.read_text(encoding="utf-8"))
            files = tuple(root / name for name in sorted(set(raw.get("weight_map", {}).values())))
        elif path.suffix == ".safetensors":
            root = path.parent
            files = (path,)
        else:
            raise CheckpointError(f"unsupported model file: {path}")
    elif path.is_dir():
        root = path
        index_files = sorted(root.glob("*.safetensors.index.json"))
        if index_files:
            raw = json.loads(index_files[0].read_text(encoding="utf-8"))
            names = sorted(set(raw.get("weight_map", {}).values()))
            files = tuple(root / name for name in names)
        else:
            files = tuple(sorted(root.glob("*.safetensors")))
            files = tuple(f for f in files if "speech_tokenizer" not in f.parts)
    else:
        raise CheckpointError(f"model source does not exist: {path}")
    if not files or any(not file.is_file() for file in files):
        raise CheckpointError(f"no complete safetensors checkpoint found under {root}")
    cfg = Path(config_dir).expanduser().resolve() if config_dir else root
    if not (cfg / "config.json").is_file():
        raise CheckpointError("config.json is required; pass --config-dir for a standalone safetensors file")
    return ModelSource(root, files, str(path), "local", cfg)


def resolve_model_source(model: str, revision: str = "main", config_dir: str | None = None) -> ModelSource:
    local = Path(model).expanduser()
    if local.exists():
        return _resolve_local(local, config_dir)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise CheckpointError("huggingface-hub is required for Hugging Face model IDs") from exc
    snapshot = Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            allow_patterns=[
                "*.json", "*.safetensors", "*.safetensors.index.json", "*.txt",
                "*.model", "*.tiktoken", "tokenizer*", "merges.txt", "vocab.json",
                "speech_tokenizer/*",
            ],
        )
    )
    source = _resolve_local(snapshot, config_dir)
    resolved_revision = snapshot.name if snapshot.parent.name == "snapshots" else revision
    return ModelSource(source.root, source.weight_files, model, resolved_revision, source.config_root)


def load_config(source: ModelSource) -> dict:
    try:
        return json.loads((source.config_root / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"invalid config.json under {source.config_root}") from exc
