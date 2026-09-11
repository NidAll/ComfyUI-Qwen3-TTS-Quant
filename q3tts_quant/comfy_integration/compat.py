from __future__ import annotations

import logging
from pathlib import Path

from ..errors import CompatibilityError

log = logging.getLogger("q3tts.comfy")
FOLDER_NAME = "qwen3_tts_quant"


def register_model_directory() -> Path | None:
    try:
        import folder_paths
    except ImportError:
        return None
    root = Path(folder_paths.models_dir) / FOLDER_NAME
    root.mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path(FOLDER_NAME, str(root), is_default=True)
    return root


def checkpoint_choices() -> list[str]:
    root = register_model_directory()
    if root is None:
        return ["<ComfyUI unavailable>"]
    choices = []
    for config in root.rglob("config.json"):
        directory = config.parent
        if list(directory.glob("*.safetensors")):
            choices.append(directory.relative_to(root).as_posix())
    return sorted(set(choices)) or ["<place quantized model here>"]


def checkpoint_path(choice: str) -> Path:
    root = register_model_directory()
    if root is None:
        raise CompatibilityError("folder_paths is unavailable; this loader must run inside ComfyUI")
    candidate = (root / choice).resolve()
    if root.resolve() not in candidate.parents and candidate != root.resolve():
        raise CompatibilityError("checkpoint path escapes the ComfyUI model directory")
    if not (candidate / "config.json").is_file():
        raise CompatibilityError(f"invalid Qwen3-TTS checkpoint directory: {candidate}")
    return candidate


def prepare_vram(memory_required: int = 3 * 1024**3) -> None:
    """Use ComfyUI's public pressure path without patching its model lifecycle."""
    try:
        import comfy.model_management as mm

        device = mm.get_torch_device()
        mm.free_memory(memory_required, device)
    except ImportError as exc:
        raise CompatibilityError("ComfyUI model management API is unavailable") from exc
    except Exception as exc:
        raise CompatibilityError(f"ComfyUI could not reserve VRAM for Qwen3-TTS: {exc}") from exc


def release_vram_cache() -> None:
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception as exc:
        log.debug("ComfyUI cache release unavailable: %s", exc)
