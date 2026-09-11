__all__ = ["Qwen3TTSEngine"]


def __getattr__(name):
    if name == "Qwen3TTSEngine":
        from .engine import Qwen3TTSEngine

        return Qwen3TTSEngine
    raise AttributeError(name)
