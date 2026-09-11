from ..q3tts_quant.comfy_integration import Qwen3TTSHandle, checkpoint_choices, checkpoint_path
from ..q3tts_quant.config import RuntimeSettings


class Qwen3TTSQuantizedLoader:
    CATEGORY = "audio/qwen3_tts_quant"
    FUNCTION = "load"
    RETURN_TYPES = ("Q3TTS_MODEL", "STRING")
    RETURN_NAMES = ("model", "memory_plan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"checkpoint": (checkpoint_choices(),)},
            "optional": {"settings": ("Q3TTS_SETTINGS",)},
        }

    def load(self, checkpoint, settings=None):
        settings = settings or RuntimeSettings()
        handle = Qwen3TTSHandle.load(checkpoint_path(checkpoint), settings)
        import json

        plan = json.dumps(handle.engine.memory_plan.to_dict(), indent=2, sort_keys=True)
        return handle, plan
