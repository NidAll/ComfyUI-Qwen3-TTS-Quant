import json


class Qwen3TTSDiagnostics:
    CATEGORY = "audio/qwen3_tts_quant"
    FUNCTION = "inspect"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("diagnostics",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("Q3TTS_MODEL",)}}

    def inspect(self, model):
        return (json.dumps(model.require_engine().diagnostics, indent=2, sort_keys=True, default=str),)
