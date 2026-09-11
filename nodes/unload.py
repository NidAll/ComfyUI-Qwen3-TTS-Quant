class Qwen3TTSUnload:
    CATEGORY = "audio/qwen3_tts_quant"
    FUNCTION = "unload"
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("Q3TTS_MODEL",)}}

    def unload(self, model):
        model.unload()
        return ("Qwen3-TTS CUDA resources released",)
