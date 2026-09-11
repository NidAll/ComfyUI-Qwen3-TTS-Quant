from ..q3tts_quant.config import RuntimeSettings


class Qwen3TTSRuntimeSettings:
    CATEGORY = "audio/qwen3_tts_quant"
    FUNCTION = "make"
    RETURN_TYPES = ("Q3TTS_SETTINGS",)
    RETURN_NAMES = ("settings",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "compute_dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
                "codec_dtype": (["float16", "bfloat16", "float32"], {"default": "float16"}),
                "attention_backend": (["sdpa", "flash_attention_2", "eager"], {"default": "sdpa"}),
                "kernel_backend": (["cuda"], {"default": "cuda"}),
                "cuda_graphs": (["auto", "on", "off"], {"default": "auto"}),
                "memory_profile": (["speed"], {"default": "speed"}),
                "max_sequence_length": ("INT", {"default": 2048, "min": 128, "max": 8192, "step": 128}),
                "safety_headroom_mib": ("INT", {"default": 320, "min": 128, "max": 1024, "step": 32}),
                "telemetry": ("BOOLEAN", {"default": False}),
            }
        }

    def make(self, **kwargs):
        return (RuntimeSettings(**kwargs),)
