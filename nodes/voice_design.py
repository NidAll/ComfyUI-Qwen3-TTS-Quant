import json


class Qwen3TTSVoiceDesign:
    CATEGORY = "audio/qwen3_tts_quant"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "telemetry")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("Q3TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello from Qwen3 TTS."}),
                "voice_instruction": ("STRING", {"multiline": True, "default": "A clear, warm female voice."}),
                "language": (["Auto", "English", "Chinese", "Arabic", "French", "Japanese"], {"default": "Auto"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 2, "max": 8192, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.05, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01}),
            }
        }

    def generate(self, model, text, voice_instruction, language, seed, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
        audio, telemetry = model.require_engine().generate_voice_design(
            text=text,
            instruction=voice_instruction,
            language=language,
            seed=seed,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return audio, json.dumps(telemetry.to_dict(), indent=2, sort_keys=True)
