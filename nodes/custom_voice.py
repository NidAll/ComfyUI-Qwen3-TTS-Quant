import json

from .common import LANGUAGES, SAMPLING_INPUTS


SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]


class Qwen3TTSCustomVoice:
    CATEGORY = "audio/qwen3_tts_quant"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "telemetry")

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": ("Q3TTS_MODEL",),
            "text": ("STRING", {"multiline": True, "default": "Hello from Qwen3 TTS."}),
            "speaker": (SPEAKERS, {"default": "Vivian"}),
            "voice_instruction": (
                "STRING",
                {"multiline": True, "default": "Speak warmly and naturally."},
            ),
            "language": (LANGUAGES, {"default": "Auto"}),
        }
        required.update(SAMPLING_INPUTS)
        return {"required": required}

    def generate(
        self,
        model,
        text,
        speaker,
        voice_instruction,
        language,
        seed,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    ):
        audio, telemetry = model.require_engine().generate_custom_voice(
            text=text,
            speaker=speaker,
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
