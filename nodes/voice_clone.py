import json

from .common import LANGUAGES, SAMPLING_INPUTS


class Qwen3TTSVoiceClone:
    CATEGORY = "audio/qwen3_tts_quant"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "telemetry")

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": ("Q3TTS_MODEL",),
            "reference_audio": ("AUDIO",),
            "text": ("STRING", {"multiline": True, "default": "This is my cloned voice."}),
            "reference_text": (
                "STRING",
                {"multiline": True, "default": "Enter the exact transcript of the reference audio."},
            ),
            "language": (LANGUAGES, {"default": "Auto"}),
            "clone_mode": (["icl", "speaker_embedding_only"], {"default": "icl"}),
            "append_silence": ("BOOLEAN", {"default": True}),
        }
        required.update(SAMPLING_INPUTS)
        return {"required": required}

    def generate(
        self,
        model,
        reference_audio,
        text,
        reference_text,
        language,
        clone_mode,
        append_silence,
        seed,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    ):
        audio, telemetry = model.require_engine().generate_voice_clone(
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            language=language,
            clone_mode=clone_mode,
            append_silence=append_silence,
            seed=seed,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return audio, json.dumps(telemetry.to_dict(), indent=2, sort_keys=True)
