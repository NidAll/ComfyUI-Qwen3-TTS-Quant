from .voice_design import Qwen3TTSVoiceDesign


class Qwen3TTSBenchmark(Qwen3TTSVoiceDesign):
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["text"][1]["default"] = "The quick brown fox jumps over the lazy dog."
        inputs["required"]["voice_instruction"][1]["default"] = "A calm, neutral studio voice at a natural pace."
        return inputs
