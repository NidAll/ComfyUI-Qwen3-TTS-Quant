from .benchmark import Qwen3TTSBenchmark
from .custom_voice import Qwen3TTSCustomVoice
from .diagnostics import Qwen3TTSDiagnostics
from .loader import Qwen3TTSQuantizedLoader
from .settings import Qwen3TTSRuntimeSettings
from .unload import Qwen3TTSUnload
from .voice_design import Qwen3TTSVoiceDesign
from .voice_clone import Qwen3TTSVoiceClone

NODE_CLASS_MAPPINGS = {
    "Qwen3TTSQuantizedLoader": Qwen3TTSQuantizedLoader,
    "Qwen3TTSVoiceDesign": Qwen3TTSVoiceDesign,
    "Qwen3TTSVoiceClone": Qwen3TTSVoiceClone,
    "Qwen3TTSCustomVoice": Qwen3TTSCustomVoice,
    "Qwen3TTSRuntimeSettings": Qwen3TTSRuntimeSettings,
    "Qwen3TTSDiagnostics": Qwen3TTSDiagnostics,
    "Qwen3TTSBenchmark": Qwen3TTSBenchmark,
    "Qwen3TTSUnload": Qwen3TTSUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTSQuantizedLoader": "Qwen3 TTS Quantized Model Loader",
    "Qwen3TTSVoiceDesign": "Qwen3 TTS Voice Design",
    "Qwen3TTSVoiceClone": "Qwen3 TTS Voice Clone",
    "Qwen3TTSCustomVoice": "Qwen3 TTS Custom Voice",
    "Qwen3TTSRuntimeSettings": "Qwen3 TTS Runtime Settings",
    "Qwen3TTSDiagnostics": "Qwen3 TTS Diagnostics",
    "Qwen3TTSBenchmark": "Qwen3 TTS One-Shot Benchmark",
    "Qwen3TTSUnload": "Qwen3 TTS Unload Model",
}
