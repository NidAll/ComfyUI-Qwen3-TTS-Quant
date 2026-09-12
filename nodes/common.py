LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

SAMPLING_INPUTS = {
    "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
    "max_new_tokens": ("INT", {"default": 1024, "min": 2, "max": 8192, "step": 16}),
    "temperature": ("FLOAT", {"default": 0.9, "min": 0.05, "max": 2.0, "step": 0.05}),
    "top_k": ("INT", {"default": 50, "min": 1, "max": 1000}),
    "top_p": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.01}),
    "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01}),
}
