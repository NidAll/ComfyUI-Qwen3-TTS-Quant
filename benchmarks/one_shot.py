from __future__ import annotations

import argparse
import json

from q3tts_quant.config import RuntimeSettings
from q3tts_quant.runtime.engine import Qwen3TTSEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="One warmed VoiceDesign generation, not a format matrix")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--instruction", default="A calm, neutral studio voice at a natural pace.")
    parser.add_argument("--language", default="English")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--attention",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
    )
    parser.add_argument("--compute-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--cuda-graphs", choices=["auto", "on", "off"], default="auto")
    args = parser.parse_args()
    settings = RuntimeSettings(
        attention_backend=args.attention,
        compute_dtype=args.compute_dtype,
        cuda_graphs=args.cuda_graphs,
    )
    engine = Qwen3TTSEngine.load(args.checkpoint, settings)
    try:
        _, telemetry = engine.generate_voice_design(
            text=args.text,
            instruction=args.instruction,
            language=args.language,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.05,
        )
        result = telemetry.to_dict()
        result["runtime"] = engine.diagnostics
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        engine.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
