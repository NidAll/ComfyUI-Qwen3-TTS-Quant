from __future__ import annotations

import argparse
import json
import logging

from q3tts_quant.errors import Q3TTSError
from q3tts_quant.quantization.quantizer import quantize_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen3-tts-quantize",
        description="Create a Comfy-compatible W8A8 ConvRot Qwen3-TTS VoiceDesign checkpoint.",
    )
    parser.add_argument("--model", required=True, help="Hugging Face ID, snapshot, directory, or safetensors file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--config-dir", help="Config/tokenizer directory when --model is one safetensors file")
    parser.add_argument("--format", default="auto", choices=["auto", "w8a8-convrot"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--kernel-backend", default="cuda", choices=["cuda", "auto", "triton", "eager"])
    parser.add_argument("--calibration", default="auto", help="Accepted for workflow compatibility; W8A8 uses dynamic activations")
    parser.add_argument("--max-shard-mib", type=int, default=768)
    parser.add_argument("--benchmark", action="store_true", help="Run backend smoke validation; no format matrix is performed")
    parser.add_argument("--force", action="store_true", help="Replace generated checkpoint shards in the output directory")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[Q3TTS] %(levelname)s %(message)s",
    )
    if args.calibration not in {"auto", "none"}:
        logging.getLogger("q3tts.quantizer").info(
            "Calibration text is ignored for W8A8 ConvRot because activation quantization is dynamic."
        )
    try:
        result = quantize_model(
            args.model,
            args.output,
            revision=args.revision,
            config_dir=args.config_dir,
            device=args.device,
            kernel_backend=args.kernel_backend,
            max_shard_mib=args.max_shard_mib,
            force=args.force,
        )
    except (Q3TTSError, ValueError) as exc:
        logging.getLogger("q3tts.quantizer").error("%s", exc)
        return 2
    if args.benchmark:
        result["benchmark"] = "backend execution was validated during quantization; no format matrix requested"
    print(json.dumps({key: value for key, value in result.items() if key != "metadata"}, indent=2, sort_keys=True))
    return 0
