from __future__ import annotations

"""Validate the exact strided KV-cache layout before loading the TTS model."""

import argparse
import json

from q3tts_quant.runtime.flash_graph import _load_flash_kvcache


def run_case(query_length: int, dtype_name: str) -> dict:
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    flash = _load_flash_kvcache()
    batch, query_heads, kv_heads, head_dim = 1, 16, 8, 128
    old_length, maximum = 7, 32
    torch.manual_seed(1234 + query_length)
    query = torch.randn(batch, query_length, query_heads, head_dim, device="cuda", dtype=dtype)
    new_key = torch.randn(batch, query_length, kv_heads, head_dim, device="cuda", dtype=dtype)
    new_value = torch.randn_like(new_key)

    # Build the same B,H,S,D allocation and B,S,H,D strided view used by
    # Transformers StaticCache plus the runtime adapter.
    key_storage = torch.zeros(batch, kv_heads, maximum, head_dim, device="cuda", dtype=dtype)
    value_storage = torch.zeros_like(key_storage)
    key_storage[:, :, :old_length].normal_()
    value_storage[:, :, :old_length].normal_()
    key_cache = key_storage.transpose(1, 2)
    value_cache = value_storage.transpose(1, 2)
    cache_length = torch.tensor([old_length], device="cuda", dtype=torch.int32)

    eager_output = flash(
        query,
        key_cache,
        value_cache,
        k=new_key,
        v=new_value,
        cache_seqlens=cache_length,
        causal=True,
    )
    torch.cuda.synchronize()

    keys = key_storage[:, :, : old_length + query_length]
    values = value_storage[:, :, : old_length + query_length]
    keys = keys.repeat_interleave(query_heads // kv_heads, dim=1)
    values = values.repeat_interleave(query_heads // kv_heads, dim=1)
    mask = torch.arange(old_length + query_length, device="cuda").unsqueeze(0) <= (
        old_length + torch.arange(query_length, device="cuda").unsqueeze(1)
    )
    reference = functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        keys,
        values,
        attn_mask=mask,
        dropout_p=0.0,
    ).transpose(1, 2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = flash(
            query,
            key_cache,
            value_cache,
            k=new_key,
            v=new_value,
            cache_seqlens=cache_length,
            causal=True,
        )
    graph.replay()
    torch.cuda.synchronize()

    eager_error = (eager_output.float() - reference.float()).abs().max().item()
    graph_error = (graph_output.float() - reference.float()).abs().max().item()
    cosine = functional.cosine_similarity(
        graph_output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    return {
        "query_length": query_length,
        "dtype": dtype_name,
        "cache_stride": list(key_cache.stride()),
        "eager_max_abs_error": eager_error,
        "graph_max_abs_error": graph_error,
        "graph_cosine_similarity": cosine,
        "passed": cosine > 0.999,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()
    results = [run_case(1, args.dtype), run_case(2, args.dtype)]
    print(json.dumps(results, indent=2))
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

