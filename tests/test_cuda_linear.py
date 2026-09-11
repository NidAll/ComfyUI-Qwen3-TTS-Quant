import unittest

try:
    import torch
    import comfy_kitchen
except ImportError:
    torch = None
    comfy_kitchen = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available() and comfy_kitchen is not None, "CUDA and Comfy Kitchen required")
class CUDALinearTests(unittest.TestCase):
    def test_w8a8_convrot_matches_dense_reference(self):
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
        from q3tts_quant.runtime.linear import W8A8ConvRotLinear

        torch.manual_seed(7)
        weight = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
        values = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
        with comfy_kitchen.use_backend("cuda"):
            qweight, params = TensorWiseINT8Layout.quantize(
                weight, is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256
            )
        layer = W8A8ConvRotLinear(2048, 1024, False, dtype=torch.bfloat16, device="cuda")
        layer.weight.copy_(qweight)
        layer.weight_scale.copy_(params.scale)
        actual = layer(values).float()
        expected = torch.nn.functional.linear(values, weight).float()
        cosine = torch.nn.functional.cosine_similarity(actual, expected).item()
        self.assertGreater(cosine, 0.995)


if __name__ == "__main__":
    unittest.main()
