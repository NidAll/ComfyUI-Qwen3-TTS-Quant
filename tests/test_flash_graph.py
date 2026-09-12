import sys
import types
import unittest
from unittest.mock import patch

from q3tts_quant.runtime.flash_graph import _load_flash_kvcache, _window_size, inspect_flash_kvcache


class FlashGraphContractTests(unittest.TestCase):
    def test_qwen_window_maps_to_flash_inclusive_window(self):
        self.assertEqual(_window_size(None), (-1, -1))
        self.assertEqual(_window_size(4096), (4095, 0))

    def test_detects_compatible_kvcache_interface(self):
        module = types.ModuleType("flash_attn")

        def flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=None,
            v=None,
            cache_seqlens=None,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
        ):
            return q

        module.flash_attn_with_kvcache = flash_attn_with_kvcache
        with patch.dict(sys.modules, {"flash_attn": module}):
            self.assertIs(_load_flash_kvcache(), flash_attn_with_kvcache)
            self.assertTrue(inspect_flash_kvcache().available)

    def test_rejects_wrong_kvcache_interface(self):
        module = types.ModuleType("flash_attn")

        def flash_attn_with_kvcache(q):
            return q

        module.flash_attn_with_kvcache = flash_attn_with_kvcache
        with patch.dict(sys.modules, {"flash_attn": module}):
            status = inspect_flash_kvcache()
        self.assertFalse(status.available)
        self.assertIn("missing", status.reason)


if __name__ == "__main__":
    unittest.main()

