import unittest

from q3tts_quant.errors import InsufficientVRAMError
from q3tts_quant.runtime.memory import build_memory_plan, kv_cache_mib


class MemoryTests(unittest.TestCase):
    def test_voice_design_kv_formula(self):
        self.assertEqual(kv_cache_mib(2048), 224)

    def test_plan_keeps_decode_path_resident(self):
        plan = build_memory_plan(3700 * 2**20, 1700 * 2**20, 2048, 320, True, "speed")
        self.assertEqual(plan.talker, "GPU resident")
        self.assertEqual(plan.predictor, "GPU resident")
        self.assertEqual(plan.codec, "FP16 GPU resident")

    def test_rejects_unsafe_plan(self):
        with self.assertRaises(InsufficientVRAMError):
            build_memory_plan(2000 * 2**20, 1700 * 2**20, 2048, 320, True, "speed")


if __name__ == "__main__":
    unittest.main()
