import unittest

from q3tts_quant.adapters.qwen3_tts import (
    Qwen3TTS17BAdapter,
    Qwen3TTSBaseAdapter,
    Qwen3TTSCustomVoiceAdapter,
    Qwen3TTSVoiceDesignAdapter,
)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = Qwen3TTSVoiceDesignAdapter()

    def test_detects_voice_design(self):
        config = {
            "model_type": "qwen3_tts",
            "tts_model_size": "1b7",
            "tts_model_type": "voice_design",
            "talker_config": {"num_code_groups": 16, "hidden_size": 2048},
        }
        self.assertTrue(self.adapter.detect(config))

    def test_detects_all_supported_17b_variants(self):
        for variant, adapter_type in (
            ("base", Qwen3TTSBaseAdapter),
            ("custom_voice", Qwen3TTSCustomVoiceAdapter),
            ("voice_design", Qwen3TTSVoiceDesignAdapter),
        ):
            config = {
                "model_type": "qwen3_tts",
                "tts_model_size": "1b7",
                "tts_model_type": variant,
                "talker_config": {"num_code_groups": 16, "hidden_size": 2048},
            }
            with self.subTest(variant=variant):
                adapter = Qwen3TTS17BAdapter.from_config(config)
                self.assertTrue(adapter_type().detect(config))
                self.assertEqual(adapter.variant, variant)

    def test_rejects_06b_family(self):
        config = {
            "model_type": "qwen3_tts",
            "tts_model_size": "0b6",
            "tts_model_type": "base",
            "talker_config": {"num_code_groups": 16, "hidden_size": 1024},
        }
        self.assertFalse(Qwen3TTSBaseAdapter().detect(config))

    def test_quantizes_recurrent_aligned_linear(self):
        decision = self.adapter.classify_tensor("talker.model.layers.0.self_attn.q_proj.weight", (2048, 2048))
        self.assertTrue(decision.quantize)
        self.assertEqual(decision.layer_kind, "attention_q")

    def test_keeps_heads_and_embeddings_dense(self):
        for name, shape in (
            ("talker.codec_head.weight", (3072, 2048)),
            ("talker.model.text_embedding.weight", (151936, 2048)),
        ):
            with self.subTest(name=name):
                self.assertFalse(self.adapter.classify_tensor(name, shape).quantize)


if __name__ == "__main__":
    unittest.main()
