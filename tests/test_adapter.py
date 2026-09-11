import unittest

from q3tts_quant.adapters.qwen3_tts import Qwen3TTSVoiceDesignAdapter


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = Qwen3TTSVoiceDesignAdapter()

    def test_detects_voice_design(self):
        config = {"model_type": "qwen3_tts", "tts_model_type": "voice_design", "talker_config": {"num_code_groups": 16}}
        self.assertTrue(self.adapter.detect(config))

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
