import unittest

from q3tts_quant.quantization.metadata import QuantizationMetadata, QuantizedLayer


class MetadataTests(unittest.TestCase):
    def test_round_trip(self):
        original = QuantizationMetadata(
            original_model="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            source_revision="abc123",
            layers={"talker.model.layers.0.self_attn.q_proj": QuantizedLayer(shape=[2048, 2048], component="talker", layer_kind="attention_q")},
        )
        restored = QuantizationMetadata.from_json(original.to_json())
        self.assertEqual(restored.to_dict(), original.to_dict())
        self.assertIn("q3tts.metadata", restored.safetensors_metadata())

    def test_variant_round_trip(self):
        original = QuantizationMetadata(
            architecture="qwen3_tts_base_v1",
            model_variant="base",
            original_model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        )
        restored = QuantizationMetadata.from_json(original.to_json())
        self.assertEqual(restored.architecture, "qwen3_tts_base_v1")
        self.assertEqual(restored.model_variant, "base")

    def test_legacy_voice_design_metadata_remains_loadable(self):
        raw = QuantizationMetadata().to_dict()
        raw.pop("model_variant")
        restored = QuantizationMetadata.from_dict(raw)
        self.assertEqual(restored.model_variant, "voice_design")


if __name__ == "__main__":
    unittest.main()
