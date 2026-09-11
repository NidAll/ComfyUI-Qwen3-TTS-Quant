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


if __name__ == "__main__":
    unittest.main()
