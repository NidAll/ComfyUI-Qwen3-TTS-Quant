import json
import tempfile
import unittest
from pathlib import Path

from q3tts_quant.quantization.source import resolve_model_source


class SourceTests(unittest.TestCase):
    def test_safetensors_index_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "part.safetensors").write_bytes(b"stub")
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"x": "part.safetensors"}}), encoding="utf-8")
            source = resolve_model_source(str(index))
            self.assertEqual(source.weight_files, (root / "part.safetensors",))


if __name__ == "__main__":
    unittest.main()
