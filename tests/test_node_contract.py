import ast
import unittest
from pathlib import Path


class NodeContractTests(unittest.TestCase):
    def test_all_expected_nodes_are_registered(self):
        source = Path(__file__).parents[1] / "nodes" / "__init__.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        expected = {
            "Qwen3TTSQuantizedLoader", "Qwen3TTSVoiceDesign", "Qwen3TTSRuntimeSettings",
            "Qwen3TTSDiagnostics", "Qwen3TTSBenchmark", "Qwen3TTSUnload",
        }
        self.assertTrue(expected.issubset(literals))


if __name__ == "__main__":
    unittest.main()
