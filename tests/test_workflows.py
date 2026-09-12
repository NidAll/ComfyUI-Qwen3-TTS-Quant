import json
import unittest
from pathlib import Path


class WorkflowTests(unittest.TestCase):
    def test_each_generation_mode_has_a_valid_workflow(self):
        root = Path(__file__).parents[1] / "examples"
        expected = {
            "voice_design_workflow.json": "Qwen3TTSVoiceDesign",
            "voice_clone_workflow.json": "Qwen3TTSVoiceClone",
            "custom_voice_workflow.json": "Qwen3TTSCustomVoice",
        }
        for filename, node_type in expected.items():
            with self.subTest(filename=filename):
                workflow = json.loads((root / filename).read_text(encoding="utf-8"))
                types = {node["type"] for node in workflow["nodes"]}
                self.assertIn(node_type, types)
                self.assertIn("Qwen3TTSQuantizedLoader", types)
                self.assertIn("SaveAudio", types)
                self.assertTrue(workflow["links"])

    def test_clone_workflow_connects_reference_audio(self):
        path = Path(__file__).parents[1] / "examples" / "voice_clone_workflow.json"
        workflow = json.loads(path.read_text(encoding="utf-8"))
        types = {node["type"] for node in workflow["nodes"]}
        self.assertIn("LoadAudio", types)
        self.assertIn("AUDIO", {link[5] for link in workflow["links"]})


if __name__ == "__main__":
    unittest.main()
