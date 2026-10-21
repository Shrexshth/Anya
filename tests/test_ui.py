"""Unit tests for ui/server.py, ui/bridge.py, and model file integrity."""

import json
from pathlib import Path
import sys
import unittest
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.events import events
from ui.bridge import WebBridge
from ui.server import start_server, stop_server


class TestUIAndModel(unittest.TestCase):
    """Test suite verifying UI server routing, MIME types, bridge slots, and model schema."""

    @classmethod
    def setUpClass(cls) -> None:
        """Start local HTTP server on a test port."""
        cls.port = 8765
        cls.server = start_server(host="127.0.0.1", port=cls.port)
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls) -> None:
        """Stop local HTTP server."""
        stop_server(cls.server)

    def _open_url(self, url: str) -> urllib.request.urlopen:
        """Helper to open URL with explicit Connection: close header."""
        req = urllib.request.Request(url, headers={"Connection": "close"})
        return urllib.request.urlopen(req)

    def test_serve_index_html(self) -> None:
        """Verify root URL serves index.html with HTML MIME type."""
        url = f"{self.base_url}/"
        with self._open_url(url) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/html", resp.headers.get("Content-Type", ""))
            content = resp.read().decode("utf-8")
            self.assertIn("Anya Live2D Companion", content)

    def test_serve_debug_html(self) -> None:
        """Verify /debug.html serves debug panel with expression buttons."""
        url = f"{self.base_url}/debug.html"
        with self._open_url(url) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/html", resp.headers.get("Content-Type", ""))
            content = resp.read().decode("utf-8")
            self.assertIn("Anya Expressions", content)

    def test_serve_api_config(self) -> None:
        """Verify /api/config returns JSON configuration."""
        url = f"{self.base_url}/api/config"
        with self._open_url(url) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("application/json", resp.headers.get("Content-Type", ""))
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("fps_active", data)
            self.assertIn("fps_idle", data)

    def test_serve_model3_json(self) -> None:
        """Verify model3.json is served with accurate schema and Expressions list."""
        url = f"{self.base_url}/assets/model/Anya_model.model3.json"
        with self._open_url(url) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("application/json", resp.headers.get("Content-Type", ""))
            data = json.loads(resp.read().decode("utf-8"))

            file_refs = data.get("FileReferences", {})
            self.assertIn("Expressions", file_refs)
            self.assertGreater(len(file_refs["Expressions"]), 0)

            # Check LipSync group
            groups = data.get("Groups", [])
            lipsync_group = next((g for g in groups if g.get("Name") == "LipSync"), None)
            self.assertIsNotNone(lipsync_group)
            self.assertEqual(lipsync_group.get("Ids"), ["ParamMouthOpenY"])

    def test_expression_files_parameters_removed(self) -> None:
        """Verify ParamMouthOpenY, ParamEyeLOpen, ParamEyeROpen are removed from exp3.json."""
        url = f"{self.base_url}/assets/model/expressions/Angry.exp3.json"
        with self._open_url(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            param_ids = [p["Id"] for p in data.get("Parameters", [])]
            self.assertNotIn("ParamMouthOpenY", param_ids)
            self.assertNotIn("ParamEyeLOpen", param_ids)
            self.assertNotIn("ParamEyeROpen", param_ids)

    def test_serve_vendor_pixi(self) -> None:
        """Verify vendor JavaScript files are served with correct MIME type."""
        url = f"{self.base_url}/vendor/pixi.min.js"
        with self._open_url(url) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("application/javascript", resp.headers.get("Content-Type", ""))

    def test_web_bridge_slots(self) -> None:
        """Verify WebBridge modelReady and audioEnded slot handling."""
        bridge = WebBridge()
        self.assertFalse(bridge.is_model_ready)

        bridge.modelReady()
        self.assertTrue(bridge.is_model_ready)

        audio_ended_events = []
        events.audio_ended.connect(lambda: audio_ended_events.append(True))
        bridge.audioEnded("test_track_1")
        self.assertEqual(len(audio_ended_events), 1)


if __name__ == "__main__":
    unittest.main()
