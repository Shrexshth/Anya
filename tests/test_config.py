import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Settings, get_settings, load_settings, setup_logging


class TestConfig(unittest.TestCase):
    """Test suite verifying dataclass Settings loading and defaults."""

    def test_default_settings(self) -> None:
        """Verify default settings match WP1 specification."""
        settings = Settings()
        self.assertEqual(settings.OLLAMA_MODEL, "qwen2.5:7b-instruct")
        self.assertEqual(settings.OLLAMA_VISION_MODEL, "qwen2.5vl:3b")
        self.assertEqual(settings.WHISPER_MODEL, "base.en")
        self.assertEqual(settings.WAKE_MODEL, "tiny.en")
        self.assertEqual(settings.WAKE_PHRASES, ["hey anya", "anya"])
        self.assertEqual(settings.CONVERSATION_WINDOW_S, 20)
        self.assertEqual(settings.SERVER_PORT, 8000)
        self.assertEqual(settings.MODEL_PATH, "assets/model")
        self.assertEqual(settings.PIPER_VOICE, "assets/voices/en_US-lessac-medium.onnx")
        self.assertFalse(settings.BARGE_IN)
        self.assertEqual(settings.PUSH_TO_TALK_KEY, "space")
        self.assertEqual(settings.HISTORY_TURNS, 6)
        self.assertEqual(settings.NUM_CTX, 4096)
        self.assertEqual(settings.LLM_IDLE_UNLOAD_S, 300)
        self.assertEqual(settings.FPS_ACTIVE, 30)
        self.assertEqual(settings.FPS_IDLE, 10)
        self.assertFalse(settings.DRY_RUN)
        self.assertFalse(settings.DEBUG)
        self.assertEqual(settings.QUIET_HOURS, "23:00-07:00")
        self.assertEqual(settings.MAX_PROACTIVE_PER_HOUR, 3)
        self.assertEqual(settings.BLOCKED_FOCUS_APPS, ["Discord"])
        self.assertIsNone(settings.TELEGRAM_TOKEN)
        self.assertIsNone(settings.TELEGRAM_ALLOWED_ID)
        self.assertIsNone(settings.SPOTIFY_CLIENT_ID)
        self.assertIsNone(settings.SPOTIFY_CLIENT_SECRET)

    def test_env_override(self) -> None:
        """Verify custom .env file overrides default values accurately."""
        with TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "SERVER_PORT=9090\n"
                "DEBUG=true\n"
                "BARGE_IN=1\n"
                "WAKE_PHRASES=[\"anya-chan\", \"hi anya\"]\n"
                "BLOCKED_FOCUS_APPS=Slack,Discord,Steam\n"
                "SPOTIFY_CLIENT_ID=test_id\n",
                encoding="utf-8",
            )
            settings = load_settings(env_file=env_file)
            self.assertEqual(settings.SERVER_PORT, 9090)
            self.assertTrue(settings.DEBUG)
            self.assertTrue(settings.BARGE_IN)
            self.assertEqual(settings.WAKE_PHRASES, ["anya-chan", "hi anya"])
            self.assertEqual(settings.BLOCKED_FOCUS_APPS, ["Slack", "Discord", "Steam"])
            self.assertEqual(settings.SPOTIFY_CLIENT_ID, "test_id")

    def test_setup_logging(self) -> None:
        """Verify unified logging configures console and file output."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test_anya.log"
            settings = Settings(DEBUG=True)
            logger = setup_logging(settings=settings, log_file=log_file)
            logger.info("Test log line")
            self.assertTrue(log_file.exists())
            self.assertIn("Test log line", log_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
