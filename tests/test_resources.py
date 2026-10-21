from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.resources import ResourceManager, get_ram_usage


class TestResources(unittest.TestCase):
    """Test suite verifying lazy loading, idle timeout, and unloader hooks."""

    def test_lazy_loading_and_manual_unload(self) -> None:
        """Verify model loads on first use and calls unloader on unload."""
        mgr = ResourceManager()
        load_count = 0
        unloaded_count = 0

        def fake_loader() -> dict:
            nonlocal load_count
            load_count += 1
            return {"weights": [1, 2, 3]}

        def fake_unloader(instance: dict) -> None:
            nonlocal unloaded_count
            unloaded_count += 1
            instance.clear()

        mgr.register("test_model", fake_loader, fake_unloader, timeout_s=300)
        self.assertFalse(mgr.is_loaded("test_model"))
        self.assertEqual(load_count, 0)

        # First use triggers load
        model = mgr.get("test_model")
        self.assertEqual(model, {"weights": [1, 2, 3]})
        self.assertEqual(load_count, 1)
        self.assertTrue(mgr.is_loaded("test_model"))

        # Subsequent get returns cached instance without reloading
        cached = mgr.get("test_model")
        self.assertEqual(cached, model)
        self.assertEqual(load_count, 1)

        # Unload
        success = mgr.unload("test_model")
        self.assertTrue(success)
        self.assertEqual(unloaded_count, 1)
        self.assertFalse(mgr.is_loaded("test_model"))

    def test_idle_timeout_automatic_unload(self) -> None:
        """Verify model unloads automatically when idle timeout expires."""
        mgr = ResourceManager()
        unloaded = False

        def fake_unloader(inst: str) -> None:
            nonlocal unloaded
            unloaded = True

        # Fast timeout of 0.2 seconds for unit test
        mgr.register("whisper_test", lambda: "whisper_instance", fake_unloader, timeout_s=0.2)
        inst = mgr.get("whisper_test")
        self.assertEqual(inst, "whisper_instance")
        self.assertTrue(mgr.is_loaded("whisper_test"))

        # Wait for timeout to fire
        time.sleep(0.35)
        self.assertTrue(unloaded)
        self.assertFalse(mgr.is_loaded("whisper_test"))

    def test_ram_usage_helper(self) -> None:
        """Verify RAM usage helper returns positive numerical values."""
        ram = get_ram_usage()
        self.assertIn("rss_mb", ram)
        self.assertIn("sys_percent", ram)
        self.assertIsInstance(ram["rss_mb"], (int, float))
        self.assertGreaterEqual(ram["rss_mb"], 0.0)


if __name__ == "__main__":
    unittest.main()
