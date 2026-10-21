"""Anya main entry point: initializes configuration, logging, server, and avatar window."""

import argparse
import logging
import signal
import sys
import threading
from typing import Optional

from config import get_settings, setup_logging
from core.events import events
from core.resources import get_ram_usage
from ui.server import start_server, stop_server

logger = logging.getLogger("anya.main")

BANNER = r"""
   _____   _   ___     _____ 
  /  _  \ | \ | \ \   / / _ \
 /  /_\  \|  \| |\ \_/ / /_\ \
/    |    \ |\  | \   /   |   \
\____|__  /_| \_|  |_/\___|___/
        \/                     
      Local Emotional Voice Companion for macOS
============================================================
"""


def print_banner() -> None:
    """Print startup banner and current configuration summary."""
    settings = get_settings()
    ram = get_ram_usage()

    print(BANNER)
    print(f"  • LLM Model:         {settings.OLLAMA_MODEL}")
    print(f"  • Vision Model:      {settings.OLLAMA_VISION_MODEL}")
    print(f"  • Whisper Model:     {settings.WHISPER_MODEL}")
    print(f"  • Wake Model:        {settings.WAKE_MODEL} (phrases: {', '.join(settings.WAKE_PHRASES)})")
    print(f"  • Piper Voice:       {settings.PIPER_VOICE}")
    print(f"  • Live2D Model:      {settings.MODEL_PATH}")
    print(f"  • Server URL:        http://127.0.0.1:{settings.SERVER_PORT}/")
    print(f"  • Debug URL:         http://127.0.0.1:{settings.SERVER_PORT}/debug.html")
    print(f"  • FPS (Active/Idle): {settings.FPS_ACTIVE} / {settings.FPS_IDLE}")
    print(f"  • RAM Footprint:     {ram['rss_mb']:.1f} MB RSS (System: {ram['sys_percent']}%)")
    print(f"  • Quiet Hours:       {settings.QUIET_HOURS}")
    print(f"  • Debug Mode:        {'ENABLED' if settings.DEBUG else 'DISABLED'}")
    print("============================================================\n")


def _prewarm_macos_input() -> None:
    """Pre-warm pynput on the main thread to avoid macOS HIToolbox _dispatch_assert_queue_fail SIGTRAP."""
    try:
        from pynput import keyboard
        dummy = keyboard.Listener(on_press=lambda _: None, on_release=lambda _: None)
        dummy.start()
        dummy.stop()
    except Exception:
        pass


def run_app(headless: bool = False, debug_view: bool = False, reset_window: bool = False) -> int:
    """Run Anya: starts local HTTP server and PySide6 avatar overlay window."""
    _prewarm_macos_input()
    settings = get_settings()
    setup_logging(settings)
    print_banner()

    # 1. Start local HTTP asset and web server
    server = start_server(port=settings.SERVER_PORT)

    # 2. Check if running headless
    if headless:
        logger.info("Running in headless mode (--headless). Server is active.")
        from core.speaker import speak_stream
        import time

        def _run_headless_test() -> None:
            time.sleep(1.0)
            logger.info("[WP4 Test] Synthesizing happy sentence (headless)...")
            speak_stream(["I'm so glad to see you! Everything is wonderful today."], emotion="happy", sync=True)
            logger.info("[WP4 Test] Synthesizing sad sentence (headless)...")
            speak_stream(["Sometimes things feel a little quiet and lonely."], emotion="sad", sync=True)
            logger.info("[WP4 Test] Completed headless speech synthesis test.")

        threading.Thread(target=_run_headless_test, daemon=True).start()

        stop_event = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop_event.set())
        signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
        try:
            stop_event.wait()
        finally:
            stop_server(server)
        return 0

    try:
        import os

        # Allow WebAudio playback without requiring prior mouse click / user gesture
        existing_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
        if "--autoplay-policy=no-user-gesture-required" not in existing_flags:
            os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
                existing_flags + " --autoplay-policy=no-user-gesture-required"
            ).strip()

        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        from ui.window import AnyaAvatarWindow

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        window = AnyaAvatarWindow(
            debug_view=debug_view or settings.DEBUG,
            reset_window=reset_window,
        )
        window.show()

        # Connect quit signal
        events.quit_requested.connect(app.quit)

        # Temporary WP4 test: after startup say two sentences, one happy and one sad
        def _run_startup_speech_test(bridge: object) -> None:
            from core.speaker import speak_stream

            happy_sentence = "I'm so glad to see you! Everything is wonderful today."
            sad_sentence = "Sometimes things feel a little quiet and lonely."

            def _on_happy_done() -> None:
                try:
                    events.audio_ended.disconnect(_on_happy_done)
                except Exception:
                    pass
                logger.info("[WP4 Test] Happy utterance ended. Triggering sad sentence...")
                if hasattr(bridge, "set_emotion"):
                    bridge.set_emotion({"sad": 0.8}, transition_ms=500)
                speak_stream([sad_sentence], emotion="sad", intensity=0.9)

            logger.info("[WP4 Test] Starting verification: speaking happy sentence...")
            events.audio_ended.connect(_on_happy_done)
            if hasattr(bridge, "set_emotion"):
                bridge.set_emotion({"happy": 0.8}, transition_ms=500)
            speak_stream([happy_sentence], emotion="happy", intensity=0.9)

        _test_scheduled = False

        def _schedule_speech_test() -> None:
            nonlocal _test_scheduled
            if _test_scheduled:
                return
            _test_scheduled = True
            QTimer.singleShot(800, lambda: _run_startup_speech_test(window.bridge))

        if window.bridge.is_model_ready:
            _schedule_speech_test()
        else:
            window.bridge.model_ready.connect(_schedule_speech_test)

        # Allow Ctrl+C to interrupt Qt main loop
        timer = threading.Timer(1.0, lambda: None)
        timer.daemon = True
        timer.start()
        signal.signal(signal.SIGINT, lambda *_: app.quit())

        logger.info("Entering Qt event loop...")
        exit_code = app.exec()
        stop_server(server)
        return exit_code

    except ImportError as e:
        logger.warning("PySide6 GUI not available (%s). Running in headless mode.", e)
        stop_event = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop_event.set())
        stop_event.wait()
        stop_server(server)
        return 0


def main() -> int:
    """Parse CLI arguments and launch application."""
    parser = argparse.ArgumentParser(description="Anya Voice Companion")
    parser.add_argument("--headless", action="store_true", help="Run server only without Qt GUI window")
    parser.add_argument("--debug", action="store_true", help="Open Expression Lab debug panel directly")
    parser.add_argument("--reset-window", action="store_true",
                        help="Clear saved window position and size, then start centred")
    args = parser.parse_args()

    return run_app(headless=args.headless, debug_view=args.debug, reset_window=args.reset_window)


if __name__ == "__main__":
    sys.exit(main())
