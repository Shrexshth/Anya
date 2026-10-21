"""QWebChannel bridge connecting web avatar events and Python orchestrator.

New slots added for WP3:
  pushCursor(nx, ny) – push normalised cursor position [0,1]² to JS every 200 ms
  setSpeaking(flag)  – tell animation.js that TTS is playing (keeps active FPS)
  setEmotion(json)   – forward emotion weights dict to anya.setEmotion() in JS
  react(kind)        – trigger a one-shot react animation ('blush'|'shake'|'droop'|'bounce')
"""

import json
import logging
from typing import Any

from core.events import events

logger = logging.getLogger("anya.ui.bridge")

try:
    from PySide6.QtCore import QObject, Signal, Slot
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False

    def Slot(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Fallback decorator when PySide6 is not installed."""
        def decorator(func: Any) -> Any:
            return func
        return decorator

    class Signal:  # type: ignore[no-redef]
        def __init__(self, *args: Any) -> None:
            self._slots: list[Any] = []

        def connect(self, slot: Any) -> None:
            self._slots.append(slot)

        def emit(self, *args: Any) -> None:
            for s in list(self._slots):
                s(*args)

    class QObject:  # type: ignore[no-redef]
        def __init__(self, parent: Any = None) -> None:
            self._parent = parent


class WebBridge(QObject):
    """Bridge object exposed to JavaScript via QWebChannel."""

    # Python-side signals
    model_ready = Signal()
    audio_ended_signal = Signal(str)
    _js_dispatch_signal = Signal(str)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._is_model_ready = False
        self._page: Any = None  # set by window.py after WebEngineView is created

        # Ensure cross-thread JS calls are safely queued to the main Qt GUI thread
        try:
            from PySide6.QtCore import Qt
            self._js_dispatch_signal.connect(self._execute_run_js, Qt.ConnectionType.QueuedConnection)
        except Exception:
            self._js_dispatch_signal.connect(self._execute_run_js)

        # Connect core audio events to bridge slots for frontend delivery
        events.audio_clip_ready.connect(self.queue_audio)
        events.sound_requested.connect(self.play_sound)
        events.stop_speech_requested.connect(self.stop_speech)

    # ── internal ──────────────────────────────────────────────────────────────

    @Slot(str)
    def _execute_run_js(self, script: str) -> None:
        """Run arbitrary JS in the webview (executed strictly on the main thread)."""
        if self._page is not None:
            try:
                self._page.runJavaScript(script)
            except Exception as exc:
                logger.debug("runJavaScript failed: %s", exc)

    def _run_js(self, script: str) -> None:
        """Dispatch JS to the webview, safely marshaling to the main thread if called from worker."""
        import threading
        if threading.current_thread() is threading.main_thread():
            self._execute_run_js(script)
        else:
            self._js_dispatch_signal.emit(script)

    # ── slots called by JS ────────────────────────────────────────────────────

    @property
    def is_model_ready(self) -> bool:
        """Return True once the Live2D avatar finishes initialising in webview."""
        return self._is_model_ready

    @Slot()
    def modelReady(self) -> None:
        """Invoked by avatar.js once Live2D model is loaded and mounted."""
        logger.info("Live2D Avatar signaled: modelReady")
        self._is_model_ready = True
        self.model_ready.emit()

    @Slot(str)
    def audioEnded(self, audio_id: Any = "") -> None:
        """Invoked by web client when TTS audio playback finishes."""
        id_str = str(audio_id)
        logger.debug("Live2D audio playback ended (id: %s)", id_str)
        self.audio_ended_signal.emit(id_str)
        events.audio_ended.emit()

    # ── methods called by Python → pushes data into JS ────────────────────────

    def push_cursor(self, nx: float, ny: float) -> None:
        """Push normalised cursor position [0,1]² to animation.js.

        Call this from a 200 ms QTimer in the main thread.  Values are clamped
        to [0, 1] before forwarding so JS never receives out-of-range input.
        """
        nx_safe = max(0.0, min(1.0, float(nx)))
        ny_safe = max(0.0, min(1.0, float(ny)))
        self._run_js(
            f"if(window.anya&&window.anya.setCursorNorm)"
            f"window.anya.setCursorNorm({nx_safe:.4f},{ny_safe:.4f});"
        )

    def set_speaking(self, speaking: bool) -> None:
        """Notify animation.js that speech is playing (keeps active FPS)."""
        flag = "true" if speaking else "false"
        self._run_js(
            f"if(window.anya&&window.anya.setSpeaking)window.anya.setSpeaking({flag});"
        )
        logger.debug("set_speaking → %s", speaking)

    def set_emotion(self, weights: dict[str, float], transition_ms: int = 600) -> None:
        """Push an emotion-weight dict to blend.js.

        Args:
            weights:       dict mapping emotion name → weight 0-1
                           e.g. {"sad": 0.6, "shy": 0.3}
            transition_ms: ease duration in milliseconds (default 600)
        """
        # Validate: only str keys, numeric values 0-1.
        safe: dict[str, float] = {}
        for k, v in weights.items():
            try:
                safe[str(k)] = max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                logger.warning("set_emotion: invalid weight for '%s': %r – skipped", k, v)
        weights_json = json.dumps(safe)
        self._run_js(
            f"if(window.anya&&window.anya.setEmotion)"
            f"window.anya.setEmotion({weights_json},{int(transition_ms)});"
        )
        logger.info("set_emotion %s (transition %d ms)", safe, transition_ms)

    def react(self, kind: str) -> None:
        """Trigger a one-shot react animation in animation.js.

        Args:
            kind: one of 'blush', 'shake', 'droop', 'bounce'
        """
        allowed = {"blush", "shake", "droop", "bounce"}
        if kind not in allowed:
            logger.warning("react: unknown kind '%s' (allowed: %s)", kind, allowed)
            return
        # Pass as a fixed string, never via formatting that could inject code.
        self._run_js(
            f"if(window.anya&&window.anya.react)window.anya.react('{kind}');"
        )
        logger.info("react → %s", kind)

    def set_sleep(self, sleeping: bool) -> None:
        """Put the avatar into or out of sleep mode."""
        flag = "true" if sleeping else "false"
        self._run_js(
            f"if(window.anya&&window.anya.setSleep)window.anya.setSleep({flag});"
        )
        logger.info("set_sleep → %s", sleeping)

    @Slot(str, str, bool)
    @Slot(str, str)
    @Slot(str)
    def queue_audio(self, url: str, utterance_id: str = "", is_last: bool = True) -> None:
        """Push an audio clip URL to avatar.js audio queue for playback."""
        url_json = json.dumps(url)
        id_json = json.dumps(utterance_id)
        last_str = "true" if is_last else "false"
        self._run_js(
            f"if(window.anya&&window.anya.queueAudio)"
            f"window.anya.queueAudio({url_json},{id_json},{last_str});"
        )
        logger.debug("queue_audio → %s (id: %s, is_last: %s)", url, utterance_id, is_last)

    @Slot(str)
    def play_sound(self, sound_name: str) -> None:
        """Request avatar.js to play a short sound effect from assets/sounds/."""
        sound_json = json.dumps(sound_name)
        self._run_js(
            f"if(window.anya&&window.anya.playSound)"
            f"window.anya.playSound({sound_json});"
        )
        logger.info("play_sound → %s", sound_name)

    @Slot()
    def stop_speech(self) -> None:
        """Halt ongoing audio playback and clear speech queue in frontend."""
        self._run_js(
            "if(window.anya&&window.anya.stopSpeech)window.anya.stopSpeech();"
        )
        logger.info("stop_speech triggered")
