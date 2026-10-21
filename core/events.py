"""Qt signals shared across threads for thread-safe UI and audio coordination."""

import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger("anya.core.events")

try:
    from PySide6.QtCore import QObject, Signal
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False

    class Signal:  # type: ignore[no-redef]
        """Fallback lightweight Signal when PySide6 is not installed."""

        def __init__(self, *arg_types: Any) -> None:
            self.arg_types = arg_types
            self._slots: List[Callable[..., Any]] = []

        def connect(self, slot: Callable[..., Any]) -> None:
            """Connect a callback slot to this signal."""
            if slot not in self._slots:
                self._slots.append(slot)

        def disconnect(self, slot: Callable[..., Any]) -> None:
            """Disconnect a callback slot from this signal."""
            if slot in self._slots:
                self._slots.remove(slot)

        def emit(self, *args: Any, **kwargs: Any) -> None:
            """Invoke all connected callback slots with provided arguments."""
            for slot in list(self._slots):
                try:
                    slot(*args, **kwargs)
                except Exception as exc:
                    logger.exception("Error emitting fallback signal to %s: %s", slot, exc)

    class QObject:  # type: ignore[no-redef]
        """Fallback base class when PySide6 is not installed."""

        def __init__(self, parent: Any = None) -> None:
            self._parent = parent


class AppEvents(QObject):
    """Central event hub coordinating core worker threads and Qt UI."""

    # Trigger speech output with given text
    speak_requested = Signal(str)

    # Broadcast emotional state transition (e.g. happy, sad, angry)
    emotion_changed = Signal(str)

    # Broadcast state machine phase change (IDLE, LISTEN, THINK, SPEAK)
    state_changed = Signal(str)

    # Instruct application components to shut down gracefully
    quit_requested = Signal()

    # Inform orchestrator that audio playback has finished
    audio_ended = Signal()

    # Emitted when speech is transcribed or user types text
    user_text = Signal(str)

    # Emitted when a skill finishes execution (skill_name, result_payload)
    skill_result = Signal(str, dict)

    # Audio playback and streaming coordination (WP4)
    # audio_clip_ready: (clip_url, utterance_id, is_last) pushed to frontend queue
    audio_clip_ready = Signal(str, str, bool)

    # Request playback of short sound effect from assets/sounds/ (e.g. laugh, sigh)
    sound_requested = Signal(str)

    # Request immediate cancellation of speech queue and playback
    stop_speech_requested = Signal()

    # Request streaming speech synthesis: (sentences_iterable, emotion_name, intensity)
    speak_stream_requested = Signal(object, str, float)


events = AppEvents()


def get_events() -> AppEvents:
    """Return the global AppEvents singleton."""
    return events
