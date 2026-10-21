from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.events import AppEvents, get_events


class TestEvents(unittest.TestCase):
    """Test suite verifying signal registration, emissions, and slots."""

    def test_singleton_get_events(self) -> None:
        """Verify get_events returns global singleton."""
        ev1 = get_events()
        ev2 = get_events()
        self.assertIs(ev1, ev2)

    def test_signal_emission_and_callbacks(self) -> None:
        """Verify each signal emits properly to connected callbacks."""
        ev = AppEvents()

        spoken = []
        emotion = []
        states = []
        quit_called = []
        audio_ended = []
        user_texts = []
        skill_payloads = []

        ev.speak_requested.connect(lambda text: spoken.append(text))
        ev.emotion_changed.connect(lambda emo: emotion.append(emo))
        ev.state_changed.connect(lambda st: states.append(st))
        ev.quit_requested.connect(lambda: quit_called.append(True))
        ev.audio_ended.connect(lambda: audio_ended.append(True))
        ev.user_text.connect(lambda txt: user_texts.append(txt))
        ev.skill_result.connect(lambda name, res: skill_payloads.append((name, res)))

        ev.speak_requested.emit("Hello Anya!")
        ev.emotion_changed.emit("happy")
        ev.state_changed.emit("LISTEN")
        ev.quit_requested.emit()
        ev.audio_ended.emit()
        ev.user_text.emit("What time is it?")
        ev.skill_result.emit("time_check", {"time": "12:00"})

        self.assertEqual(spoken, ["Hello Anya!"])
        self.assertEqual(emotion, ["happy"])
        self.assertEqual(states, ["LISTEN"])
        self.assertEqual(quit_called, [True])
        self.assertEqual(audio_ended, [True])
        self.assertEqual(user_texts, ["What time is it?"])
        self.assertEqual(skill_payloads, [("time_check", {"time": "12:00"})])


if __name__ == "__main__":
    unittest.main()
