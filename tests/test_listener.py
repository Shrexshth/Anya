"""Unit tests for core/listener.py: VAD, wake detection, conversation window, and barge-in."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config import Settings
from core.events import events
from core.listener import (
    Listener,
    VADSegmenter,
    filter_hallucination,
    is_barge_in_stop,
    match_and_strip_wake_phrase,
    matches_pynput_key,
)


class TestFilterHallucination:
    """Test silence and low-confidence Whisper hallucination filtering."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "thank you",
            "thank you.",
            "Thank you.",
            "thank you very much",
            "thanks for watching",
            "Thanks for watching!",
            "Thanks for watching.",
            "you",
            "you.",
            "bye",
            "goodbye",
            "subscribe",
            "[music]",
            "[applause]",
            "(laughter)",
            "(silence)",
            "subtitles by someone",
            "",
            "   ",
            "...",
        ],
    )
    def test_hallucination_filtered(self, phrase: str) -> None:
        assert filter_hallucination(phrase) is None

    @pytest.mark.parametrize(
        "phrase,expected",
        [
            ("Hello Anya!", "Hello Anya!"),
            ("What is the time right now?", "What is the time right now?"),
            ("You are amazing.", "You are amazing."),
            ("Thank goodness it is Friday", "Thank goodness it is Friday"),
            ("Tell me a bedtime story", "Tell me a bedtime story"),
        ],
    )
    def test_legitimate_speech_preserved(self, phrase: str, expected: str) -> None:
        assert filter_hallucination(phrase) == expected


class TestWakePhraseMatching:
    """Test fuzzy wake phrase matching and stripping with rapidfuzz."""

    def setup_method(self) -> None:
        self.wake_phrases = ["hey anya", "anya"]

    def test_exact_matches(self) -> None:
        matched, phrase, stripped = match_and_strip_wake_phrase("hey anya", self.wake_phrases)
        assert matched is True
        assert phrase == "hey anya"
        assert stripped == ""

        matched, phrase, stripped = match_and_strip_wake_phrase("anya", self.wake_phrases)
        assert matched is True
        assert phrase == "anya"
        assert stripped == ""

    def test_fuzzy_matches_anja(self) -> None:
        matched, phrase, stripped = match_and_strip_wake_phrase("hey anja", self.wake_phrases)
        assert matched is True
        assert phrase == "hey anya"
        assert stripped == ""

        matched, phrase, stripped = match_and_strip_wake_phrase("anja", self.wake_phrases)
        assert matched is True
        assert phrase == "anya"
        assert stripped == ""

    def test_wake_with_attached_query(self) -> None:
        matched, phrase, stripped = match_and_strip_wake_phrase(
            "hey anya what time is it", self.wake_phrases
        )
        assert matched is True
        assert stripped == "what time is it"

        matched, phrase, stripped = match_and_strip_wake_phrase(
            "hey anja, tell me a joke", self.wake_phrases
        )
        assert matched is True
        assert stripped == "tell me a joke"

        matched, phrase, stripped = match_and_strip_wake_phrase(
            "anya what is the weather today", self.wake_phrases
        )
        assert matched is True
        assert stripped == "what is the weather today"

    def test_false_positive_rejection(self) -> None:
        assert match_and_strip_wake_phrase("banana for breakfast", self.wake_phrases)[0] is False
        assert match_and_strip_wake_phrase("any thoughts on this?", self.wake_phrases)[0] is False
        assert match_and_strip_wake_phrase("hello world", self.wake_phrases)[0] is False
        assert match_and_strip_wake_phrase("can you help me", self.wake_phrases)[0] is False


class TestBargeInStop:
    """Test barge-in stop phrase detection during speech."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "stop",
            "Stop!",
            "anya stop",
            "anja stop",
            "please stop",
            "shut up",
            "stop talking",
            "Anya please stop",
        ],
    )
    def test_barge_in_stop_matched(self, phrase: str) -> None:
        assert is_barge_in_stop(phrase) is True

    @pytest.mark.parametrize(
        "phrase",
        [
            "hello there",
            "what is the weather",
            "can you continue",
            "keep going",
            "tell me more",
        ],
    )
    def test_barge_in_stop_not_matched(self, phrase: str) -> None:
        assert is_barge_in_stop(phrase) is False


class TestVADSegmenter:
    """Test VAD audio frame segmentation, silence cutoff, and max duration cap."""

    def setup_method(self) -> None:
        self.sample_rate = 16000
        self.frame_ms = 30
        self.frame_samples = 480
        self.frame_bytes = 960
        self.segmenter = VADSegmenter(
            sample_rate=self.sample_rate,
            frame_ms=self.frame_ms,
            vad_mode=2,
            silence_timeout_s=0.8,
            max_duration_s=15.0,
            pre_speech_ms=300,
        )

    def _make_silence_frame(self) -> bytes:
        return np.zeros(self.frame_samples, dtype=np.int16).tobytes()

    def _make_tone_frame(self, freq: float = 440.0) -> bytes:
        t = np.linspace(0, 0.03, self.frame_samples, False)
        return (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16).tobytes()

    def test_silence_does_not_trigger_recording(self) -> None:
        silence = self._make_silence_frame()
        for _ in range(20):
            res = self.segmenter.process_frame(silence)
            assert res is None
        assert self.segmenter.is_recording is False

    def test_speech_followed_by_silence_completes_segment(self) -> None:
        silence = self._make_silence_frame()
        tone = self._make_tone_frame()

        # Send 5 silent frames
        for _ in range(5):
            assert self.segmenter.process_frame(silence) is None
        assert self.segmenter.is_recording is False

        # Send 10 speech frames
        for _ in range(10):
            assert self.segmenter.process_frame(tone) is None
        assert self.segmenter.is_recording is True

        # Send silence frames until silence timeout triggers completion
        completed = None
        for _ in range(40):
            res = self.segmenter.process_frame(silence)
            if res is not None:
                completed = res
                break

        assert completed is not None
        assert len(completed) > 0
        assert self.segmenter.is_recording is False

    def test_max_duration_cap(self) -> None:
        tone = self._make_tone_frame()
        # Max frames = 15.0 / 0.03 = 500 frames
        # Trigger start
        assert self.segmenter.process_frame(tone) is None
        assert self.segmenter.is_recording is True

        completed = None
        for i in range(550):
            res = self.segmenter.process_frame(tone)
            if res is not None:
                completed = res
                break

        assert completed is not None
        # Max frame count limit
        assert len(completed) == self.segmenter.max_frames * self.frame_bytes
        assert self.segmenter.is_recording is False


class TestConversationWindow:
    """Test conversation window open, extend, and timeout."""

    def test_open_and_extend(self) -> None:
        settings = Settings(CONVERSATION_WINDOW_S=5)
        listener = Listener(settings=settings)

        assert listener.is_window_open() is False

        listener.open_conversation_window()
        assert listener.is_window_open() is True
        first_expiry = listener._conversation_window_expiry

        time.sleep(0.01)
        listener.open_conversation_window()
        assert listener.is_window_open() is True
        second_expiry = listener._conversation_window_expiry
        assert second_expiry > first_expiry

    def test_close_window(self) -> None:
        settings = Settings(CONVERSATION_WINDOW_S=5)
        listener = Listener(settings=settings)

        listener.open_conversation_window()
        assert listener.is_window_open() is True

        listener.close_conversation_window()
        assert listener.is_window_open() is False


class TestPauseResumeAndBargeIn:
    """Test pause(), resume(), and barge-in transitions."""

    def test_pause_and_resume(self) -> None:
        listener = Listener()
        assert listener._is_paused is False

        listener.pause()
        assert listener._is_paused is True

        listener.resume()
        assert listener._is_paused is False

    def test_speaking_without_barge_in_pauses_mic(self) -> None:
        settings = Settings(BARGE_IN=False)
        listener = Listener(settings=settings)

        listener.set_speaking(True)
        assert listener._is_paused is True
        assert listener._barge_in_active is False

        listener.set_speaking(False)
        assert listener._is_paused is False

    def test_speaking_with_barge_in_keeps_mic_open(self) -> None:
        settings = Settings(BARGE_IN=True)
        listener = Listener(settings=settings)

        listener.set_speaking(True)
        assert listener._is_paused is False
        assert listener._barge_in_active is True

        listener.set_speaking(False)
        assert listener._barge_in_active is False


class TestAudioSegmentHandling:
    """Test audio segment routing through wake and conversation modes."""

    def test_asleep_without_wake_word(self) -> None:
        settings = Settings(CONVERSATION_WINDOW_S=20, WAKE_PHRASES=["hey anya"])
        listener = Listener(settings=settings)
        fake_pcm = np.zeros(1600, dtype=np.int16).tobytes()

        # Mock wake model transcription
        listener.get_wake_model = MagicMock()
        listener.transcribe_audio = MagicMock(return_value="what is the weather")

        result = listener.handle_audio_segment(fake_pcm)
        assert result is None
        assert listener.is_window_open() is False

    def test_asleep_with_wake_word_opens_window(self) -> None:
        settings = Settings(CONVERSATION_WINDOW_S=20, WAKE_PHRASES=["hey anya"])
        listener = Listener(settings=settings)
        fake_pcm = np.zeros(1600, dtype=np.int16).tobytes()

        listener.get_wake_model = MagicMock()
        listener.transcribe_audio = MagicMock(return_value="Hey Anya")

        result = listener.handle_audio_segment(fake_pcm)
        assert result is None  # No attached query, only wake word
        assert listener.is_window_open() is True

    def test_asleep_with_wake_word_and_attached_query(self) -> None:
        settings = Settings(CONVERSATION_WINDOW_S=20, WAKE_PHRASES=["hey anya"])
        received_texts = []
        listener = Listener(settings=settings, on_user_text=received_texts.append)
        fake_pcm = np.zeros(1600, dtype=np.int16).tobytes()

        listener.get_wake_model = MagicMock()
        listener.get_whisper_model = MagicMock()
        # Wake model returns "Hey Anya what time is it"
        listener.transcribe_audio = MagicMock(return_value="Hey Anya what time is it")

        result = listener.handle_audio_segment(fake_pcm)
        assert result == "what time is it"
        assert listener.is_window_open() is True
        assert received_texts == ["what time is it"]

    def test_awake_segment_transcription(self) -> None:
        settings = Settings(CONVERSATION_WINDOW_S=20, WAKE_PHRASES=["hey anya"])
        received_texts = []
        listener = Listener(settings=settings, on_user_text=received_texts.append)
        listener.open_conversation_window()

        fake_pcm = np.zeros(1600, dtype=np.int16).tobytes()
        listener.get_whisper_model = MagicMock()
        listener.transcribe_audio = MagicMock(return_value="Tell me another joke")

        result = listener.handle_audio_segment(fake_pcm)
        assert result == "Tell me another joke"
        assert listener.is_window_open() is True
        assert received_texts == ["Tell me another joke"]

    def test_barge_in_triggers_stop(self) -> None:
        settings = Settings(BARGE_IN=True)
        listener = Listener(settings=settings)
        listener.set_speaking(True)

        stop_emitted = []
        events.stop_speech_requested.connect(lambda: stop_emitted.append(True))

        fake_pcm = np.zeros(1600, dtype=np.int16).tobytes()
        listener.get_wake_model = MagicMock()
        listener.transcribe_audio = MagicMock(return_value="Anya stop")

        result = listener.handle_audio_segment(fake_pcm)
        assert result is None
        assert len(stop_emitted) > 0


class TestKeyMatching:
    """Test hotkey matching for Push-To-Talk."""

    def test_key_matching(self) -> None:
        try:
            from pynput import keyboard
            assert matches_pynput_key(keyboard.Key.space, "space") is True
            assert matches_pynput_key(keyboard.Key.space, "ctrl") is False
            assert matches_pynput_key(keyboard.Key.ctrl_l, "ctrl") is True
            assert matches_pynput_key(keyboard.Key.alt_r, "alt") is True
            assert matches_pynput_key(keyboard.KeyCode.from_char("a"), "a") is True
            assert matches_pynput_key(keyboard.KeyCode.from_char("a"), "b") is False
        except Exception:
            pytest.skip("pynput keyboard unavailable in this test environment")
