"""Unit tests for core/voice_style.py and core/speaker.py."""

from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import get_settings
from core.events import events
from core.speaker import (
    _fallback_synthesize_say,
    cleanup_cache,
    compute_cache_path,
    speak_stream,
    split_into_sentences,
    synthesize,
)
from core.voice_style import (
    VoiceStyle,
    apply_pitch_shift,
    get_voice_style,
    load_voice_styles,
)
from ui.bridge import WebBridge


class TestVoiceStyle(unittest.TestCase):
    """Test suite verifying voice style mappings, interpolation, and pitch shifting."""

    def test_load_voice_styles_from_yaml(self) -> None:
        """Verify styles are loaded dynamically from data/emotion.yaml."""
        styles = load_voice_styles()
        self.assertIn("baseline", styles)
        self.assertIn("angry", styles)
        self.assertIn("sad", styles)
        self.assertIn("romantic", styles)
        self.assertIn("excited", styles)

    def test_baseline_style(self) -> None:
        """Verify neutral emotion yields baseline parameters."""
        style = get_voice_style("neutral", intensity=1.0)
        self.assertAlmostEqual(style.length_scale, 1.0, places=2)
        self.assertAlmostEqual(style.noise_scale, 0.667, places=2)
        self.assertAlmostEqual(style.noise_w, 0.8, places=2)
        self.assertAlmostEqual(style.pitch_shift, 1.0, places=2)

    def test_emotion_characteristics(self) -> None:
        """Verify prompt examples: angry faster/sharper, sad slower/flatter, romantic softer, excited livelier."""
        angry = get_voice_style("angry", intensity=1.0)
        self.assertLess(angry.length_scale, 1.0)  # faster
        self.assertGreater(angry.noise_scale, 0.667)  # sharper

        sad = get_voice_style("sad", intensity=1.0)
        self.assertGreater(sad.length_scale, 1.0)  # slower
        self.assertLess(sad.noise_scale, 0.667)  # flatter

        romantic = get_voice_style("romantic", intensity=1.0)
        self.assertGreater(romantic.length_scale, 1.0)  # slower
        self.assertLess(romantic.noise_scale, 0.667)  # softer

        excited = get_voice_style("excited", intensity=1.0)
        self.assertLess(excited.length_scale, 1.0)  # faster
        self.assertGreater(excited.noise_w, 0.8)  # livelier

    def test_intensity_interpolation(self) -> None:
        """Verify intensity scales linearly between baseline and target emotion."""
        baseline = get_voice_style("excited", intensity=0.0)
        half = get_voice_style("excited", intensity=0.5)
        full = get_voice_style("excited", intensity=1.0)

        self.assertAlmostEqual(baseline.length_scale, 1.0, places=2)
        expected_half = round(1.0 + 0.5 * (full.length_scale - 1.0), 4)
        self.assertAlmostEqual(half.length_scale, expected_half, places=3)

    def test_apply_pitch_shift(self) -> None:
        """Verify 1D numpy linear interpolation pitch shift."""
        audio = np.linspace(-1000, 1000, 1000, dtype=np.int16)

        # 1.0 factor should return original
        res_same = apply_pitch_shift(audio, 1.0)
        self.assertEqual(len(res_same), len(audio))

        # > 1.0 factor (higher pitch) compresses time / decreases length
        res_high = apply_pitch_shift(audio, 1.1)
        self.assertLess(len(res_high), len(audio))

        # < 1.0 factor (lower pitch) expands time / increases length
        res_low = apply_pitch_shift(audio, 0.9)
        self.assertGreater(len(res_low), len(audio))


class TestSpeaker(unittest.TestCase):
    """Test suite verifying Piper synthesis, cache expiry, sentence splitting, and fallback."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="anya_test_cache_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_cache_path(self) -> None:
        """Verify cache path is deterministic for identical text and style."""
        style1 = VoiceStyle(length_scale=0.9, noise_scale=0.7)
        style2 = VoiceStyle(length_scale=0.9, noise_scale=0.7)
        style3 = VoiceStyle(length_scale=1.1, noise_scale=0.7)

        path1 = compute_cache_path("Hello Anya", style1, cache_dir=self.temp_dir)
        path2 = compute_cache_path("Hello Anya", style2, cache_dir=self.temp_dir)
        path3 = compute_cache_path("Hello Anya", style3, cache_dir=self.temp_dir)

        self.assertEqual(path1, path2)
        self.assertNotEqual(path1, path3)

    def test_cleanup_cache_older_than_10_minutes(self) -> None:
        """Verify cleanup_cache removes files older than threshold."""
        old_file = self.temp_dir / "old_clip.wav"
        recent_file = self.temp_dir / "recent_clip.wav"

        old_file.write_bytes(b"RIFF dummy old")
        recent_file.write_bytes(b"RIFF dummy recent")

        # Set old_file mtime to 15 minutes ago
        old_mtime = time.time() - 900
        time_tuple = (old_mtime, old_mtime)
        import os
        os.utime(str(old_file), time_tuple)

        deleted = cleanup_cache(max_age_seconds=600, cache_dir=self.temp_dir)
        self.assertEqual(deleted, 1)
        self.assertFalse(old_file.exists())
        self.assertTrue(recent_file.exists())

    def test_split_into_sentences(self) -> None:
        """Verify sentence splitting respects punctuation."""
        text = "Hello there! How are you doing today? I am doing great."
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "Hello there!")
        self.assertEqual(sentences[1], "How are you doing today?")
        self.assertEqual(sentences[2], "I am doing great.")

    def test_synthesize_and_cache(self) -> None:
        """Verify synthesize creates a valid WAV file and uses cache on subsequent calls."""
        style = get_voice_style("happy", 0.8)
        wav_path = synthesize("Hi, this is a test.", style)

        self.assertTrue(wav_path.exists())
        self.assertGreater(wav_path.stat().st_size, 1000)

        # Check WAV header validity
        with wave.open(str(wav_path), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), 22050)
            self.assertGreater(wf.getnframes(), 0)

        # Second call should return cached file
        wav_path2 = synthesize("Hi, this is a test.", style)
        self.assertEqual(wav_path, wav_path2)

    def test_macos_say_fallback(self) -> None:
        """Verify fallback to macOS say + afconvert works with argument lists."""
        dest = self.temp_dir / "fallback.wav"
        style = VoiceStyle(length_scale=1.0, pitch_shift=1.0)
        res_path = _fallback_synthesize_say("Testing macOS speech fallback.", dest, style)

        self.assertTrue(res_path.exists())
        self.assertGreater(res_path.stat().st_size, 500)
        with wave.open(str(res_path), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getframerate(), 22050)

    def test_speak_stream_synchronous(self) -> None:
        """Verify speak_stream synthesizes sentences and emits audio_clip_ready signals."""
        clips_received = []

        def _on_clip(url: str, utterance_id: str, is_last: bool) -> None:
            clips_received.append((url, utterance_id, is_last))

        events.audio_clip_ready.connect(_on_clip)
        try:
            sentences = ["First sentence.", "Second sentence."]
            uid = speak_stream(sentences, emotion="happy", sync=True)

            self.assertEqual(len(clips_received), 2)
            # First clip is_last should be False
            self.assertEqual(clips_received[0][1], uid)
            self.assertFalse(clips_received[0][2])
            # Second clip is_last should be True
            self.assertEqual(clips_received[1][1], uid)
            self.assertTrue(clips_received[1][2])
        finally:
            events.audio_clip_ready.disconnect(_on_clip)


class TestBridgeAudio(unittest.TestCase):
    """Test suite verifying WebBridge audio slots and event wiring."""

    def test_bridge_audio_methods(self) -> None:
        """Verify WebBridge exposes queue_audio, play_sound, stop_speech without error."""
        bridge = WebBridge()
        # Should not raise even when page is not wired
        bridge.queue_audio("/assets/cache/test.wav", "test_id", True)
        bridge.play_sound("laugh")
        bridge.stop_speech()

    def test_bridge_audio_ended_signal(self) -> None:
        """Verify bridge.audioEnded emits audio_ended_signal and events.audio_ended."""
        bridge = WebBridge()
        ended_events = []
        events.audio_ended.connect(lambda: ended_events.append(True))
        try:
            bridge.audioEnded("utt_123")
            self.assertEqual(len(ended_events), 1)
        finally:
            events.audio_ended.disconnect()


if __name__ == "__main__":
    unittest.main()
