"""Piper TTS synthesis pipeline with sentence queuing, caching, and macOS say fallback."""

import hashlib
import logging
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Iterable, List, Optional
import uuid
import wave

import numpy as np

from config import get_settings
from core.events import events
from core.voice_style import VoiceStyle, apply_pitch_shift, get_voice_style

logger = logging.getLogger("anya.core.speaker")

# Module-level voice model instance
_PIPER_VOICE_INSTANCE: Any = None
_VOICE_LOCK = threading.Lock()
_ACTIVE_SYNTH_LOCK = threading.Lock()
_STOP_SPEECH_FLAG = threading.Event()


def cleanup_cache(max_age_seconds: int = 600, cache_dir: Optional[Path | str] = None) -> int:
    """Delete cached audio files older than max_age_seconds (default 10 minutes).

    Args:
        max_age_seconds: Maximum age in seconds before a file is considered stale.
        cache_dir: Optional custom cache directory.

    Returns:
        Number of stale cache files successfully deleted.
    """
    settings = get_settings()
    target_dir = Path(cache_dir) if cache_dir else settings.ASSETS_DIR / "cache"
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        return 0

    now = time.time()
    deleted_count = 0

    for item in target_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".wav":
            try:
                mtime = item.stat().st_mtime
                if now - mtime > max_age_seconds:
                    item.unlink(missing_ok=True)
                    deleted_count += 1
            except Exception as exc:
                logger.warning("Could not delete stale cache file %s: %s", item, exc)

    if deleted_count > 0:
        logger.info("Cleaned up %d stale TTS cache file(s) from %s", deleted_count, target_dir)
    return deleted_count


def ensure_voice_downloaded(model_path: Optional[Path | str] = None) -> Path:
    """Verify Piper voice model and config exist, downloading them on first run if missing.

    Args:
        model_path: Path to .onnx model file.

    Returns:
        Resolved Path to the voice model file.
    """
    settings = get_settings()
    target_path = Path(model_path) if model_path else settings.BASE_DIR / settings.PIPER_VOICE
    config_path = target_path.with_suffix(".onnx.json")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and config_path.exists():
        return target_path

    voice_name = target_path.stem
    logger.info("Piper voice model not found at %s. Downloading voice '%s'...", target_path, voice_name)

    try:
        from piper.download_voices import download_voice

        download_voice(voice_name, target_path.parent)
        logger.info("Successfully downloaded Piper voice '%s' into %s", voice_name, target_path.parent)
    except Exception as exc:
        logger.error("Failed to download Piper voice '%s': %s", voice_name, exc)
        raise RuntimeError(f"Could not download Piper voice model: {exc}") from exc

    return target_path


def get_piper_voice(model_path: Optional[Path | str] = None) -> Any:
    """Return loaded PiperVoice instance, initializing on first access."""
    global _PIPER_VOICE_INSTANCE
    with _VOICE_LOCK:
        if _PIPER_VOICE_INSTANCE is not None:
            return _PIPER_VOICE_INSTANCE

        resolved_model = ensure_voice_downloaded(model_path)
        try:
            from piper import PiperVoice

            logger.info("Loading Piper TTS model: %s", resolved_model)
            _PIPER_VOICE_INSTANCE = PiperVoice.load(str(resolved_model))
            return _PIPER_VOICE_INSTANCE
        except Exception as exc:
            logger.error("Failed to initialize PiperVoice from %s: %s", resolved_model, exc)
            raise


def unload_piper_voice() -> None:
    """Unload Piper model from memory when idle to reduce RAM footprint."""
    global _PIPER_VOICE_INSTANCE
    with _VOICE_LOCK:
        if _PIPER_VOICE_INSTANCE is not None:
            logger.info("Unloading Piper voice model to conserve memory.")
            _PIPER_VOICE_INSTANCE = None


def compute_cache_path(text: str, style: VoiceStyle, cache_dir: Optional[Path | str] = None) -> Path:
    """Return path for cached speech WAV based on hash of text and style settings."""
    settings = get_settings()
    target_dir = Path(cache_dir) if cache_dir else settings.ASSETS_DIR / "cache"
    target_dir.mkdir(parents=True, exist_ok=True)

    key_repr = (
        f"{text.strip()}|"
        f"{style.length_scale:.4f}|"
        f"{style.noise_scale:.4f}|"
        f"{style.noise_w:.4f}|"
        f"{style.pitch_shift:.4f}"
    )
    cache_hash = hashlib.sha256(key_repr.encode("utf-8")).hexdigest()[:24]
    return target_dir / f"{cache_hash}.wav"


def _pitch_shift_wav_inplace(wav_path: Path, pitch_shift: float) -> None:
    """Apply pitch shift to an existing WAV file using 1D numpy linear interpolation."""
    if abs(pitch_shift - 1.0) < 0.005:
        return

    with wave.open(str(wav_path), "rb") as wf:
        params = wf.getparams()
        n_frames = wf.getnframes()
        raw_frames = wf.readframes(n_frames)

    audio_data = np.frombuffer(raw_frames, dtype=np.int16)
    shifted_data = apply_pitch_shift(audio_data, pitch_shift)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setparams(params)
        wf.writeframes(shifted_data.tobytes())


def _fallback_synthesize_say(text: str, output_path: Path, style: VoiceStyle) -> Path:
    """Synthesize speech using macOS say + afconvert as an argument list (no shell=True).

    Args:
        text: Text to speak.
        output_path: Destination WAV file path.
        style: Voice style settings.

    Returns:
        Path to generated WAV file.
    """
    logger.info("Using macOS 'say' + 'afconvert' fallback for: %r", text[:40])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary AIFF output from say command
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tf:
        temp_aiff = tf.name

    try:
        # Approximate words per minute (macOS say default ~185 wpm)
        rate = max(100, min(300, int(185 / style.length_scale)))
        say_cmd = ["say", "-r", str(rate), "-o", temp_aiff, text]
        logger.debug("Executing say fallback: %s", say_cmd)
        subprocess.run(say_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Convert AIFF to standard 16-bit 22050Hz LE WAV
        convert_cmd = [
            "afconvert",
            "-f",
            "WAVE",
            "-d",
            "LEI16@22050",
            temp_aiff,
            str(output_path),
        ]
        logger.debug("Executing afconvert fallback: %s", convert_cmd)
        subprocess.run(convert_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if abs(style.pitch_shift - 1.0) >= 0.005 and output_path.exists():
            _pitch_shift_wav_inplace(output_path, style.pitch_shift)

        return output_path
    finally:
        if os.path.exists(temp_aiff):
            try:
                os.remove(temp_aiff)
            except OSError:
                pass


def synthesize(text: str, style: Optional[VoiceStyle] = None) -> Path:
    """Synthesize text to WAV file path using Piper TTS, with caching and macOS say fallback.

    Args:
        text: Plaintext sentence to synthesize.
        style: VoiceStyle parameters. If None, default neutral style is used.

    Returns:
        Path to the synthesized .wav audio file.
    """
    active_style = style or VoiceStyle()
    clean_text = text.strip()
    if not clean_text:
        clean_text = "..."

    output_path = compute_cache_path(clean_text, active_style)

    # 1. Cache hit check
    if output_path.exists() and output_path.stat().st_size > 44:  # standard wav header is 44 bytes
        try:
            os.utime(str(output_path), None)  # refresh access timestamp
            logger.debug("TTS Cache hit for: %r", clean_text[:30])
            return output_path
        except OSError:
            pass

    # 2. Piper synthesis attempt
    with _ACTIVE_SYNTH_LOCK:
        try:
            voice = get_piper_voice()
            from piper.config import SynthesisConfig

            syn_cfg = SynthesisConfig(
                length_scale=active_style.length_scale,
                noise_scale=active_style.noise_scale,
                noise_w_scale=active_style.noise_w,
            )

            temp_dest = output_path.with_suffix(".tmp.wav")
            with wave.open(str(temp_dest), "wb") as wf:
                voice.synthesize_wav(clean_text, wf, syn_config=syn_cfg)

            if abs(active_style.pitch_shift - 1.0) >= 0.005:
                _pitch_shift_wav_inplace(temp_dest, active_style.pitch_shift)

            temp_dest.replace(output_path)
            logger.debug("Synthesized via Piper: %s (size: %d bytes)", output_path.name, output_path.stat().st_size)
            return output_path

        except Exception as exc:
            logger.warning("Piper synthesis failed (%s). Routing to macOS say fallback.", exc)
            return _fallback_synthesize_say(clean_text, output_path, active_style)


def split_into_sentences(text: str) -> List[str]:
    """Split string into individual sentences based on punctuation."""
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    clean = [s.strip() for s in raw_sentences if s.strip()]
    return clean if clean else [text.strip()]


class SpeechQueueWorker:
    """Asynchronous worker that processes speak_stream requests sequentially."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = True
        self._start_worker()

    def _start_worker(self) -> None:
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="AnyaSpeechWorker")
        self._thread.start()

    def _worker_loop(self) -> None:
        while self._running:
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is None:
                break

            sentences, emotion, intensity, utterance_id = task
            self._process_stream(sentences, emotion, intensity, utterance_id)
            self._queue.task_done()

    def _process_stream(
        self,
        sentences: Iterable[str],
        emotion: str,
        intensity: float,
        utterance_id: str,
    ) -> None:
        """Stream sentences: synthesize sentence N while sentence N-1 is played in frontend."""
        style = get_voice_style(emotion, intensity)
        sentence_list = list(sentences) if not isinstance(sentences, list) else sentences
        total = len(sentence_list)
        if total == 0:
            return

        logger.info("speak_stream started: %d sentence(s), emotion=%s, utterance_id=%s", total, emotion, utterance_id)
        _STOP_SPEECH_FLAG.clear()

        for idx, sentence in enumerate(sentence_list):
            if _STOP_SPEECH_FLAG.is_set():
                logger.info("speak_stream cancelled for utterance_id: %s", utterance_id)
                break

            clean_sentence = sentence.strip()
            if not clean_sentence:
                continue

            is_last = (idx == total - 1)
            try:
                wav_path = synthesize(clean_sentence, style)
                # Front-end relative asset URL
                clip_url = f"/assets/cache/{wav_path.name}"
                logger.debug("Emitting audio_clip_ready (%d/%d): %s (is_last=%s)", idx + 1, total, clip_url, is_last)
                events.audio_clip_ready.emit(clip_url, utterance_id, is_last)
            except Exception as exc:
                logger.error("Error synthesizing sentence '%s': %s", clean_sentence[:30], exc)

    def enqueue(
        self,
        sentences: Iterable[str],
        emotion: str = "neutral",
        intensity: float = 1.0,
        utterance_id: Optional[str] = None,
    ) -> str:
        """Enqueue speech request and return unique utterance ID."""
        uid = utterance_id or uuid.uuid4().hex[:8]
        self._queue.put((sentences, emotion, intensity, uid))
        return uid

    def stop(self) -> None:
        """Halt ongoing speech generation and clear the queue."""
        _STOP_SPEECH_FLAG.set()
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except (queue.Empty, ValueError):
                break
        events.stop_speech_requested.emit()
        logger.info("Speech worker halted and queue drained.")


_WORKER_INSTANCE: Optional[SpeechQueueWorker] = None


def get_speech_worker() -> SpeechQueueWorker:
    """Return singleton SpeechQueueWorker instance."""
    global _WORKER_INSTANCE
    if _WORKER_INSTANCE is None:
        _WORKER_INSTANCE = SpeechQueueWorker()
    return _WORKER_INSTANCE


def speak_stream(
    sentences: Iterable[str],
    emotion: str = "neutral",
    intensity: float = 1.0,
    sync: bool = False,
) -> str:
    """Stream speech synthesis for sentences, delivering audio clips sequentially to frontend.

    Args:
        sentences: Iterable of text sentences to speak.
        emotion: Emotion name from emotion.yaml (e.g. happy, sad, angry).
        intensity: Expression intensity factor (0.0 to 1.0).
        sync: If True, execute synchronously; if False, enqueue in background speech worker.

    Returns:
        Unique utterance ID for the speech sequence.
    """
    worker = get_speech_worker()
    utterance_id = uuid.uuid4().hex[:8]

    if sync:
        worker._process_stream(sentences, emotion, intensity, utterance_id)
        return utterance_id

    return worker.enqueue(sentences, emotion, intensity, utterance_id)


def speak(text: str, emotion: str = "neutral", intensity: float = 1.0) -> str:
    """Convenience helper to speak a paragraph by splitting into sentences and streaming."""
    sentences = split_into_sentences(text)
    return speak_stream(sentences, emotion, intensity)


def stop_speech() -> None:
    """Stop active speech synthesis and tell frontend to clear its audio queue."""
    worker = get_speech_worker()
    worker.stop()


# ── Wire Qt Signals on module import ─────────────────────────────────────────

def _on_speak_requested(text: str) -> None:
    """Handle events.speak_requested signal."""
    speak(text)


def _on_speak_stream_requested(sentences: Any, emotion: str, intensity: float) -> None:
    """Handle events.speak_stream_requested signal."""
    if isinstance(sentences, (list, tuple)):
        speak_stream(sentences, emotion, intensity)
    elif isinstance(sentences, str):
        speak(sentences, emotion, intensity)
    else:
        speak_stream(list(sentences), emotion, intensity)


def _on_stop_requested() -> None:
    """Handle events.stop_speech_requested signal."""
    _STOP_SPEECH_FLAG.set()


events.speak_requested.connect(_on_speak_requested)
events.speak_stream_requested.connect(_on_speak_stream_requested)
events.stop_speech_requested.connect(_on_stop_requested)

# Delete stale cache files older than 10 minutes on startup
cleanup_cache(max_age_seconds=600)
