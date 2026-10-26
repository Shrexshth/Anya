"""Microphone capture, WebRTC VAD, wake word detection, and Whisper transcription."""

import collections
import logging
import math
import os
import queue
import re
import sys
import threading
import time
from typing import Any, Callable, Deque, List, Optional, Sequence, Tuple

import numpy as np
from rapidfuzz import fuzz

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    sd = None  # type: ignore[assignment]
    SOUNDDEVICE_AVAILABLE = False

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    webrtcvad = None  # type: ignore[assignment]
    WEBRTCVAD_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None  # type: ignore[assignment]
    FASTER_WHISPER_AVAILABLE = False

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except (ImportError, Exception):
    keyboard = None  # type: ignore[assignment]
    PYNPUT_AVAILABLE = False

from config import Settings, get_settings, setup_logging
from core.events import events, get_events

logger = logging.getLogger("anya.core.listener")

# Known Whisper silence / low-confidence hallucination phrases
HALLUCINATION_EXACT = {
    "thank you",
    "thank you.",
    "thank you very much",
    "thank you so much",
    "thanks for watching",
    "thanks for watching.",
    "thanks for watching!",
    "thanks for listening",
    "thank you for watching",
    "you",
    "you.",
    "bye",
    "bye.",
    "goodbye",
    "subscribe",
    "subtitles by",
    "subtitles by the amara.org community",
    "like and subscribe",
    "please subscribe",
}

HALLUCINATION_PATTERNS = [
    re.compile(r"^\[.*\]$"),                # [music], [applause], etc.
    re.compile(r"^\(.*\)$"),                # (laughter), (bell), etc.
    re.compile(r"^thanks?\s+for\s+watching.*$", re.IGNORECASE),
    re.compile(r"^thank\s+you.*$", re.IGNORECASE),
    re.compile(r"^subtitles\s+by.*$", re.IGNORECASE),
]

STOPWORDS = {"any", "and", "an", "in", "on", "at", "to", "for", "with", "is", "it"}
BARGE_IN_PHRASES = ["stop", "anya stop", "anja stop", "please stop", "shut up", "stop talking"]


def filter_hallucination(text: Optional[str]) -> Optional[str]:
    """Filter out silence hallucinations produced by Whisper, returning None if matched."""
    if not text:
        return None
    trimmed = text.strip()
    if not trimmed:
        return None
    # Strip enclosing punctuation
    clean = re.sub(r"[^\w\s]", "", trimmed).lower().strip()
    if not clean:
        return None
    if clean in HALLUCINATION_EXACT or trimmed.lower() in HALLUCINATION_EXACT:
        return None
    for pat in HALLUCINATION_PATTERNS:
        if pat.match(trimmed) or pat.match(clean):
            return None
    return trimmed


def match_and_strip_wake_phrase(
    text: str,
    wake_phrases: Sequence[str],
    threshold: float = 75.0,
) -> Tuple[bool, Optional[str], str]:
    """Check if text contains any wake phrase using rapidfuzz, returning (matched, matched_phrase, stripped_text)."""
    if not text:
        return False, None, text

    clean = re.sub(r"[^\w\s]", " ", text.lower())
    words = clean.split()
    if not words:
        return False, None, text

    # Sort wake phrases by token length descending to match longest first (e.g. "hey anya" before "anya")
    sorted_phrases = sorted(wake_phrases, key=lambda p: len(p.split()), reverse=True)

    for wp in sorted_phrases:
        wp_clean = re.sub(r"[^\w\s]", " ", wp.lower()).strip()
        wp_words = wp_clean.split()
        n = len(wp_words)
        if n == 0:
            continue
        for i in range(len(words) - n + 1):
            cand_words = words[i : i + n]
            # Avoid single-word short stopwords (e.g. 'any') falsely matching 'anya'
            if n == 1 and cand_words[0] in STOPWORDS:
                continue
            window = " ".join(cand_words)
            score = fuzz.ratio(wp_clean, window)
            if score >= threshold:
                # Strip the wake phrase words from original
                stripped_words = words[:i] + words[i + n :]
                stripped_text = " ".join(stripped_words).strip()
                return True, wp, stripped_text

    return False, None, text


def is_barge_in_stop(text: str, threshold: float = 75.0) -> bool:
    """Check if transcribed speech contains a stop command ("stop", "anya stop", etc.)."""
    if not text:
        return False
    clean = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    words = clean.split()
    if not words:
        return False
    for phrase in BARGE_IN_PHRASES:
        p_clean = phrase.lower().strip()
        p_words = p_clean.split()
        n = len(p_words)
        for i in range(len(words) - n + 1):
            window = " ".join(words[i : i + n])
            if fuzz.ratio(p_clean, window) >= threshold:
                return True
    return False


def matches_pynput_key(event_key: Any, target_key_name: str) -> bool:
    """Return True if a pynput key event matches the configured target key name."""
    if not PYNPUT_AVAILABLE or keyboard is None:
        return False
    target = target_key_name.strip().lower()
    if target == "space":
        return event_key == keyboard.Key.space
    elif target in ("ctrl", "control"):
        return event_key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)
    elif target in ("alt", "option"):
        return event_key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r)
    elif target in ("cmd", "command"):
        return event_key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)
    elif target == "shift":
        return event_key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r)
    elif hasattr(keyboard.Key, target):
        return event_key == getattr(keyboard.Key, target)
    elif hasattr(event_key, "char") and event_key.char:
        return event_key.char.lower() == target
    return False


class VADSegmenter:
    """Segments a continuous stream of PCM audio frames using WebRTC VAD."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        vad_mode: int = 2,
        silence_timeout_s: float = 0.8,
        max_duration_s: float = 15.0,
        pre_speech_ms: int = 300,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * (frame_ms / 1000.0))
        self.frame_bytes = self.frame_samples * 2  # 16-bit mono PCM

        self.silence_timeout_frames = max(1, math.ceil(silence_timeout_s / (frame_ms / 1000.0)))
        self.max_frames = max(1, math.ceil(max_duration_s / (frame_ms / 1000.0)))
        pre_speech_frames = max(1, int(pre_speech_ms / frame_ms))

        self.vad = webrtcvad.Vad(vad_mode) if WEBRTCVAD_AVAILABLE else None
        self._pre_speech_buffer: Deque[bytes] = collections.deque(maxlen=pre_speech_frames)
        self._current_segment: List[bytes] = []
        self._is_recording: bool = False
        self._silent_frames: int = 0

    @property
    def is_recording(self) -> bool:
        """Return True if currently accumulating voiced speech frames."""
        return self._is_recording

    def reset(self) -> None:
        """Reset internal segmenter buffers and counters."""
        self._pre_speech_buffer.clear()
        self._current_segment.clear()
        self._is_recording = False
        self._silent_frames = 0

    def process_frame(self, frame: bytes) -> Optional[bytes]:
        """Feed a 30 ms PCM frame, returning complete utterance bytes when finished, or None."""
        if len(frame) != self.frame_bytes:
            return None

        is_speech = False
        if self.vad is not None:
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
            except Exception as exc:
                logger.debug("webrtcvad error on frame: %s", exc)
                is_speech = False
        else:
            # Fallback simple energy-based detection if webrtcvad is unavailable
            samples = np.frombuffer(frame, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            is_speech = rms > 500.0

        if not self._is_recording:
            if is_speech:
                self._is_recording = True
                self._current_segment = list(self._pre_speech_buffer)
                self._current_segment.append(frame)
                self._silent_frames = 0
                return None
            else:
                self._pre_speech_buffer.append(frame)
                return None
        else:
            self._current_segment.append(frame)
            if is_speech:
                self._silent_frames = 0
            else:
                self._silent_frames += 1

            silence_reached = self._silent_frames >= self.silence_timeout_frames
            max_cap_reached = len(self._current_segment) >= self.max_frames

            if silence_reached or max_cap_reached:
                completed = b"".join(self._current_segment)
                self.reset()
                return completed

            return None


class Listener:
    """Manages audio capture, WebRTC VAD, wake word detection, push-to-talk, and Whisper."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        push_to_talk: Optional[bool] = None,
        on_user_text: Optional[Callable[[str], None]] = None,
        on_state_change: Optional[Callable[[str], None]] = None,
        on_window_event: Optional[Callable[[str], None]] = None,
        on_wake_event: Optional[Callable[[str, str], None]] = None,
        on_segment_event: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.push_to_talk = (
            push_to_talk
            if push_to_talk is not None
            else (
                getattr(self.settings, "PUSH_TO_TALK", False)
                or os.getenv("PUSH_TO_TALK", "").lower() in ("1", "true", "yes", "on")
                or os.getenv("LISTENER_MODE", "").lower() in ("ptt", "push_to_talk")
            )
        )
        self.on_user_text = on_user_text
        self.on_state_change = on_state_change
        self.on_window_event = on_window_event
        self.on_wake_event = on_wake_event
        self.on_segment_event = on_segment_event

        self.sample_rate = 16000
        self.frame_ms = 30
        self.frame_samples = int(self.sample_rate * (self.frame_ms / 1000.0))  # 480
        self.frame_bytes = self.frame_samples * 2  # 960 bytes

        self.silence_timeout_s = 0.8
        self.max_duration_s = 15.0
        self.conversation_window_s = self.settings.CONVERSATION_WINDOW_S
        self.wake_phrases = self.settings.WAKE_PHRASES
        self.barge_in = self.settings.BARGE_IN

        self.segmenter = VADSegmenter(
            sample_rate=self.sample_rate,
            frame_ms=self.frame_ms,
            vad_mode=2,
            silence_timeout_s=self.silence_timeout_s,
            max_duration_s=self.max_duration_s,
        )

        self._lock = threading.RLock()
        self._running = False
        self._is_paused = False
        self._is_speaking = False
        self._barge_in_active = False

        self._is_awake = False
        self._conversation_window_expiry: float = 0.0

        # Push-to-talk state
        self._ptt_pressed = False
        self._ptt_buffer: List[bytes] = []
        self._keyboard_listener: Optional[Any] = None

        # Audio streaming
        self._audio_queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._stream: Optional[Any] = None
        self._worker_thread: Optional[threading.Thread] = None

        # Cached Whisper models
        self._wake_model: Optional[Any] = None
        self._whisper_model: Optional[Any] = None

        # Connect central Qt signals to automatically coordinate state
        try:
            events.audio_ended.connect(lambda: self.set_speaking(False))
            events.stop_speech_requested.connect(lambda: self.set_speaking(False))
            events.speak_requested.connect(lambda _txt: self.set_speaking(True))
            events.speak_stream_requested.connect(lambda _s, _e, _i: self.set_speaking(True))
        except Exception as exc:
            logger.debug("Failed to bind events in listener: %s", exc)

    def is_window_open(self) -> bool:
        """Return True if the conversation window is currently open and unexpired."""
        with self._lock:
            return self._is_awake and (time.time() < self._conversation_window_expiry)

    def open_conversation_window(self) -> None:
        """Open or extend the active conversation window."""
        with self._lock:
            now = time.time()
            was_open = self.is_window_open()
            self._is_awake = True
            self._conversation_window_expiry = now + self.conversation_window_s

            if not was_open:
                msg = f"Conversation window OPEN ({self.conversation_window_s} s)"
                logger.info(msg)
                if self.on_window_event:
                    self.on_window_event(msg)
                events.state_changed.emit("LISTEN")
                if self.on_state_change:
                    self.on_state_change("LISTEN")
            else:
                msg = f"Conversation window extended (+{self.conversation_window_s} s)"
                logger.info(msg)
                if self.on_window_event:
                    self.on_window_event(msg)

    def close_conversation_window(self) -> None:
        """Close the conversation window and transition back to asleep."""
        with self._lock:
            if self._is_awake:
                self._is_awake = False
                self._conversation_window_expiry = 0.0
                msg = "Conversation window CLOSED (timeout). Returning to sleep."
                logger.info(msg)
                if self.on_window_event:
                    self.on_window_event(msg)
                events.state_changed.emit("IDLE")
                if self.on_state_change:
                    self.on_state_change("IDLE")
                self.unload_whisper_model()

    def pause(self) -> None:
        """Pause/close the microphone so Anya never hears herself."""
        with self._lock:
            if self._is_paused:
                return
            self._is_paused = True
            logger.debug("Listener paused: microphone closed.")
            if self._stream is not None:
                try:
                    self._stream.stop()
                except Exception as exc:
                    logger.debug("Error stopping stream on pause: %s", exc)

    def resume(self) -> None:
        """Resume/open the microphone capture."""
        with self._lock:
            if not self._is_paused:
                return
            self._is_paused = False
            logger.debug("Listener resumed: microphone opened.")
            if self._stream is not None:
                try:
                    self._stream.start()
                except Exception as exc:
                    logger.debug("Error starting stream on resume: %s", exc)

    def set_speaking(self, speaking: bool) -> None:
        """Update speaking state. If barge-in is disabled, mic is closed while speaking."""
        with self._lock:
            self._is_speaking = speaking
            if speaking:
                if self.barge_in:
                    self._barge_in_active = True
                    logger.debug("Speaking started: BARGE_IN active (listening only for stop).")
                else:
                    self._barge_in_active = False
                    self.pause()
            else:
                self._barge_in_active = False
                if self._is_paused:
                    self.resume()

    def get_wake_model(self) -> Any:
        """Load or return the cached tiny.en wake model."""
        with self._lock:
            if self._wake_model is None and FASTER_WHISPER_AVAILABLE and WhisperModel is not None:
                logger.info("Loading WAKE_MODEL (%s)...", self.settings.WAKE_MODEL)
                self._wake_model = WhisperModel(
                    self.settings.WAKE_MODEL,
                    device="cpu",
                    compute_type="int8",
                )
            return self._wake_model

    def get_whisper_model(self) -> Any:
        """Load or return the cached base.en conversational Whisper model."""
        with self._lock:
            if self._whisper_model is None and FASTER_WHISPER_AVAILABLE and WhisperModel is not None:
                logger.info("Loading WHISPER_MODEL (%s)...", self.settings.WHISPER_MODEL)
                self._whisper_model = WhisperModel(
                    self.settings.WHISPER_MODEL,
                    device="cpu",
                    compute_type="int8",
                )
            return self._whisper_model

    def unload_whisper_model(self) -> None:
        """Unload conversational WHISPER_MODEL when idle to free memory."""
        with self._lock:
            if self._whisper_model is not None:
                logger.info("Unloading WHISPER_MODEL to free memory.")
                del self._whisper_model
                self._whisper_model = None

    def transcribe_audio(self, model: Any, audio_pcm: bytes, vad_filter: bool = False) -> str:
        """Transcribe 16 kHz int16 PCM audio bytes with faster-whisper."""
        if not audio_pcm or model is None:
            return ""
        try:
            audio_np = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = model.transcribe(audio_np, language="en", vad_filter=vad_filter)
            text_chunks = [s.text.strip() for s in segments if s.text.strip()]
            return " ".join(text_chunks).strip()
        except Exception as exc:
            logger.exception("Error during Whisper transcription: %s", exc)
            return ""

    def _dispatch_user_text(self, text: str) -> None:
        """Emit user text signal and call external callback."""
        events.user_text.emit(text)
        if self.on_user_text:
            self.on_user_text(text)

    def handle_audio_segment(self, audio_pcm: bytes) -> Optional[str]:
        """Process a completed audio utterance through wake word or conversational transcription."""
        if not audio_pcm:
            return None

        # 1. Barge-in handling during speech
        if self._barge_in_active:
            wake_model = self.get_wake_model()
            text = self.transcribe_audio(wake_model, audio_pcm, vad_filter=False)
            if is_barge_in_stop(text):
                logger.info("Barge-in matched: '%s' -> emitting stop_speech_requested.", text)
                events.stop_speech_requested.emit()
                self._barge_in_active = False
            # Never treat own-voice speech as conversational input
            return None

        # 2. Asleep state: check for wake phrase with WAKE_MODEL
        if not self.is_window_open():
            wake_model = self.get_wake_model()
            raw_text = self.transcribe_audio(wake_model, audio_pcm, vad_filter=False)
            matched, phrase, stripped = match_and_strip_wake_phrase(raw_text, self.wake_phrases)

            if not matched:
                logger.debug("Asleep: audio segment did not match wake phrase ('%s')", raw_text)
                return None

            logger.info("Wake phrase '%s' matched in: '%s'", phrase, raw_text)
            if self.on_wake_event:
                self.on_wake_event(phrase or "", raw_text)
            self.open_conversation_window()

            # If user said wake phrase + query together (e.g. "hey anya what's the weather?")
            stripped_clean = stripped.strip()
            if stripped_clean:
                whisper_model = self.get_whisper_model()
                full_text = self.transcribe_audio(whisper_model, audio_pcm, vad_filter=True)
                _, _, final_stripped = match_and_strip_wake_phrase(full_text, self.wake_phrases)
                candidate = final_stripped.strip() if final_stripped.strip() else stripped_clean
                cleaned = filter_hallucination(candidate)
                if cleaned:
                    logger.info("Transcribed wake command: '%s'", cleaned)
                    self._dispatch_user_text(cleaned)
                    return cleaned
            return None

        # 3. Awake state: conversation window active, use WHISPER_MODEL
        else:
            whisper_model = self.get_whisper_model()
            raw_text = self.transcribe_audio(whisper_model, audio_pcm, vad_filter=True)
            cleaned = filter_hallucination(raw_text)
            if not cleaned:
                logger.debug("Whisper output filtered as hallucination/silence: '%s'", raw_text)
                return None

            # Strip wake phrase if user repeated it
            _, _, stripped = match_and_strip_wake_phrase(cleaned, self.wake_phrases)
            final_text = stripped.strip() if stripped.strip() else cleaned

            self.open_conversation_window()
            logger.info("Transcribed conversational utterance: '%s'", final_text)
            self._dispatch_user_text(final_text)
            return final_text

    def _audio_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """Sounddevice raw audio callback pushing chunks to processing queue."""
        if status:
            logger.debug("Sounddevice status: %s", status)
        if self._running and not self._is_paused:
            self._audio_queue.put(bytes(indata))

    def _on_key_press(self, key: Any) -> None:
        """Push-to-talk key press handler."""
        if matches_pynput_key(key, self.settings.PUSH_TO_TALK_KEY):
            with self._lock:
                if not self._ptt_pressed:
                    self._ptt_pressed = True
                    self._ptt_buffer = []
                    logger.info("Push-to-talk ACTIVE (key held)...")
                    events.state_changed.emit("LISTEN")
                    if self.on_state_change:
                        self.on_state_change("LISTEN")

    def _on_key_release(self, key: Any) -> None:
        """Push-to-talk key release handler."""
        if matches_pynput_key(key, self.settings.PUSH_TO_TALK_KEY):
            audio_to_process: Optional[bytes] = None
            with self._lock:
                if self._ptt_pressed:
                    self._ptt_pressed = False
                    logger.info("Push-to-talk RELEASED: transcribing...")
                    if self._ptt_buffer:
                        audio_to_process = b"".join(self._ptt_buffer)
                        self._ptt_buffer.clear()

            if audio_to_process:
                whisper_model = self.get_whisper_model()
                raw_text = self.transcribe_audio(whisper_model, audio_to_process, vad_filter=True)
                cleaned = filter_hallucination(raw_text)
                if cleaned:
                    _, _, stripped = match_and_strip_wake_phrase(cleaned, self.wake_phrases)
                    final_text = stripped.strip() if stripped.strip() else cleaned
                    self.open_conversation_window()
                    logger.info("PTT Transcribed: '%s'", final_text)
                    self._dispatch_user_text(final_text)

    def _worker_loop(self) -> None:
        """Background thread consuming audio frames and driving VAD / conversation timing."""
        logger.debug("Audio processing worker thread started.")
        remainder = bytearray()

        while self._running:
            # Check conversation window expiry
            if self._is_awake and time.time() >= self._conversation_window_expiry:
                self.close_conversation_window()

            try:
                chunk = self._audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if chunk is None:
                break

            # If Push-to-Talk mode is active, buffer while key is pressed
            if self.push_to_talk:
                with self._lock:
                    if self._ptt_pressed:
                        self._ptt_buffer.append(chunk)
                continue

            # Standard VAD mode: slice chunk into exact 30 ms (frame_bytes) frames
            remainder.extend(chunk)
            while len(remainder) >= self.frame_bytes:
                frame = bytes(remainder[: self.frame_bytes])
                del remainder[: self.frame_bytes]

                completed_utterance = self.segmenter.process_frame(frame)
                if completed_utterance:
                    logger.debug("Completed speech segment detected (%d bytes)", len(completed_utterance))
                    if self.on_segment_event:
                        self.on_segment_event(f"Utterance captured ({len(completed_utterance)} bytes), transcribing...")
                    try:
                        self.handle_audio_segment(completed_utterance)
                    except Exception as exc:
                        logger.exception("Error processing speech segment: %s", exc)

        logger.debug("Audio processing worker thread terminated.")

    def start(self) -> None:
        """Start microphone capture stream, processing thread, and optional PTT listener."""
        with self._lock:
            if self._running:
                return
            self._running = True

        # Pre-warm wake model
        try:
            self.get_wake_model()
        except Exception as exc:
            logger.warning("Could not pre-warm WAKE_MODEL: %s", exc)

        # Start worker thread
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="AnyaListenerWorker", daemon=True
        )
        self._worker_thread.start()

        # Start push-to-talk keyboard listener if enabled
        if self.push_to_talk and PYNPUT_AVAILABLE and keyboard is not None:
            try:
                self._keyboard_listener = keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release,
                )
                self._keyboard_listener.daemon = True
                self._keyboard_listener.start()
                logger.info(
                    "Push-to-talk listener active on key: '%s'", self.settings.PUSH_TO_TALK_KEY
                )
            except Exception as exc:
                logger.warning("Could not start pynput keyboard listener: %s", exc)

        # Start sounddevice input stream
        if SOUNDDEVICE_AVAILABLE and sd is not None:
            try:
                self._stream = sd.RawInputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.frame_samples,
                    dtype="int16",
                    channels=1,
                    callback=self._audio_callback,
                )
                self._stream.start()
                logger.info(
                    "Microphone recording started (16 kHz mono, mode: %s)",
                    "Push-to-Talk" if self.push_to_talk else "VAD/Wake-Word",
                )
            except Exception as exc:
                logger.error("Failed to start sounddevice audio stream: %s", exc)
        else:
            logger.warning("sounddevice is unavailable; running listener without physical audio stream.")

    def stop(self) -> None:
        """Stop microphone stream, PTT listener, and worker thread."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                logger.debug("Error closing sounddevice stream: %s", exc)
            self._stream = None

        if self._keyboard_listener is not None:
            try:
                self._keyboard_listener.stop()
            except Exception as exc:
                logger.debug("Error stopping pynput listener: %s", exc)
            self._keyboard_listener = None

        self._audio_queue.put(None)
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

        self.unload_whisper_model()
        logger.info("Listener stopped.")

    def is_alive(self) -> bool:
        """Return True if listener is actively running."""
        return self._running


def main() -> None:
    """Run interactive CLI demo of the listener."""
    setup_logging()
    print("=" * 60)
    print("  ANYA LISTENER DEMO")
    print("=" * 60)
    settings = get_settings()
    print(f"Sample Rate: 16000 Hz, VAD Silence: 0.8s, Max Segment: 15s")
    print(f"Wake Model: {settings.WAKE_MODEL}, Whisper Model: {settings.WHISPER_MODEL}")
    print(f"Wake Phrases: {settings.WAKE_PHRASES}")
    print(f"Conversation Window: {settings.CONVERSATION_WINDOW_S}s")
    print(f"Barge-In: {settings.BARGE_IN}, Push-to-Talk Key: {settings.PUSH_TO_TALK_KEY}")
    print("-" * 60)

    def on_text(text: str) -> None:
        print(f"\n[LISTENER HEARD] >>> {text}\n")

    def on_window(evt: str) -> None:
        print(f"[WINDOW EVENT] >>> {evt}")

    def on_wake(phrase: str, raw: str) -> None:
        print(f"[WAKE DETECTED] >>> Matched: '{phrase}' (Utterance: '{raw}')")

    def on_segment(msg: str) -> None:
        print(f"[VAD EVENT] >>> {msg}")

    listener = Listener(
        settings=settings,
        on_user_text=on_text,
        on_window_event=on_window,
        on_wake_event=on_wake,
        on_segment_event=on_segment,
    )

    try:
        listener.start()
        print("[DEMO READY] Listening for wake phrase (e.g. 'Hey Anya'). Press Ctrl+C to stop.\n")
        while listener.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping demo...")
    finally:
        listener.stop()
        print("Demo ended.")


if __name__ == "__main__":
    main()
