"""Translates emotional state to Piper TTS speech speed and expressiveness parameters."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from config import get_settings

logger = logging.getLogger("anya.core.voice_style")

_STYLES_CACHE: Optional[Dict[str, Dict[str, float]]] = None
_CACHE_MTIME: float = 0.0


@dataclass(frozen=True)
class VoiceStyle:
    """Piper TTS synthesis parameters and pitch shift multiplier."""

    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    pitch_shift: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        """Return dict representation of style parameters."""
        return {
            "length_scale": self.length_scale,
            "noise_scale": self.noise_scale,
            "noise_w": self.noise_w,
            "pitch_shift": self.pitch_shift,
        }


def _resolve_yaml_path(custom_path: Optional[Path | str] = None) -> Path:
    """Return path to data/emotion.yaml from config or custom argument."""
    if custom_path is not None:
        return Path(custom_path)
    settings = get_settings()
    return settings.DATA_DIR / "emotion.yaml"


def load_voice_styles(yaml_path: Optional[Path | str] = None) -> Dict[str, Dict[str, float]]:
    """Load voice style definitions from data/emotion.yaml with mtime caching."""
    global _STYLES_CACHE, _CACHE_MTIME
    target = _resolve_yaml_path(yaml_path)

    if not target.exists():
        logger.warning("Emotion configuration file not found at %s", target)
        return {}

    try:
        current_mtime = target.stat().st_mtime
        if _STYLES_CACHE is not None and yaml_path is None and current_mtime == _CACHE_MTIME:
            return _STYLES_CACHE

        with open(target, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        styles: Dict[str, Dict[str, float]] = data.get("voice_styles", {})
        if yaml_path is None:
            _STYLES_CACHE = styles
            _CACHE_MTIME = current_mtime

        return styles
    except Exception as exc:
        logger.error("Failed to parse voice styles from %s: %s", target, exc)
        return _STYLES_CACHE or {}


def get_voice_style(
    emotion: str = "neutral",
    intensity: float = 1.0,
    yaml_path: Optional[Path | str] = None,
) -> VoiceStyle:
    """Map emotion and intensity to Piper settings and pitch shift.

    Interpolates linearly between baseline and target emotion style based on intensity.
    All style numbers are read directly from data/emotion.yaml.

    Args:
        emotion: Target emotion name (e.g. 'angry', 'sad', 'romantic', 'excited').
        intensity: Expression strength between 0.0 and 1.0.
        yaml_path: Optional custom path to emotion.yaml (for testing).

    Returns:
        VoiceStyle instance with length_scale, noise_scale, noise_w, and pitch_shift.
    """
    styles = load_voice_styles(yaml_path)
    baseline_dict = styles.get("baseline", {})
    if not baseline_dict:
        baseline_dict = {
            "length_scale": 1.0,
            "noise_scale": 0.667,
            "noise_w": 0.8,
            "pitch_shift": 1.0,
        }

    clean_emotion = emotion.strip().lower() if emotion else "neutral"
    target_dict = styles.get(clean_emotion, baseline_dict)

    clamped_intensity = max(0.0, min(1.0, float(intensity)))

    def _interp(param: str) -> float:
        b = float(baseline_dict.get(param, 1.0))
        t = float(target_dict.get(param, b))
        return b + clamped_intensity * (t - b)

    style = VoiceStyle(
        length_scale=round(_interp("length_scale"), 4),
        noise_scale=round(_interp("noise_scale"), 4),
        noise_w=round(_interp("noise_w"), 4),
        pitch_shift=round(_interp("pitch_shift"), 4),
    )
    logger.debug("Mapped emotion=%s intensity=%.2f to %s", emotion, clamped_intensity, style)
    return style


def apply_pitch_shift(audio_data: np.ndarray, pitch_shift: float) -> np.ndarray:
    """Pitch-shift audio using 1D numpy linear interpolation resampling.

    Args:
        audio_data: 1D numpy array of audio samples (e.g. int16 or float32).
        pitch_shift: Multiplier for pitch frequency (e.g. 1.05 = higher, 0.92 = lower).

    Returns:
        Resampled numpy array with scaled frequency content.
    """
    if abs(pitch_shift - 1.0) < 0.005 or len(audio_data) < 2:
        return audio_data

    orig_len = len(audio_data)
    new_len = max(1, int(round(orig_len / pitch_shift)))

    x_orig = np.arange(orig_len)
    x_new = np.linspace(0, orig_len - 1, new_len)

    resampled = np.interp(x_new, x_orig, audio_data)
    return resampled.astype(audio_data.dtype)
