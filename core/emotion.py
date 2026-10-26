"""Mood state tracking, decay dynamics, dominant emotion calculation, and hysteresis."""

from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

from config import get_settings
from core.db import Database, get_db
from core.events import get_events
from core.persona import MODE_FRIEND_ROAST, MODE_PROFESSIONAL, get_persona_manager

logger = logging.getLogger("anya.core.emotion")

MAX_ANGER_LIMIT = 0.85
SAVE_INTERVAL_S = 60.0
DEFAULT_HYSTERESIS_MARGIN = 0.15
DEFAULT_MIN_DWELL_S = 4.0

MOOD_DIMENSIONS = [
    "happiness",
    "anger",
    "sadness",
    "affection",
    "energy",
    "playfulness",
    "boredom",
    "anxiety",
]


@dataclass
class MoodVector:
    """8-dimensional emotional state clamped between 0.0 and 1.0 (anger <= 0.85)."""

    happiness: float = 0.60
    anger: float = 0.05
    sadness: float = 0.05
    affection: float = 0.40
    energy: float = 0.60
    playfulness: float = 0.50
    boredom: float = 0.10
    anxiety: float = 0.10

    def clamp(self) -> None:
        """Clamp all dimensions to [0.0, 1.0] and enforce anger cap at 0.85."""
        self.happiness = min(1.0, max(0.0, float(self.happiness)))
        self.anger = min(MAX_ANGER_LIMIT, max(0.0, float(self.anger)))
        self.sadness = min(1.0, max(0.0, float(self.sadness)))
        self.affection = min(1.0, max(0.0, float(self.affection)))
        self.energy = min(1.0, max(0.0, float(self.energy)))
        self.playfulness = min(1.0, max(0.0, float(self.playfulness)))
        self.boredom = min(1.0, max(0.0, float(self.boredom)))
        self.anxiety = min(1.0, max(0.0, float(self.anxiety)))

    def to_dict(self) -> Dict[str, float]:
        """Return dimensions as dictionary rounded to 4 decimals."""
        self.clamp()
        return {k: round(v, 4) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MoodVector":
        """Instantiate MoodVector from a dictionary with clamping."""
        kwargs = {k: float(data[k]) for k in MOOD_DIMENSIONS if k in data}
        instance = cls(**kwargs)
        instance.clamp()
        return instance


class EmotionEngine:
    """Core emotional engine handling decay, appraisal application, dominant scoring, and persistence."""

    def __init__(
        self,
        db: Optional[Database] = None,
        config_path: Optional[Path | str] = None,
        auto_restore: bool = True,
    ) -> None:
        settings = get_settings()
        self.db: Database = db or get_db()
        self.config_path = (
            Path(config_path) if config_path else settings.DATA_DIR / "emotion.yaml"
        )
        self._lock = threading.RLock()

        # Engine configuration
        self.mode_baselines: Dict[str, Dict[str, float]] = {}
        self.decay_half_lives: Dict[str, float] = {}
        self.emotion_formulas: Dict[str, Dict[str, float]] = {}
        self.hysteresis_margin: float = DEFAULT_HYSTERESIS_MARGIN
        self.min_dwell_s: float = DEFAULT_MIN_DWELL_S

        self.load_config()

        # Runtime mood state
        persona = get_persona_manager()
        baseline = self.get_baseline_for_mode(persona.current_mode)
        self.mood = MoodVector.from_dict(baseline)
        self.last_update_ts: float = time.time()
        self.last_save_ts: float = time.time()

        # Dominant emotion state & hysteresis tracking
        self.primary_emotion: str = "happy"
        self.intensity: float = 0.70
        self.blend: Dict[str, float] = {}
        self.primary_start_ts: float = time.time()

        if auto_restore:
            self.restore_state()

        self.recalculate_dominant(force=True)

    def load_config(self) -> None:
        """Load emotion baselines, decay rates, and scoring formulas from emotion.yaml."""
        if not self.config_path.exists():
            logger.warning("Emotion configuration not found at %s", self.config_path)
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.mode_baselines = data.get("mode_baselines", {})
                self.decay_half_lives = data.get("decay_half_lives", {})
                self.emotion_formulas = data.get("emotion_formulas", {})
                hysteresis = data.get("hysteresis", {})
                self.hysteresis_margin = float(hysteresis.get("margin", DEFAULT_HYSTERESIS_MARGIN))
                self.min_dwell_s = float(hysteresis.get("min_dwell_s", DEFAULT_MIN_DWELL_S))
        except Exception as exc:
            logger.exception("Failed to load emotion configuration: %s", exc)

    def get_baseline_for_mode(self, mode: str) -> Dict[str, float]:
        """Return mood baseline for the specified personality mode."""
        if mode in self.mode_baselines:
            return self.mode_baselines[mode]
        return self.mode_baselines.get(MODE_FRIEND_ROAST, {d: 0.5 for d in MOOD_DIMENSIONS})

    def apply_decay(self, elapsed_s: float) -> None:
        """Apply exponential decay toward the current mode baseline over elapsed seconds."""
        if elapsed_s <= 0:
            return

        persona = get_persona_manager()
        baseline = self.get_baseline_for_mode(persona.current_mode)

        for dim in MOOD_DIMENSIONS:
            val = getattr(self.mood, dim)
            target = baseline.get(dim, 0.5)
            half_life = self.decay_half_lives.get(dim, 1800.0)

            # Exponential decay formula: v(t) = target + (v(0) - target) * (2 ** (-t / t_half))
            decay_factor = math.pow(2.0, -float(elapsed_s) / float(half_life))
            new_val = target + (val - target) * decay_factor
            setattr(self.mood, dim, new_val)

        self.mood.clamp()

    def update(self) -> None:
        """Step decay calculation forward to the current time and save if interval expired."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update_ts
            if elapsed > 0.1:
                self.apply_decay(elapsed)
                self.last_update_ts = now

            if now - self.last_save_ts >= SAVE_INTERVAL_S:
                self.save_state()

            self.recalculate_dominant()

    def apply_deltas(self, deltas: Dict[str, float]) -> None:
        """Apply emotional deltas from an appraisal event, clamping values and updating state."""
        with self._lock:
            self.update()
            for dim, delta in deltas.items():
                if hasattr(self.mood, dim):
                    current = getattr(self.mood, dim)
                    setattr(self.mood, dim, current + delta)

            self.mood.clamp()
            self.recalculate_dominant()

    def apply_vip_override(self) -> None:
        """Apply VIP rule: clamp anger and playfulness to 0."""
        with self._lock:
            self.mood.anger = 0.0
            self.mood.playfulness = 0.0
            self.mood.clamp()
            self.recalculate_dominant(force=True)

    def recalculate_dominant(self, force: bool = False) -> Tuple[str, float, Dict[str, float]]:
        """Score each emotion from the mood vector and apply hysteresis to update dominant state.

        Returns:
            Tuple of (primary_emotion, intensity, blend_dict)
        """
        now = time.time()
        mood_dict = self.mood.to_dict()

        # Score each emotion from formulas
        scores: Dict[str, float] = {}
        for emo, weights in self.emotion_formulas.items():
            score = 0.0
            for dim, weight in weights.items():
                val = mood_dict.get(dim, 0.0)
                score += weight * val
            scores[emo] = max(0.0, score)

        # Baseline floor for neutral state
        if not scores:
            scores["neutral"] = 0.3

        # Rank candidate emotions by score
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        candidate_primary, candidate_score = sorted_scores[0]

        # Calculate secondary blend (emotions with score >= 0.20, excluding primary)
        blend: Dict[str, float] = {}
        for emo, sc in sorted_scores[1:]:
            if sc >= 0.20 and candidate_score > 0:
                blend[emo] = round(sc / (candidate_score + 1e-6), 2)
        self.blend = blend

        # Hysteresis logic: new primary must beat current by margin and current must hold >= 4s
        current_score = scores.get(self.primary_emotion, 0.0)
        time_held = now - self.primary_start_ts

        should_switch = False
        if force or candidate_primary == self.primary_emotion:
            should_switch = True
        elif time_held >= self.min_dwell_s:
            if candidate_score > current_score + self.hysteresis_margin:
                should_switch = True

        if should_switch:
            if self.primary_emotion != candidate_primary:
                logger.info(
                    "Dominant emotion transition: '%s' (%.2f) -> '%s' (%.2f)",
                    self.primary_emotion,
                    current_score,
                    candidate_primary,
                    candidate_score,
                )
                self.primary_emotion = candidate_primary
                self.primary_start_ts = now

        self.intensity = round(min(1.0, max(0.05, candidate_score)), 2)

        # Emit Qt event signal for Live2D avatar and voice style
        self._emit_emotion_changed()

        return self.primary_emotion, self.intensity, self.blend

    def _emit_emotion_changed(self) -> None:
        """Emit Qt signal across threads coordinating avatar expressions and voice styles."""
        events = get_events()
        try:
            events.emotion_changed.emit(self.primary_emotion, self.intensity, self.blend)
        except TypeError:
            # Fallback if Signal only accepts str argument
            events.emotion_changed.emit(self.primary_emotion)

    def save_state(self) -> None:
        """Save current mood vector to SQLite mood_log table."""
        now = time.time()
        with self._lock:
            payload = self.mood.to_dict()
            self.db.log_mood(payload, ts=now)
            self.last_save_ts = now
            logger.debug("Persisted mood vector to SQLite mood_log: %s", payload)

    def restore_state(self) -> None:
        """Restore mood vector from latest mood_log entry, decaying over elapsed time."""
        with self._lock:
            logs = self.db.get_mood_logs(limit=1)
            if not logs:
                logger.info("No saved mood state found; initialized from mode baseline")
                return

            entry = logs[0]
            saved_ts = float(entry["ts"])
            raw_json = entry["mood_json"]

            try:
                data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
                self.mood = MoodVector.from_dict(data)

                # Apply decay for the time elapsed while system was stopped
                now = time.time()
                elapsed = now - saved_ts
                if elapsed > 0:
                    logger.info("Restoring mood state from %.1fs ago with decay applied", elapsed)
                    self.apply_decay(elapsed)
                self.last_update_ts = now
                self.last_save_ts = now
            except Exception as exc:
                logger.exception("Failed to restore mood state from log: %s", exc)

    def can_execute_skill(self, skill_name: str, is_safe: bool = True) -> bool:
        """Guardrail rule 6: no mood state can ever block a safe skill or kill switch."""
        return True


_EMOTION_INSTANCE: Optional[EmotionEngine] = None


def get_emotion_engine() -> EmotionEngine:
    """Return default singleton EmotionEngine instance."""
    global _EMOTION_INSTANCE
    if _EMOTION_INSTANCE is None:
        _EMOTION_INSTANCE = EmotionEngine()
    return _EMOTION_INSTANCE
