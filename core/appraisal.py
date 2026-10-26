"""Sentiment analysis and event appraisal turning interactions into mood deltas."""

import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yaml

from config import get_settings

logger = logging.getLogger("anya.core.appraisal")


class AppraisalEngine:
    """Evaluates user text and system events, producing scaled mood dimension deltas."""

    def __init__(self, config_path: Optional[Path | str] = None) -> None:
        settings = get_settings()
        self.config_path = (
            Path(config_path) if config_path else settings.DATA_DIR / "emotion.yaml"
        )
        self.vader = SentimentIntensityAnalyzer()
        self.keywords: Dict[str, List[str]] = {}
        self.event_deltas: Dict[str, Dict[str, float]] = {}
        self.load_config()

    def load_config(self) -> None:
        """Load keyword lists and event deltas from emotion.yaml."""
        if not self.config_path.exists():
            logger.warning("Emotion config not found at %s", self.config_path)
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.keywords = data.get("keywords", {})
                self.event_deltas = data.get("event_deltas", {})
        except Exception as exc:
            logger.exception("Failed to load appraisal config from %s: %s", self.config_path, exc)

    def extract_keywords(self, text: str) -> List[str]:
        """Detect matched keyword categories in user text."""
        clean = text.lower()
        matched: List[str] = []

        for category, terms in self.keywords.items():
            for term in terms:
                # Word boundary match for single words, substring match for multi-word phrases
                if " " in term:
                    if term in clean:
                        matched.append(category)
                        break
                else:
                    pattern = r"\b" + re.escape(term) + r"\b"
                    if re.search(pattern, clean):
                        matched.append(category)
                        break
        return matched

    def appraise_user_text(self, text: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Compute mood deltas from VADER sentiment and keyword rules for user text.

        Returns:
            Tuple of (deltas_dict, metadata_dict)
        """
        sentiment = self.vader.polarity_scores(text)
        compound = float(sentiment["compound"])
        categories = self.extract_keywords(text)

        deltas: Dict[str, float] = {
            "happiness": 0.0,
            "anger": 0.0,
            "sadness": 0.0,
            "affection": 0.0,
            "energy": 0.0,
            "playfulness": 0.0,
            "boredom": 0.0,
            "anxiety": 0.0,
        }

        # Sentiment strength multiplier (0.5 to 1.5)
        strength = 1.0 + 0.5 * abs(compound)

        # Baseline VADER sentiment shift
        if compound >= 0.05:
            deltas["happiness"] += 0.12 * compound
            deltas["energy"] += 0.08 * compound
            deltas["boredom"] -= 0.10 * compound
        elif compound <= -0.05:
            neg_strength = abs(compound)
            deltas["sadness"] += 0.12 * neg_strength
            deltas["happiness"] -= 0.15 * neg_strength
            deltas["energy"] -= 0.05 * neg_strength

        # Apply deltas for matched keyword categories (handle plural and singular)
        cat_alias_map = {
            "compliments": "compliment",
            "insults": "insult",
            "apologies": "apology",
            "roast_requests": "roast_request",
        }
        for cat in categories:
            delta_key = cat_alias_map.get(cat, cat)
            cat_deltas = self.event_deltas.get(delta_key) or self.event_deltas.get(cat, {})
            for dim, delta in cat_deltas.items():
                if dim in deltas:
                    deltas[dim] += delta * strength

        # Filter negligible deltas
        final_deltas = {k: round(v, 4) for k, v in deltas.items() if abs(v) > 0.0001}

        meta = {
            "sentiment": sentiment,
            "categories": categories,
            "strength": round(strength, 2),
        }
        return final_deltas, meta

    def appraise_event(
        self, event_name: str, payload: Optional[Any] = None
    ) -> Dict[str, float]:
        """Convert a discrete event or user text into a dictionary of mood dimension deltas."""
        if event_name == "user_text" and isinstance(payload, str):
            deltas, _ = self.appraise_user_text(payload)
            return deltas

        base_deltas = self.event_deltas.get(event_name, {})
        return {k: float(v) for k, v in base_deltas.items()}


_APPRAISAL_INSTANCE: Optional[AppraisalEngine] = None


def get_appraisal_engine() -> AppraisalEngine:
    """Return default singleton AppraisalEngine instance."""
    global _APPRAISAL_INSTANCE
    if _APPRAISAL_INSTANCE is None:
        _APPRAISAL_INSTANCE = AppraisalEngine()
    return _APPRAISAL_INSTANCE
