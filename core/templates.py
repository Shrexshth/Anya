"""Canned speech replies selected by emotion and operational mode without LLM."""

from collections import defaultdict, deque
import logging
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Set

import yaml

from config import get_settings

logger = logging.getLogger("anya.core.templates")

# Guilt-tripping forbidden patterns (Guardrail 6)
_GUILT_TRIPPING_PATTERNS: List[re.Pattern] = [
    re.compile(r"\byou don't care\b", re.IGNORECASE),
    re.compile(r"\byou abandoned me\b", re.IGNORECASE),
    re.compile(r"\bwhy do you hate me\b", re.IGNORECASE),
    re.compile(r"\byou never listen\b", re.IGNORECASE),
    re.compile(r"\byou forgot about me\b", re.IGNORECASE),
    re.compile(r"\bnobody loves me\b", re.IGNORECASE),
]


def is_guilt_tripping(text: str) -> bool:
    """Return True if text contains guilt-tripping phrases."""
    for pattern in _GUILT_TRIPPING_PATTERNS:
        if pattern.search(text):
            return True
    return False


class SafeTemplateDict(dict):
    """Dictionary that leaves undefined formatting slots unchanged."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class TemplateManager:
    """Loads and selects canned responses from data/templates.yaml with 3-turn non-repeat memory."""

    def __init__(self, templates_path: Optional[Path | str] = None) -> None:
        settings = get_settings()
        self.templates_path = (
            Path(templates_path) if templates_path else settings.DATA_DIR / "templates.yaml"
        )
        self.templates: Dict[str, Any] = {}
        # History tracks the last 3 chosen template strings per (kind, emotion, mode)
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=3))
        self.load_templates()

    def load_templates(self) -> None:
        """Load canned templates from YAML."""
        if not self.templates_path.exists():
            logger.warning("Templates file not found at %s", self.templates_path)
            return

        try:
            with open(self.templates_path, "r", encoding="utf-8") as f:
                self.templates = yaml.safe_load(f) or {}
                logger.debug("Loaded templates with kinds: %s", list(self.templates.keys()))
        except Exception as exc:
            logger.exception("Failed to load templates from %s: %s", self.templates_path, exc)

    def pick(
        self,
        kind: str,
        emotion: str = "neutral",
        mode: str = "friend_roast",
        **slots: Any,
    ) -> str:
        """Pick a canned template for the given kind, emotion, and mode without repeating recent 3.

        Args:
            kind: Template context (ack_done, ack_fail, confirm_ask, greet, goodnight, miss_you, etc.)
            emotion: Dominant emotion state (happy, excited, smug, romantic, shy, sad, angry, neutral)
            mode: Personality mode (friend_roast, romantic, study, sassy, chill, professional)
            **slots: Format arguments to substitute into template

        Returns:
            The formatted response string.
        """
        kind_group = self.templates.get(kind, {})

        candidates: List[str] = []
        if isinstance(kind_group, dict):
            # Check mode first (for mode_switched), then emotion, then neutral/default
            if mode in kind_group and isinstance(kind_group[mode], list):
                candidates = list(kind_group[mode])
            elif emotion in kind_group and isinstance(kind_group[emotion], list):
                candidates = list(kind_group[emotion])
            elif "neutral" in kind_group and isinstance(kind_group["neutral"], list):
                candidates = list(kind_group["neutral"])
            elif "default" in kind_group and isinstance(kind_group["default"], list):
                candidates = list(kind_group["default"])
        elif isinstance(kind_group, list):
            candidates = list(kind_group)

        if not candidates:
            # Fallback if no templates defined
            fallback_map = {
                "ack_done": "Task complete.",
                "ack_fail": "An error occurred.",
                "confirm_ask": "Please confirm if you want to proceed.",
                "greet": "Hello! How can I help you?",
                "goodnight": "Goodnight! Sleep well.",
                "miss_you": "Welcome back! Good to see you.",
                "unknown": "I didn't quite catch that.",
                "brain_offline": "Inference model is currently offline.",
                "mode_switched": f"Switched to {mode} mode.",
            }
            raw = fallback_map.get(kind, "Understood.")
            return raw.format_map(SafeTemplateDict(**slots))

        # Filter out guilt-tripping text (Guardrail 6)
        valid_candidates = [c for c in candidates if not is_guilt_tripping(c)]
        if not valid_candidates:
            valid_candidates = candidates

        history_key = f"{kind}:{emotion}:{mode}"
        recent: Set[str] = set(self._history[history_key])

        # Filter candidates not in the last 3 picks
        unpicked = [c for c in valid_candidates if c not in recent]
        if not unpicked:
            unpicked = valid_candidates

        chosen = random.choice(unpicked)
        self._history[history_key].append(chosen)

        # Substitute slots safely
        try:
            return chosen.format_map(SafeTemplateDict(**slots))
        except Exception:
            return chosen


_TEMPLATES_INSTANCE: Optional[TemplateManager] = None


def get_template_manager() -> TemplateManager:
    """Return default singleton TemplateManager instance."""
    global _TEMPLATES_INSTANCE
    if _TEMPLATES_INSTANCE is None:
        _TEMPLATES_INSTANCE = TemplateManager()
    return _TEMPLATES_INSTANCE


def pick(
    kind: str,
    emotion: str = "neutral",
    mode: str = "friend_roast",
    **slots: Any,
) -> str:
    """Convenience functional wrapper around TemplateManager.pick."""
    return get_template_manager().pick(kind=kind, emotion=emotion, mode=mode, **slots)
