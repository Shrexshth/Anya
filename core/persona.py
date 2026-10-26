"""Personality modes, style hints for LLM, mode switching, and VIP override rule."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import yaml

from config import get_settings

logger = logging.getLogger("anya.core.persona")

MODE_FRIEND_ROAST = "friend_roast"
MODE_ROMANTIC = "romantic"
MODE_STUDY = "study"
MODE_SASSY = "sassy"
MODE_CHILL = "chill"
MODE_PROFESSIONAL = "professional"

ALL_MODES = [
    MODE_FRIEND_ROAST,
    MODE_ROMANTIC,
    MODE_STUDY,
    MODE_SASSY,
    MODE_CHILL,
    MODE_PROFESSIONAL,
]

MODE_STYLE_HINTS: Dict[str, str] = {
    MODE_FRIEND_ROAST: "Tease and playfully banter with the user like a witty, snarky best friend.",
    MODE_ROMANTIC: "Speak affectionately, warmly, and lovingly with sweet devotion and care.",
    MODE_STUDY: "Keep responses concise, disciplined, encouraging, and focused on helping the user study and stay productive.",
    MODE_SASSY: "Respond with sharp, humorous attitude, witty comebacks, and confident flair.",
    MODE_CHILL: "Speak softly, calmly, and lazily with tranquil, easygoing vibes.",
    MODE_PROFESSIONAL: "Speak with utmost politeness, clarity, concise precision, and executive professionalism.",
}

# Voice command switch regex triggers
_MODE_SWITCH_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\b(?:switch to|enter|activate|turn on|go into|be in)?\s*(?:friend|roast|friend roast)\s*(?:mode)?\b",
            re.IGNORECASE,
        ),
        MODE_FRIEND_ROAST,
    ),
    (
        re.compile(
            r"\b(?:switch to|enter|activate|turn on|go into|be in)?\s*romantic\s*(?:mode)?\b",
            re.IGNORECASE,
        ),
        MODE_ROMANTIC,
    ),
    (
        re.compile(
            r"\b(?:switch to|enter|activate|turn on|go into|be in)?\s*study\s*(?:mode)?\b",
            re.IGNORECASE,
        ),
        MODE_STUDY,
    ),
    (
        re.compile(
            r"\b(?:switch to|enter|activate|turn on|go into|be in)?\s*sassy\s*(?:mode)?\b",
            re.IGNORECASE,
        ),
        MODE_SASSY,
    ),
    (
        re.compile(
            r"\b(?:switch to|enter|activate|turn on|go into|be in)?\s*chill\s*(?:mode)?\b",
            re.IGNORECASE,
        ),
        MODE_CHILL,
    ),
    (
        re.compile(
            r"\b(?:switch to|enter|activate|turn on|go into|be in)?\s*(?:professional|work)\s*(?:mode)?\b",
            re.IGNORECASE,
        ),
        MODE_PROFESSIONAL,
    ),
]

DEFAULT_VIP_NAMES = ["Professor", "Dr. Smith", "Boss", "Inspector", "Loid", "Yor"]


def load_vip_names(lore_path: Optional[Path | str] = None) -> List[str]:
    """Load VIP names from data/lore.yaml or return defaults."""
    settings = get_settings()
    path = Path(lore_path) if lore_path else settings.DATA_DIR / "lore.yaml"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                vips = data.get("vip_names")
                if isinstance(vips, list) and vips:
                    return [str(v).strip() for v in vips if str(v).strip()]
        except Exception as exc:
            logger.warning("Could not read vip_names from %s: %s", path, exc)
    return list(DEFAULT_VIP_NAMES)


def match_mode_command(text: str) -> Optional[str]:
    """Check if user text is a voice command to switch personality modes.

    Returns:
        The target mode name if a command is detected, else None.
    """
    clean_text = text.strip()
    if not clean_text:
        return None

    # Specific rule: "friend mode" always leaves romantic mode and switches to friend_roast
    for pattern, target_mode in _MODE_SWITCH_PATTERNS:
        if pattern.search(clean_text):
            return target_mode
    return None


def check_vip_presence(text: str, vip_names: Optional[List[str]] = None) -> Optional[str]:
    """Check if text contains any VIP name from lore."""
    names = vip_names if vip_names is not None else load_vip_names()
    clean_text = text.strip()
    for name in names:
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, clean_text, re.IGNORECASE):
            return name
    return None


@dataclass
class PersonaManager:
    """Manages active personality mode, style hints, and VIP overrides."""

    current_mode: str = MODE_FRIEND_ROAST
    vip_names: List[str] = field(default_factory=load_vip_names)

    def set_mode(self, mode: str) -> bool:
        """Switch to a valid personality mode.

        Note: Switching to 'friend_roast' automatically exits romantic mode.
        """
        if mode not in ALL_MODES:
            logger.warning("Attempted to set unknown personality mode: '%s'", mode)
            return False

        old_mode = self.current_mode
        self.current_mode = mode
        logger.info("Personality mode transitioned: '%s' -> '%s'", old_mode, mode)
        return True

    def get_style_hint(self, mode: Optional[str] = None) -> str:
        """Get the one-line LLM prompt style hint for the mode."""
        active = mode or self.current_mode
        return MODE_STYLE_HINTS.get(active, MODE_STYLE_HINTS[MODE_FRIEND_ROAST])

    def evaluate_turn_mode(self, user_text: str) -> Tuple[str, bool, Optional[str]]:
        """Evaluate mode for the current turn, applying voice switches and VIP override.

        Returns:
            Tuple of (effective_mode, is_vip_override, matched_vip_name)
        """
        # 1. VIP rule in code: if text contains a VIP name, force professional mode for that turn
        vip_found = check_vip_presence(user_text, self.vip_names)
        if vip_found is not None:
            logger.info("VIP '%s' detected in text; forcing professional mode for this turn", vip_found)
            return MODE_PROFESSIONAL, True, vip_found

        # 2. Check for explicit mode switch command
        switch_mode = match_mode_command(user_text)
        if switch_mode is not None:
            self.set_mode(switch_mode)
            return self.current_mode, False, None

        return self.current_mode, False, None


_PERSONA_INSTANCE: Optional[PersonaManager] = None


def get_persona_manager() -> PersonaManager:
    """Return singleton PersonaManager instance."""
    global _PERSONA_INSTANCE
    if _PERSONA_INSTANCE is None:
        _PERSONA_INSTANCE = PersonaManager()
    return _PERSONA_INSTANCE
