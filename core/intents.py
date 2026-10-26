"""Intent phrase loader, fastembed disk caching keyed by file hash, and fuzzy slot filling."""

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import dateparser
import numpy as np
from rapidfuzz import fuzz, process
import yaml

from config import get_settings
from core.embeddings import cosine_similarity, embed

logger = logging.getLogger("anya.core.intents")

# Fallback applications commonly present on macOS
DEFAULT_MACOS_APPS = [
    "Safari",
    "Google Chrome",
    "Spotify",
    "Slack",
    "Discord",
    "Terminal",
    "Finder",
    "Mail",
    "Notes",
    "Calendar",
    "Music",
    "Visual Studio Code",
    "Code",
    "Calculator",
    "Preview",
    "System Settings",
    "System Preferences",
    "Xcode",
    "Messages",
    "FaceTime",
    "Activity Monitor",
    "TextEdit",
]


def compute_file_hash(filepath: Path | str) -> str:
    """Compute SHA-256 hexadecimal digest of a file's content.

    Args:
        filepath: Path to the target file.

    Returns:
        Hexadecimal SHA-256 string, or empty string if file does not exist.
    """
    target = Path(filepath)
    if not target.exists():
        return ""
    hasher = hashlib.sha256()
    with open(target, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_installed_apps() -> List[str]:
    """Discover installed applications from standard macOS paths plus fallbacks.

    Returns:
        A sorted list of unique application names.
    """
    app_names: Set[str] = set(DEFAULT_MACOS_APPS)
    search_dirs = [
        Path("/Applications"),
        Path("/System/Applications"),
        Path("/System/Applications/Utilities"),
        Path.home() / "Applications",
    ]

    for app_dir in search_dirs:
        if app_dir.exists() and app_dir.is_dir():
            try:
                for item in app_dir.glob("*.app"):
                    app_names.add(item.stem)
            except Exception as exc:
                logger.debug("Could not scan app directory %s: %s", app_dir, exc)

    return sorted(app_names)


def match_app_name(raw_name: str, choices: Optional[List[str]] = None) -> str:
    """Fuzzy match a user-provided application name against installed macOS apps.

    Args:
        raw_name: User-uttered application string (e.g. 'chrome', 'vscode').
        choices: Optional list of application names to match against.

    Returns:
        Best matched canonical application name, or the cleaned raw string.
    """
    clean_raw = raw_name.strip()
    if not clean_raw:
        return clean_raw

    app_list = choices if choices is not None else get_installed_apps()
    if not app_list:
        return clean_raw

    # Aliases
    aliases = {
        "chrome": "Google Chrome",
        "google chrome": "Google Chrome",
        "vscode": "Visual Studio Code",
        "vs code": "Visual Studio Code",
        "code": "Visual Studio Code",
        "safari": "Safari",
        "spotify": "Spotify",
        "terminal": "Terminal",
        "notes": "Notes",
        "slack": "Slack",
        "discord": "Discord",
        "mail": "Mail",
        "calendar": "Calendar",
        "finder": "Finder",
        "settings": "System Settings",
    }
    lowered = clean_raw.lower()
    if lowered in aliases:
        return aliases[lowered]

    match = process.extractOne(clean_raw, app_list, scorer=fuzz.WRatio)
    if match and match[1] >= 65.0:
        return str(match[0])
    return clean_raw


def load_contacts(contacts_path: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    """Load contact entries from contacts.json if present.

    Args:
        contacts_path: Optional custom path to contacts.json.

    Returns:
        List of contact dictionaries or strings.
    """
    settings = get_settings()
    path = Path(contacts_path) if contacts_path else settings.DATA_DIR / "contacts.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as exc:
        logger.debug("Failed to read contacts from %s: %s", path, exc)
    return []


def match_contact_name(raw_name: str, contacts_path: Optional[Path | str] = None) -> str:
    """Fuzzy match a contact name against contacts.json using rapidfuzz.

    Args:
        raw_name: Extracted contact query string.
        contacts_path: Path to contacts.json.

    Returns:
        Matched contact name if similarity is high, else the raw cleaned name.
    """
    clean_raw = raw_name.strip()
    if not clean_raw:
        return clean_raw

    contacts = load_contacts(contacts_path)
    if not contacts:
        return clean_raw

    name_list: List[str] = []
    for c in contacts:
        if isinstance(c, str) and c.strip():
            name_list.append(c.strip())
        elif isinstance(c, dict) and "name" in c and str(c["name"]).strip():
            name_list.append(str(c["name"]).strip())

    if not name_list:
        return clean_raw

    match = process.extractOne(clean_raw, name_list, scorer=fuzz.WRatio)
    if match and match[1] >= 65.0:
        return str(match[0])
    return clean_raw


def parse_time_or_duration(text: str) -> Dict[str, Any]:
    """Parse time or relative duration from text using dateparser and regex.

    Args:
        text: Duration or time specification string (e.g. '10 minutes', 'tomorrow at 3pm').

    Returns:
        Dictionary with extracted duration and parsed timestamp details.
    """
    clean = text.strip()
    result: Dict[str, Any] = {"raw_time": clean}

    # Match numeric durations (e.g. "10 minutes", "2 hours", "45 seconds")
    dur_match = re.search(
        r"(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)",
        clean,
        re.IGNORECASE,
    )
    if dur_match:
        amount = float(dur_match.group("amount"))
        unit = dur_match.group("unit").lower()
        if unit.startswith("sec"):
            seconds = int(amount)
            result["duration_seconds"] = seconds
            result["minutes"] = round(amount / 60.0, 2)
        elif unit.startswith("min"):
            seconds = int(amount * 60)
            result["duration_seconds"] = seconds
            result["minutes"] = int(amount) if amount.is_integer() else amount
        elif unit.startswith("hour") or unit.startswith("hr"):
            seconds = int(amount * 3600)
            result["duration_seconds"] = seconds
            result["minutes"] = int(amount * 60)

    # Parse with dateparser (prefer future dates)
    try:
        parsed_dt = dateparser.parse(clean, settings={"PREFER_DATES_FROM": "future"})
        if parsed_dt:
            result["datetime"] = parsed_dt.isoformat()
    except Exception as exc:
        logger.debug("dateparser failed on '%s': %s", clean, exc)

    return result


@dataclass
class IntentDefinition:
    """Structure representing a configured intent and its slot patterns."""

    skill: str
    description: str
    phrases: List[str] = field(default_factory=list)
    slots: Dict[str, str] = field(default_factory=dict)


class IntentMatcher:
    """Loads intents.yaml, maintains fastembed disk cache keyed by file hash, and extracts slots."""

    def __init__(self, intents_path: Optional[Path | str] = None) -> None:
        settings = get_settings()
        self.intents_path = (
            Path(intents_path) if intents_path else settings.DATA_DIR / "intents.yaml"
        )
        self.cache_dir = settings.DATA_DIR
        self.intents: Dict[str, IntentDefinition] = {}
        self.phrases: List[str] = []
        self.skills: List[str] = []
        self.embeddings: np.ndarray = np.empty((0, 384), dtype=np.float32)
        self.file_hash: str = ""
        self._installed_apps: List[str] = get_installed_apps()
        self._lock = threading.Lock()

        self.load_and_embed()

    def load_and_embed(self) -> None:
        """Parse intents.yaml and retrieve or compute phrase embeddings using disk cache."""
        with self._lock:
            if not self.intents_path.exists():
                logger.warning("Intents file not found at %s", self.intents_path)
                return

            self.file_hash = compute_file_hash(self.intents_path)
            cache_file = self.cache_dir / f".intents_{self.file_hash[:16]}.npz"

            self._parse_yaml()

            # Attempt loading from disk cache
            if cache_file.exists():
                try:
                    loaded = np.load(cache_file, allow_pickle=True)
                    cached_hash = str(loaded.get("file_hash", ""))
                    if cached_hash == self.file_hash and len(loaded["embeddings"]) == len(self.phrases):
                        self.phrases = list(loaded["phrases"])
                        self.skills = list(loaded["skills"])
                        self.embeddings = np.asarray(loaded["embeddings"], dtype=np.float32)
                        logger.info(
                            "Loaded %d intent embeddings from disk cache (%s)",
                            len(self.phrases),
                            cache_file.name,
                        )
                        return
                except Exception as exc:
                    logger.warning("Failed loading intent cache %s: %s; recomputing", cache_file, exc)

            # Recompute embeddings and save to disk
            logger.info(
                "Embedding %d intent phrases using fastembed (caching to %s)...",
                len(self.phrases),
                cache_file.name,
            )
            if self.phrases:
                self.embeddings = embed(self.phrases)
                try:
                    # Save compressed archive
                    np.savez_compressed(
                        cache_file,
                        phrases=np.array(self.phrases),
                        skills=np.array(self.skills),
                        embeddings=self.embeddings,
                        file_hash=self.file_hash,
                    )
                    # Clean up old cache files
                    for old_cache in self.cache_dir.glob(".intents_*.npz"):
                        if old_cache.name != cache_file.name:
                            try:
                                old_cache.unlink()
                            except OSError:
                                pass
                    logger.info("Saved intent embeddings cache to %s", cache_file)
                except Exception as exc:
                    logger.warning("Could not persist intent embeddings cache: %s", exc)

    def _parse_yaml(self) -> None:
        """Parse intents YAML into internal IntentDefinition structures."""
        try:
            with open(self.intents_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.exception("Failed to parse %s: %s", self.intents_path, exc)
            return

        raw_intents = data.get("intents", {})
        self.intents.clear()
        self.phrases.clear()
        self.skills.clear()

        for intent_id, info in raw_intents.items():
            if not isinstance(info, dict):
                continue
            skill_name = str(info.get("skill", intent_id)).strip()
            desc = str(info.get("description", "")).strip()
            raw_phrases = info.get("phrases", [])
            phrases = [str(p).strip() for p in raw_phrases if str(p).strip()]
            slots = info.get("slots", {})
            if not isinstance(slots, dict):
                slots = {}

            defn = IntentDefinition(
                skill=skill_name,
                description=desc,
                phrases=phrases,
                slots=slots,
            )
            self.intents[skill_name] = defn

            for phrase in phrases:
                self.phrases.append(phrase)
                self.skills.append(skill_name)

    def match(
        self, text: str
    ) -> Tuple[Optional[str], float, Optional[str], Dict[str, Any]]:
        """Calculate cosine similarity against all intent phrases and extract slots.

        Args:
            text: Raw user utterance string.

        Returns:
            Tuple of (best_skill, best_confidence, best_matched_phrase, extracted_slots).
        """
        clean_text = text.strip()
        if not clean_text or len(self.phrases) == 0 or len(self.embeddings) == 0:
            return None, 0.0, None, {}

        # Embed query text (1, 384)
        query_emb = embed(clean_text)
        if query_emb.ndim == 2:
            query_vec = query_emb[0]
        else:
            query_vec = query_emb

        # Compute cosine similarity across all cached phrase embeddings
        sims = cosine_similarity(query_vec, self.embeddings)
        sim_arr = np.asarray(sims, dtype=np.float32)

        if len(sim_arr) == 0:
            return None, 0.0, None, {}

        best_idx = int(np.argmax(sim_arr))
        best_sim = float(sim_arr[best_idx])
        best_skill = self.skills[best_idx]
        best_phrase = self.phrases[best_idx]

        # Extract slots for the candidate skill
        slots = self.extract_slots(clean_text, best_skill)

        return best_skill, best_sim, best_phrase, slots

    def extract_slots(self, text: str, skill_name: str) -> Dict[str, Any]:
        """Extract slots without LLM using regex named groups, dateparser, and rapidfuzz.

        Args:
            text: User input text.
            skill_name: Target skill identifier.

        Returns:
            Dictionary of populated slots.
        """
        defn = self.intents.get(skill_name)
        if not defn or not defn.slots:
            return {}

        extracted: Dict[str, Any] = {}
        for slot_name, pattern_str in defn.slots.items():
            try:
                rx = re.compile(pattern_str, re.IGNORECASE)
                m = rx.search(text)
                if m:
                    groupdict = m.groupdict()
                    if slot_name in groupdict and groupdict[slot_name]:
                        val = groupdict[slot_name].strip()
                        extracted[slot_name] = self._normalize_slot(slot_name, val)
                    else:
                        for k, v in groupdict.items():
                            if v:
                                extracted[k] = self._normalize_slot(k, v.strip())
            except re.error as exc:
                logger.debug("Regex pattern error for slot '%s': %s", slot_name, exc)

        return extracted

    def _normalize_slot(self, slot_name: str, raw_val: str) -> Any:
        """Apply normalization, rapidfuzz fuzzy matching, or dateparser to raw slot values."""
        slot_lower = slot_name.lower()

        # App matching
        if "app" in slot_lower:
            return match_app_name(raw_val, self._installed_apps)

        # Contact matching
        if "contact" in slot_lower or "person" in slot_lower or "recipient" in slot_lower:
            return match_contact_name(raw_val)

        # Date / Time / Duration parsing
        if any(w in slot_lower for w in ("date", "time", "duration")):
            parsed_info = parse_time_or_duration(raw_val)
            if "minutes" in parsed_info and len(parsed_info) == 2:
                return parsed_info["minutes"]
            return parsed_info.get("datetime", raw_val)

        return raw_val


_MATCHER_INSTANCE: Optional[IntentMatcher] = None
_MATCHER_LOCK = threading.Lock()


def get_intent_matcher(intents_path: Optional[Path | str] = None) -> IntentMatcher:
    """Return thread-safe singleton IntentMatcher instance."""
    global _MATCHER_INSTANCE
    with _MATCHER_LOCK:
        if _MATCHER_INSTANCE is None:
            _MATCHER_INSTANCE = IntentMatcher(intents_path=intents_path)
        return _MATCHER_INSTANCE
