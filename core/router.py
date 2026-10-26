"""Tier 0 exact regex rules, Tier 1 embedding intent matching, and Tier 2 LLM hand-off router."""

from dataclasses import dataclass, field
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from config import get_settings
from core.intents import (
    get_installed_apps,
    get_intent_matcher,
    match_app_name,
    parse_time_or_duration,
)
from core.templates import pick

logger = logging.getLogger("anya.core.router")

# Default thresholds as specified in WP8 (can be overridden in config or .env)
DEFAULT_INTENT_HIGH_THRESHOLD = 0.82
DEFAULT_INTENT_LOW_THRESHOLD = 0.65


@dataclass
class RouteResult:
    """Routing outcome containing identified tier, target skill, arguments, and confidence."""

    tier: int
    skill: str
    args: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    latency_ms: float = 0.0
    disambiguation_prompt: Optional[str] = None


class RouterStats:
    """Thread-safe statistics tracking for router tiers and latencies."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.tier_0_count: int = 0
        self.tier_1_count: int = 0
        self.tier_2_count: int = 0
        self.disambiguation_count: int = 0
        self.total_latency_ms: float = 0.0

    def record(self, tier: int, latency_ms: float, is_disambiguation: bool = False) -> None:
        """Record a completed routing request into counters."""
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            if tier == 0:
                self.tier_0_count += 1
            elif tier == 1:
                self.tier_1_count += 1
                if is_disambiguation:
                    self.disambiguation_count += 1
            elif tier == 2:
                self.tier_2_count += 1

    def reset(self) -> None:
        """Reset all counters to zero."""
        with self._lock:
            self.total_requests = 0
            self.tier_0_count = 0
            self.tier_1_count = 0
            self.tier_2_count = 0
            self.disambiguation_count = 0
            self.total_latency_ms = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Return a snapshot dictionary of current routing metrics."""
        with self._lock:
            total = self.total_requests
            avg_latency = (self.total_latency_ms / total) if total > 0 else 0.0
            llm_pct = (self.tier_2_count / total * 100.0) if total > 0 else 0.0
            t0_pct = (self.tier_0_count / total * 100.0) if total > 0 else 0.0
            t1_pct = (self.tier_1_count / total * 100.0) if total > 0 else 0.0

            return {
                "total_requests": total,
                "tier_0_count": self.tier_0_count,
                "tier_0_percent": round(t0_pct, 2),
                "tier_1_count": self.tier_1_count,
                "tier_1_percent": round(t1_pct, 2),
                "tier_2_count": self.tier_2_count,
                "tier_2_percent": round(llm_pct, 2),
                "disambiguation_count": self.disambiguation_count,
                "avg_latency_ms": round(avg_latency, 2),
            }

    def format_summary(self) -> str:
        """Produce a formatted human-readable summary of routing metrics."""
        stats = self.get_stats()
        return (
            f"Router Stats — Total: {stats['total_requests']} | "
            f"Tier 0: {stats['tier_0_count']} ({stats['tier_0_percent']}%) | "
            f"Tier 1: {stats['tier_1_count']} ({stats['tier_1_percent']}%) | "
            f"Tier 2 (LLM): {stats['tier_2_count']} ({stats['tier_2_percent']}%) | "
            f"Avg Latency: {stats['avg_latency_ms']:.2f} ms"
        )


_STATS = RouterStats()


def get_router_stats() -> RouterStats:
    """Return the global RouterStats instance."""
    return _STATS


class Tier0Matcher:
    """Whole-word regex-based zero-shot rule matcher for critical and local commands."""

    def __init__(self) -> None:
        self._build_rules()

    def _build_rules(self) -> None:
        """Compile regex patterns with strict word boundaries."""
        # 1. Mode switch canonical mapping
        self.mode_map = {
            "friend roast": "friend_roast",
            "friend": "friend_roast",
            "roast": "friend_roast",
            "romantic": "romantic",
            "study": "study",
            "sassy": "sassy",
            "chill": "chill",
            "professional": "professional",
            "work": "professional",
        }

        # 2. Shutdown / Quit / Exit
        # Strictly matches whole-word quit, shut down, exit; quit inside quite will NOT match.
        self.rx_quit_standalone = re.compile(
            r"^(?:please\s+)?(?:shut\s*down|shutdown|quit|exit|goodbye|bye(?:\s+bye)?|power\s*off)(?:\s+(?:anya|the\s+app|application))?[!\.]*$",
            re.IGNORECASE,
        )

        # 3. Stop / Hush
        self.rx_stop = re.compile(
            r"^(?:please\s+)?(?:stop(?:\s+speaking|\s+talking|\s+it)?|shut\s*up|be\s*quiet|silence|hush|cancel)[!\.]*$",
            re.IGNORECASE,
        )

        # 4. Mute / Unmute
        self.rx_unmute = re.compile(
            r"^(?:please\s+)?unmute(?:\s+(?:the\s+)?(?:audio|sound|volume|music|speaker))?[!\.]*$",
            re.IGNORECASE,
        )
        self.rx_mute = re.compile(
            r"^(?:please\s+)?mute(?:\s+(?:the\s+)?(?:audio|sound|volume|music|speaker))?[!\.]*$",
            re.IGNORECASE,
        )

        # 5. Mode switches
        self.rx_mode_switch = re.compile(
            r"^(?:please\s+)?(?:switch\s+(?:mode\s+)?to|change\s+(?:mode\s+)?to|enter|activate|turn\s+on|go\s+into|be\s+in)?\s*"
            r"(?P<mode>friend\s+roast|friend|roast|romantic|study|sassy|chill|professional|work)"
            r"(?:\s+mode)?[!\.]*$",
            re.IGNORECASE,
        )

        # 6. Volume N
        self.rx_volume_n = re.compile(
            r"^(?:please\s+)?(?:set\s+(?:the\s+)?volume\s+(?:to\s+)?|volume\s+(?:to\s+)?|turn\s+(?:the\s+)?volume\s+(?:to\s+)?)"
            r"(?P<level>\d+)(?:\s*%)?[!\.]*$",
            re.IGNORECASE,
        )

        # 7. Volume up / down
        self.rx_volume_up = re.compile(
            r"^(?:please\s+)?(?:volume\s+up|turn\s+(?:the\s+)?volume\s+up|turn\s+it\s+up|increase\s+(?:the\s+)?volume|louder)[!\.]*$",
            re.IGNORECASE,
        )
        self.rx_volume_down = re.compile(
            r"^(?:please\s+)?(?:volume\s+down|turn\s+(?:the\s+)?volume\s+down|turn\s+it\s+down|decrease\s+(?:the\s+)?volume|lower\s+(?:the\s+)?volume|softer)[!\.]*$",
            re.IGNORECASE,
        )

        # 8. Time / Date
        self.rx_time = re.compile(
            r"^(?:please\s+)?(?:what(?:\s+is|\s*'s)?\s+(?:the\s+)?time(?:\s+now|\s+is\s+it)?|tell\s+me\s+(?:the\s+)?time|current\s+time)[?\.]*$",
            re.IGNORECASE,
        )
        self.rx_date = re.compile(
            r"^(?:please\s+)?(?:what(?:\s+is|\s*'s)?\s+(?:the\s+|today\s*'s\s+)?date(?:\s+today)?|tell\s+me\s+(?:the\s+)?date|what\s+day\s+is\s+it(?:\s+today)?|today\s*'s\s+date)[?\.]*$",
            re.IGNORECASE,
        )

        # 9. Music controls
        self.rx_music_pause = re.compile(
            r"^(?:please\s+)?(?:pause\s+(?:the\s+)?(?:music|song|playback|audio|spotify)|music\s+pause|pause)[!\.]*$",
            re.IGNORECASE,
        )
        self.rx_music_resume = re.compile(
            r"^(?:please\s+)?(?:resume\s+(?:the\s+)?(?:music|song|playback|audio)|music\s+resume|unpause(?:\s+music)?|continue\s+music|resume|play\s+music)[!\.]*$",
            re.IGNORECASE,
        )
        self.rx_music_next = re.compile(
            r"^(?:please\s+)?(?:next\s+(?:song|track)|skip\s+(?:song|track)?|music\s+next|next)[!\.]*$",
            re.IGNORECASE,
        )
        self.rx_music_prev = re.compile(
            r"^(?:please\s+)?(?:previous\s+(?:song|track)|last\s+(?:song|track)|music\s+previous|go\s+back\s+a\s+song|previous)[!\.]*$",
            re.IGNORECASE,
        )

        # 10. Timer
        self.rx_timer = re.compile(
            r"^(?:please\s+)?(?:set\s+(?:a\s+)?timer\s+(?:for\s+)?(?P<dur>.+)|timer\s+(?:for\s+)?(?P<dur_alt>\d+\s*(?:minutes?|mins?|hours?|hrs?|seconds?|secs?)))[!\.]*$",
            re.IGNORECASE,
        )

        # 11. Open App / Close App
        self.rx_open = re.compile(
            r"^(?:please\s+)?(?:open|launch|start)\s+(?P<app>[a-zA-Z0-9_\-\.\s]+?)(?:\s+(?:app|application))?[!\.]*$",
            re.IGNORECASE,
        )
        self.rx_close = re.compile(
            r"^(?:please\s+)?(?:close|terminate|kill|quit)\s+(?P<app>[a-zA-Z0-9_\-\.\s]+?)(?:\s+(?:app|application))?[!\.]*$",
            re.IGNORECASE,
        )

        # 12. Show stats
        self.rx_stats = re.compile(
            r"^(?:please\s+)?(?:show\s+(?:router\s+)?stats|router\s+stats|show\s+routing\s+stats|stats)[!\.]*$",
            re.IGNORECASE,
        )

    def match(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Evaluate text against Tier 0 rules using whole-word regexes.

        Args:
            text: Clean user utterance string.

        Returns:
            Tuple of (skill_name, args) if matched, else None.
        """
        clean = text.strip()
        if not clean:
            return None

        # Stats query
        if self.rx_stats.search(clean):
            stats = get_router_stats().get_stats()
            summary = get_router_stats().format_summary()
            return "router.stats", {"stats": stats, "summary": summary}

        # Timer query (check before open/close)
        m_timer = self.rx_timer.search(clean)
        if m_timer:
            dur_str = m_timer.group("dur") or m_timer.group("dur_alt") or ""
            parsed = parse_time_or_duration(dur_str)
            return "system.timer", parsed

        # Stop command
        if self.rx_stop.search(clean):
            return "system.stop", {}

        # Mute / Unmute
        if self.rx_unmute.search(clean):
            return "system.unmute", {"action": "unmute"}
        if self.rx_mute.search(clean):
            return "system.mute", {"action": "mute"}

        # Volume N
        m_vol = self.rx_volume_n.search(clean)
        if m_vol:
            lvl = int(m_vol.group("level"))
            clamped = max(0, min(100, lvl))
            return "system.volume_set", {"level": clamped}

        # Volume Up / Down
        if self.rx_volume_up.search(clean):
            return "system.volume_up", {"step": 10}
        if self.rx_volume_down.search(clean):
            return "system.volume_down", {"step": 10}

        # Time / Date
        if self.rx_time.search(clean):
            return "system.time", {}
        if self.rx_date.search(clean):
            return "system.date", {}

        # Music controls
        if self.rx_music_pause.search(clean):
            return "media.pause", {}
        if self.rx_music_resume.search(clean):
            return "media.resume", {}
        if self.rx_music_next.search(clean):
            return "media.next", {}
        if self.rx_music_prev.search(clean):
            return "media.previous", {}

        # Mode switches
        m_mode = self.rx_mode_switch.search(clean)
        if m_mode:
            raw_mode = m_mode.group("mode").lower()
            canon_mode = self.mode_map.get(raw_mode, raw_mode)
            return "persona.switch_mode", {"mode": canon_mode}

        # Close App (checked before standalone quit so 'close Notes' closes Notes)
        m_close = self.rx_close.search(clean)
        if m_close:
            raw_app = m_close.group("app").strip()
            # If user said 'quit anya' or 'quit the app', that's system.quit, handled below
            if raw_app.lower() not in {"anya", "the app", "application", "this app", "companion"}:
                matched_app = match_app_name(raw_app)
                return "apps.close", {"app_name": matched_app, "raw_app": raw_app}

        # Open App
        m_open = self.rx_open.search(clean)
        if m_open:
            raw_app = m_open.group("app").strip()
            # Prevent open from swallowing conversational phrases like 'open your eyes' or 'open mind'
            non_app_words = {"your", "the door", "my", "our", "a", "up", "a window", "mind"}
            if raw_app.lower() not in non_app_words:
                matched_app = match_app_name(raw_app)
                return "apps.open", {"app_name": matched_app, "raw_app": raw_app}

        # Standalone Shutdown / Quit / Exit
        # Ensure 'quit' inside 'quite' or other words does not trigger.
        if self.rx_quit_standalone.search(clean):
            return "system.quit", {}

        return None


_TIER0_MATCHER = Tier0Matcher()


class Router:
    """Multi-tier intent router coordinating exact regex rules, embeddings, and LLM fallback."""

    def __init__(self, intents_path: Optional[os.PathLike | str] = None) -> None:
        self.settings = get_settings()
        self.tier0 = _TIER0_MATCHER
        self.intents_matcher = get_intent_matcher(intents_path=intents_path)
        self.stats = get_router_stats()

    @property
    def high_threshold(self) -> float:
        """Confidence threshold above which Tier 1 directly executes skill."""
        return float(
            getattr(
                self.settings,
                "INTENT_HIGH_THRESHOLD",
                os.getenv("INTENT_HIGH_THRESHOLD", DEFAULT_INTENT_HIGH_THRESHOLD),
            )
        )

    @property
    def low_threshold(self) -> float:
        """Confidence threshold below which intent is routed to Tier 2 LLM."""
        return float(
            getattr(
                self.settings,
                "INTENT_LOW_THRESHOLD",
                os.getenv("INTENT_LOW_THRESHOLD", DEFAULT_INTENT_LOW_THRESHOLD),
            )
        )

    def route(self, text: str) -> RouteResult:
        """Route user text through Tier 0 regex, Tier 1 intent embedding, or Tier 2 LLM.

        Args:
            text: Raw user utterance string.

        Returns:
            RouteResult instance with tier, target skill, args, and confidence.
        """
        start_time = time.perf_counter()
        clean = text.strip()

        # Empty input edge case
        if not clean:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.stats.record(tier=0, latency_ms=latency_ms)
            return RouteResult(
                tier=0,
                skill="noop",
                args={},
                confidence=1.0,
                latency_ms=latency_ms,
            )

        # ----------------------------------------------------------------------
        # Tier 0: Exact Rules with whole-word regex
        # ----------------------------------------------------------------------
        tier0_match = self.tier0.match(clean)
        if tier0_match:
            skill_name, args = tier0_match
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.stats.record(tier=0, latency_ms=latency_ms)
            logger.info(
                "Route '%s' -> Tier 0 [%s] (latency=%.2f ms, confidence=1.00)",
                clean,
                skill_name,
                latency_ms,
            )
            return RouteResult(
                tier=0,
                skill=skill_name,
                args=args,
                confidence=1.0,
                latency_ms=latency_ms,
            )

        # ----------------------------------------------------------------------
        # Tier 1: Fuzzy Intent Match via Fastembed
        # ----------------------------------------------------------------------
        best_skill, sim, best_phrase, slots = self.intents_matcher.match(clean)

        high_thresh = self.high_threshold
        low_thresh = self.low_threshold

        # High confidence match: Run the skill directly
        if best_skill is not None and sim >= high_thresh:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.stats.record(tier=1, latency_ms=latency_ms, is_disambiguation=False)
            logger.info(
                "Route '%s' -> Tier 1 [%s] (latency=%.2f ms, confidence=%.3f)",
                clean,
                best_skill,
                latency_ms,
                sim,
            )
            return RouteResult(
                tier=1,
                skill=best_skill,
                args=slots,
                confidence=sim,
                latency_ms=latency_ms,
            )

        # Borderline confidence match: Disambiguate with template ("Did you mean X?")
        if best_skill is not None and sim >= low_thresh:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.stats.record(tier=1, latency_ms=latency_ms, is_disambiguation=True)

            # Generate clarification question using template system
            clarify_prompt = f"Did you mean {best_skill}?"
            try:
                clarify_prompt = pick(
                    "did_you_mean",
                    suggestion=best_skill,
                    phrase=best_phrase,
                )
                if not clarify_prompt or clarify_prompt.strip() == "Understood.":
                    clarify_prompt = f"Did you mean {best_skill}?"
            except Exception:
                clarify_prompt = f"Did you mean {best_skill}?"

            disambiguation_args = {
                "suggested_skill": best_skill,
                "matched_phrase": best_phrase,
                "prompt": clarify_prompt,
                **slots,
            }

            logger.info(
                "Route '%s' -> Tier 1 Disambiguate [%s] (latency=%.2f ms, confidence=%.3f)",
                clean,
                best_skill,
                latency_ms,
                sim,
            )
            return RouteResult(
                tier=1,
                skill="clarify",
                args=disambiguation_args,
                confidence=sim,
                latency_ms=latency_ms,
                disambiguation_prompt=clarify_prompt,
            )

        # ----------------------------------------------------------------------
        # Tier 2: LLM Fallback (sim < low_threshold)
        # ----------------------------------------------------------------------
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self.stats.record(tier=2, latency_ms=latency_ms)
        logger.info(
            "Route '%s' -> Tier 2 (LLM) [sim=%.3f < %.2f] (latency=%.2f ms)",
            clean,
            sim,
            low_thresh,
            latency_ms,
        )
        return RouteResult(
            tier=2,
            skill="llm",
            args={"query": clean},
            confidence=sim,
            latency_ms=latency_ms,
        )


_ROUTER_INSTANCE: Optional[Router] = None
_ROUTER_LOCK = threading.Lock()


def get_router(intents_path: Optional[os.PathLike | str] = None) -> Router:
    """Return thread-safe singleton Router instance."""
    global _ROUTER_INSTANCE
    with _ROUTER_LOCK:
        if _ROUTER_INSTANCE is None:
            _ROUTER_INSTANCE = Router(intents_path=intents_path)
        return _ROUTER_INSTANCE


def route(text: str) -> RouteResult:
    """Route user text through the Anya multi-tier intent architecture.

    Args:
        text: User transcription or text input.

    Returns:
        RouteResult describing tier (0, 1, or 2), target skill, args, and confidence.
    """
    return get_router().route(text)
