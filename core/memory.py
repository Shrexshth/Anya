"""Recent turn buffer, long-term semantic facts, and episodic memory manager for Anya."""

from collections import deque
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from config import get_settings
from core.db import Database, get_db
from core.embeddings import cosine_similarity, embed

logger = logging.getLogger("anya.core.memory")

# Rule regexes for deterministic rule capture (no LLM)
_RULE_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # (pattern, kind, response_template)
    (
        re.compile(r"^\s*(?:please\s+)?remember\s+that\s+(.+?)[.!?]*$", re.IGNORECASE),
        "general",
        "I'll remember that!",
    ),
    (
        re.compile(r"^\s*my\s+name\s+(?:is|'s)\s+([A-Za-z0-9\s'-]+?)[.!?]*$", re.IGNORECASE),
        "identity",
        "Nice to meet you, {value}! I'll remember your name.",
    ),
    (
        re.compile(r"^\s*my\s+birthday\s+(?:is|'s)\s+(.+?)[.!?]*$", re.IGNORECASE),
        "identity",
        "Got it! I'll remember your birthday is {value}.",
    ),
    (
        re.compile(
            r"^\s*i\s+(?:really\s+)?(like|love|hate|dislike)\s+(.+?)[.!?]*$", re.IGNORECASE
        ),
        "preference",
        "Noted! I'll remember that you {verb} {value}.",
    ),
    (
        re.compile(r"^\s*(?:you\s+can\s+)?call\s+me\s+([A-Za-z0-9\s'-]+?)[.!?]*$", re.IGNORECASE),
        "identity",
        "Understood, I'll call you {value}!",
    ),
]

_WHAT_REMEMBER_PATTERN = re.compile(
    r"^\s*what\s+(?:do\s+you\s+)?(?:remember|know)(?:\s+about\s+me)?\??\s*$", re.IGNORECASE
)

_FORGET_PATTERN = re.compile(
    r"^\s*(?:please\s+)?forget\s+(?:about\s+)?(.+?)[.!?]*$", re.IGNORECASE
)


def estimate_tokens(text: str) -> int:
    """Conservatively estimate token count for a text string."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(int(chars / 3.5), int(words * 1.3))


def _extract_json_block(text: str) -> str:
    """Extract JSON content from potential markdown code fences or raw text."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        return brace_match.group(1).strip()
    return text


class MemoryManager:
    """Manages short-term conversational turns, long-term semantic facts, and episodic memory."""

    def __init__(
        self,
        db: Optional[Database] = None,
        history_turns: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        idle_threshold_s: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self.db: Database = db or get_db()
        self.history_turns: int = (
            history_turns
            if history_turns is not None
            else getattr(settings, "HISTORY_TURNS", 6)
        )
        self.similarity_threshold: float = (
            similarity_threshold
            if similarity_threshold is not None
            else float(
                getattr(
                    settings,
                    "FACT_SIMILARITY_THRESHOLD",
                    os.getenv("FACT_SIMILARITY_THRESHOLD", 0.9),
                )
            )
        )
        self.idle_threshold_s: float = (
            idle_threshold_s
            if idle_threshold_s is not None
            else float(
                getattr(
                    settings,
                    "LLM_IDLE_UNLOAD_S",
                    os.getenv("SESSION_IDLE_SUMMARIZE_S", 300),
                )
            )
        )

        self.prompt_max_facts: int = int(
            getattr(settings, "PROMPT_MAX_FACTS", os.getenv("PROMPT_MAX_FACTS", 5))
        )
        self.prompt_max_episodes: int = int(
            getattr(settings, "PROMPT_MAX_EPISODES", os.getenv("PROMPT_MAX_EPISODES", 1))
        )
        self.prompt_max_tokens: int = int(
            getattr(settings, "PROMPT_MAX_TOKENS", os.getenv("PROMPT_MAX_TOKENS", 150))
        )

        self.recent_turns: deque = deque(maxlen=self.history_turns)
        self.last_interaction_ts: float = time.time()
        self._session_summarized: bool = True
        self._unsummarized_turns: int = 0

    # ---------------- Turn Management ----------------

    def add_turn(self, role: str, text: str, ts: Optional[float] = None) -> None:
        """Add a turn to the recent turns buffer."""
        turn_ts = ts if ts is not None else time.time()
        self.recent_turns.append({"role": role, "text": text, "ts": turn_ts})
        self.last_interaction_ts = turn_ts
        self._session_summarized = False
        self._unsummarized_turns += 1

    def get_recent_turns(self) -> List[Dict[str, Any]]:
        """Return a copy of recent turns in chronological order."""
        return list(self.recent_turns)

    def clear_turns(self) -> None:
        """Clear recent turns buffer."""
        self.recent_turns.clear()
        self._unsummarized_turns = 0
        self._session_summarized = True

    # ---------------- Facts & Semantic Search ----------------

    def add_fact(
        self,
        text: str,
        kind: str = "user",
        source: str = "rule",
        embedding: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        """Add a new semantic fact if no existing fact has cosine similarity >= 0.9."""
        clean_text = text.strip()
        if not clean_text:
            return None

        if embedding is None:
            emb_arr = embed([clean_text])[0]
        else:
            emb_arr = np.asarray(embedding, dtype=np.float32)

        existing_facts = self.db.get_all_facts()
        if existing_facts:
            # Check for high similarity against existing facts
            for fact in existing_facts:
                fact_emb = fact.get("embedding")
                if fact_emb is not None and len(fact_emb) > 0:
                    sim = float(cosine_similarity(emb_arr, fact_emb))
                    if sim >= self.similarity_threshold:
                        logger.info(
                            "Skipping fact '%s': similar to existing '%s' (similarity: %.3f >= %.2f)",
                            clean_text,
                            fact["text"],
                            sim,
                            self.similarity_threshold,
                        )
                        return None

        fact_id = self.db.add_fact(
            text=clean_text,
            kind=kind,
            source=source,
            embedding=emb_arr,
        )
        logger.info("Added fact [%d]: '%s' (kind=%s, source=%s)", fact_id, clean_text, kind, source)
        return fact_id

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search top-k facts by cosine similarity and update their last_used timestamp."""
        clean_query = query.strip()
        if not clean_query:
            return []

        existing_facts = self.db.get_all_facts()
        if not existing_facts:
            return []

        q_emb = embed([clean_query])[0]
        scored_facts: List[Tuple[float, Dict[str, Any]]] = []

        for fact in existing_facts:
            fact_emb = fact.get("embedding")
            if fact_emb is not None and len(fact_emb) > 0:
                sim = float(cosine_similarity(q_emb, fact_emb))
                fact_copy = dict(fact)
                fact_copy["similarity"] = round(sim, 4)
                scored_facts.append((sim, fact_copy))

        scored_facts.sort(key=lambda item: item[0], reverse=True)
        top_k = [item[1] for item in scored_facts[:k]]

        # Update last_used for retrieved facts
        now = time.time()
        for fact in top_k:
            self.db.update_fact_last_used(fact["id"], now)

        return top_k

    # ---------------- Rule-Based Memory (No LLM) ----------------

    def check_rule_capture(self, text: str) -> Optional[Tuple[str, str]]:
        """Detect deterministic memory capture rules without calling an LLM.

        Returns:
            Tuple of (confirmation_message, stored_fact_text) if captured, else None.
        """
        clean_text = text.strip()

        # 1. "remember that ..."
        match = _RULE_PATTERNS[0][0].match(clean_text)
        if match:
            captured = match.group(1).strip()
            fact_text = f"User: {captured}"
            self.add_fact(fact_text, kind="general", source="rule")
            return _RULE_PATTERNS[0][2], fact_text

        # 2. "my name is ..."
        match = _RULE_PATTERNS[1][0].match(clean_text)
        if match:
            name = match.group(1).strip()
            fact_text = f"User's name is {name}"
            self.add_fact(fact_text, kind="identity", source="rule")
            confirmation = _RULE_PATTERNS[1][2].format(value=name)
            return confirmation, fact_text

        # 3. "my birthday is ..."
        match = _RULE_PATTERNS[2][0].match(clean_text)
        if match:
            bday = match.group(1).strip()
            fact_text = f"User's birthday is {bday}"
            self.add_fact(fact_text, kind="identity", source="rule")
            confirmation = _RULE_PATTERNS[2][2].format(value=bday)
            return confirmation, fact_text

        # 4. "I like/hate ..."
        match = _RULE_PATTERNS[3][0].match(clean_text)
        if match:
            verb = match.group(1).lower()
            target = match.group(2).strip()
            fact_text = f"User {verb}s {target}"
            self.add_fact(fact_text, kind="preference", source="rule")
            confirmation = _RULE_PATTERNS[3][2].format(verb=verb, value=target)
            return confirmation, fact_text

        # 5. "call me ..."
        match = _RULE_PATTERNS[4][0].match(clean_text)
        if match:
            nickname = match.group(1).strip()
            fact_text = f"User prefers to be called {nickname}"
            self.add_fact(fact_text, kind="identity", source="rule")
            confirmation = _RULE_PATTERNS[4][2].format(value=nickname)
            return confirmation, fact_text

        return None

    def handle_direct_command(self, text: str) -> Optional[str]:
        """Handle direct memory queries ('what do you remember about me', 'forget X', rules).

        Returns:
            Direct response string if handled without LLM, else None.
        """
        clean_text = text.strip()

        # Check "what do you remember about me"
        if _WHAT_REMEMBER_PATTERN.match(clean_text):
            facts = self.db.get_all_facts()
            if not facts:
                return "I don't have any memories stored about you yet."
            lines = [f"- {fact['text']}" for fact in facts]
            return "Here is what I remember about you:\n" + "\n".join(lines)

        # Check "forget X"
        forget_match = _FORGET_PATTERN.match(clean_text)
        if forget_match:
            target = forget_match.group(1).strip()
            if target.lower() in {"everything", "all", "all memories", "all facts"}:
                self.db.clear_facts()
                return "I have forgotten everything I remembered about you."

            # Search by substring first
            all_facts = self.db.get_all_facts()
            matched = [f for f in all_facts if target.lower() in f["text"].lower()]

            # If no substring match, check similarity
            if not matched and all_facts:
                target_emb = embed([target])[0]
                for f in all_facts:
                    f_emb = f.get("embedding")
                    if f_emb is not None and len(f_emb) > 0:
                        sim = float(cosine_similarity(target_emb, f_emb))
                        if sim >= 0.7:
                            matched.append(f)

            if matched:
                for f in matched:
                    self.db.delete_fact(f["id"])
                names = "; ".join(f"'{f['text']}'" for f in matched)
                return f"I've forgotten: {names}."
            return f"I couldn't find any memory about '{target}' to forget."

        # Check rule capture
        rule_result = self.check_rule_capture(clean_text)
        if rule_result is not None:
            confirmation, _ = rule_result
            return confirmation

        return None

    # ---------------- Session Summarization ----------------

    def summarize_session(
        self,
        llm_fn: Callable[[str], str],
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Summarize conversation session when idle 5+ minutes, extracting episode and key facts.

        Called ONLY when idle 5+ minutes after a conversation. One LLM call returns JSON
        {summary (max 3 sentences), facts (max 5)}; stored in SQLite. Never called per turn.
        """
        now = time.time()
        if not force:
            idle_duration = now - self.last_interaction_ts
            if idle_duration < self.idle_threshold_s:
                return None
            if self._session_summarized or self._unsummarized_turns == 0:
                return None

        turns = list(self.recent_turns)
        if not turns:
            return None

        # Build prompt for single LLM call
        transcript_lines = [f"{turn['role'].capitalize()}: {turn['text']}" for turn in turns]
        transcript = "\n".join(transcript_lines)

        prompt = (
            "You are a memory consolidation assistant for Anya.\n"
            "Analyze the conversation below and extract:\n"
            "1. A concise episodic summary of the conversation (maximum 3 sentences).\n"
            "2. Key facts or user preferences learned during this session (maximum 5 facts).\n\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            "{\n"
            '  "summary": "...",\n'
            '  "facts": ["..."]\n'
            "}\n\n"
            f"Conversation:\n{transcript}"
        )

        try:
            raw_response = llm_fn(prompt)
            json_str = _extract_json_block(raw_response)
            parsed = json.loads(json_str)

            summary = str(parsed.get("summary", "")).strip()
            # Enforce max 3 sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
            if len(sentences) > 3:
                summary = " ".join(sentences[:3])

            raw_facts = parsed.get("facts", [])
            facts: List[str] = []
            if isinstance(raw_facts, list):
                for item in raw_facts[:5]:
                    item_str = str(item).strip()
                    if item_str:
                        facts.append(item_str)

            # Store episode summary
            ep_emb = embed([summary])[0] if summary else None
            episode_id = self.db.add_episode(
                summary=summary,
                mood="neutral",
                ts=now,
                embedding=ep_emb,
            )

            # Store extracted facts with similarity check
            stored_fact_ids: List[int] = []
            for fact_text in facts:
                fid = self.add_fact(fact_text, kind="user", source="session_summary")
                if fid is not None:
                    stored_fact_ids.append(fid)

            self._session_summarized = True
            self._unsummarized_turns = 0

            return {
                "episode_id": episode_id,
                "summary": summary,
                "facts": facts,
                "stored_fact_ids": stored_fact_ids,
            }

        except Exception as exc:
            logger.exception("Failed to summarize session: %s", exc)
            return None

    # ---------------- Context Generation ----------------

    def context_for_prompt(self, user_text: str) -> str:
        """Construct context for prompt: at most 5 facts + 1 episode, strictly under 150 tokens."""
        clean_text = user_text.strip()
        if not clean_text:
            return ""

        # Fetch top facts (max 5)
        facts = self.search(clean_text, k=self.prompt_max_facts)

        # Fetch top episode (max 1)
        all_episodes = self.db.get_all_episodes()
        selected_episode: Optional[Dict[str, Any]] = None

        if all_episodes:
            u_emb = embed([clean_text])[0]
            scored_episodes: List[Tuple[float, Dict[str, Any]]] = []
            for ep in all_episodes:
                ep_emb = ep.get("embedding")
                if ep_emb is not None and len(ep_emb) > 0:
                    sim = float(cosine_similarity(u_emb, ep_emb))
                    scored_episodes.append((sim, ep))

            if scored_episodes:
                scored_episodes.sort(key=lambda x: x[0], reverse=True)
                selected_episode = scored_episodes[0][1]
            else:
                selected_episode = all_episodes[0]

        # Assemble and enforce token limit under 150 tokens
        def format_context(f_list: List[Dict[str, Any]], ep: Optional[Dict[str, Any]]) -> str:
            sections: List[str] = []
            if f_list:
                fact_lines = [f"- {f['text']}" for f in f_list]
                sections.append("[Relevant Memory]\n" + "\n".join(fact_lines))
            if ep and ep.get("summary"):
                sections.append(f"[Past Context]\n{ep['summary']}")
            return "\n\n".join(sections)

        active_facts = list(facts)
        active_ep = selected_episode
        context_str = format_context(active_facts, active_ep)

        # Prune if estimated tokens >= prompt_max_tokens (150)
        while estimate_tokens(context_str) >= self.prompt_max_tokens and (active_facts or active_ep):
            if active_ep is not None:
                # Try truncating episode first
                ep_words = active_ep.get("summary", "").split()
                if len(ep_words) > 10:
                    active_ep = dict(active_ep)
                    active_ep["summary"] = " ".join(ep_words[: len(ep_words) // 2]) + "..."
                else:
                    active_ep = None
            elif active_facts:
                # Drop least similar fact
                active_facts.pop()

            context_str = format_context(active_facts, active_ep)

        return context_str


_MEMORY_INSTANCE: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Return default singleton MemoryManager instance."""
    global _MEMORY_INSTANCE
    if _MEMORY_INSTANCE is None:
        _MEMORY_INSTANCE = MemoryManager()
    return _MEMORY_INSTANCE
