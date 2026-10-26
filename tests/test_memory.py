"""Unit and integration tests for SQLite DB, fastembed embeddings, and MemoryManager."""

import json
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List
import unittest

import numpy as np

from core.db import Database, deserialize_embedding, serialize_embedding
from core.embeddings import (
    DEFAULT_CACHE_SIZE,
    EMBEDDING_DIM,
    clear_cache,
    cosine_similarity,
    embed,
    get_cache_size,
)
from core.memory import MemoryManager, estimate_tokens
from core.resources import get_resource_manager


class TestDatabase(unittest.TestCase):
    """Test SQLite schema, migrations, WAL mode, and CRUD queries."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test_anya.db"
        self.db = Database(self.db_path)

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_schema_migrations_and_wal(self) -> None:
        """Verify startup migrations create all required tables and WAL mode is set."""
        with self.db.connection() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode;")
            journal_mode = cur.fetchone()[0]
            self.assertEqual(journal_mode.lower(), "wal")

            # Check all required tables exist
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cur.fetchall()}
            expected_tables = {
                "facts",
                "episodes",
                "mood_log",
                "actions",
                "routines",
                "settings",
                "schema_migrations",
            }
            self.assertTrue(expected_tables.issubset(tables))

    def test_facts_crud_and_embeddings(self) -> None:
        """Verify insert, get, update, delete, and embedding serialization for facts."""
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        fact_id = self.db.add_fact(
            text="User likes dark mode",
            kind="preference",
            source="conversation",
            embedding=emb,
        )
        self.assertIsInstance(fact_id, int)

        fact = self.db.get_fact(fact_id)
        self.assertIsNotNone(fact)
        assert fact is not None
        self.assertEqual(fact["text"], "User likes dark mode")
        self.assertEqual(fact["kind"], "preference")
        self.assertEqual(fact["source"], "conversation")
        self.assertTrue(np.allclose(fact["embedding"], emb))

        # Update last_used
        new_ts = time.time() + 100.0
        self.db.update_fact_last_used(fact_id, new_ts)
        updated_fact = self.db.get_fact(fact_id)
        assert updated_fact is not None
        self.assertAlmostEqual(updated_fact["last_used"], new_ts, places=1)

        # Delete by text pattern
        deleted_count = self.db.delete_facts_by_text("dark mode")
        self.assertEqual(deleted_count, 1)
        self.assertIsNone(self.db.get_fact(fact_id))

    def test_episodes_crud(self) -> None:
        """Verify episode storage and retrieval."""
        emb = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        ep_id = self.db.add_episode(
            summary="User worked on Python code.",
            mood="happy",
            embedding=emb,
        )
        self.assertIsInstance(ep_id, int)

        episodes = self.db.get_episodes(limit=5)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["summary"], "User worked on Python code.")
        self.assertEqual(episodes[0]["mood"], "happy")
        self.assertTrue(np.allclose(episodes[0]["embedding"], emb))

    def test_mood_and_actions_log(self) -> None:
        """Verify mood logs and executed actions logs."""
        self.db.log_mood({"valence": 0.8, "arousal": 0.4})
        logs = self.db.get_mood_logs(limit=5)
        self.assertEqual(len(logs), 1)
        self.assertIn("valence", logs[0]["mood_json"])

        self.db.log_action("spotify_play", {"track": "Song A"}, {"status": "ok"})
        actions = self.db.get_actions(limit=5)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["skill"], "spotify_play")

    def test_routines_and_settings(self) -> None:
        """Verify routines upsert/retrieval and key-value settings."""
        # Routines
        routine_steps = [{"action": "greet"}, {"action": "check_calendar"}]
        self.db.save_routine("morning", routine_steps)
        loaded_routine = self.db.get_routine("morning")
        self.assertEqual(loaded_routine, routine_steps)

        routines_map = self.db.list_routines()
        self.assertIn("morning", routines_map)

        self.assertTrue(self.db.delete_routine("morning"))
        self.assertIsNone(self.db.get_routine("morning"))

        # Settings
        self.db.set_setting("theme", "dracula")
        self.assertEqual(self.db.get_setting("theme"), "dracula")
        self.assertEqual(self.db.get_all_settings(), {"theme": "dracula"})
        self.assertTrue(self.db.delete_setting("theme"))
        self.assertIsNone(self.db.get_setting("theme"))


class TestEmbeddings(unittest.TestCase):
    """Test fastembed lazy loading, LRU caching, and cosine similarity."""

    def setUp(self) -> None:
        clear_cache()

    def tearDown(self) -> None:
        clear_cache()

    def test_embed_dimensions_and_types(self) -> None:
        """Verify embed returns float32 ndarray with expected dimensions."""
        texts = ["hello world", "Anya companion"]
        arr = embed(texts)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (2, EMBEDDING_DIM))
        self.assertEqual(arr.dtype, np.float32)

        # Single string input
        arr_single = embed("hello world")
        self.assertEqual(arr_single.shape, (1, EMBEDDING_DIM))

        # Empty list
        arr_empty = embed([])
        self.assertEqual(arr_empty.shape, (0, EMBEDDING_DIM))

    def test_lru_cache_hit(self) -> None:
        """Verify that caching avoids redundant model embedding passes."""
        self.assertEqual(get_cache_size(), 0)
        embed(["test phrase"])
        self.assertEqual(get_cache_size(), 1)

        # Second call should hit the cache
        arr_cached = embed(["test phrase"])
        self.assertEqual(arr_cached.shape, (1, EMBEDDING_DIM))
        self.assertEqual(get_cache_size(), 1)

        # Mixed batch (1 cached, 1 new)
        arr_mixed = embed(["test phrase", "new phrase"])
        self.assertEqual(arr_mixed.shape, (2, EMBEDDING_DIM))
        self.assertEqual(get_cache_size(), 2)

    def test_cosine_similarity(self) -> None:
        """Verify cosine similarity calculation for 1D and 2D arrays."""
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 1.0], dtype=np.float32)
        v4 = np.array([-1.0, 0.0], dtype=np.float32)

        self.assertAlmostEqual(float(cosine_similarity(v1, v2)), 1.0, places=5)
        self.assertAlmostEqual(float(cosine_similarity(v1, v3)), 0.0, places=5)
        self.assertAlmostEqual(float(cosine_similarity(v1, v4)), -1.0, places=5)

        # Matrix of target vectors
        mat = np.stack([v2, v3, v4], axis=0)
        sims = cosine_similarity(v1, mat)
        self.assertIsInstance(sims, np.ndarray)
        self.assertEqual(len(sims), 3)
        self.assertAlmostEqual(float(sims[0]), 1.0, places=5)
        self.assertAlmostEqual(float(sims[1]), 0.0, places=5)
        self.assertAlmostEqual(float(sims[2]), -1.0, places=5)


class TestMemoryManager(unittest.TestCase):
    """Test recent turns buffer, deduplicated facts, rule capture, session summary, and context."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test_memory.db"
        self.db = Database(self.db_path)
        self.memory = MemoryManager(
            db=self.db,
            history_turns=4,
            similarity_threshold=0.9,
            idle_threshold_s=300.0,
        )

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_recent_turns_deque_limit(self) -> None:
        """Verify turns buffer is capped at history_turns (4)."""
        self.assertEqual(len(self.memory.get_recent_turns()), 0)
        for i in range(6):
            self.memory.add_turn("user" if i % 2 == 0 else "assistant", f"Turn {i}")

        turns = self.memory.get_recent_turns()
        self.assertEqual(len(turns), 4)
        self.assertEqual(turns[0]["text"], "Turn 2")
        self.assertEqual(turns[-1]["text"], "Turn 5")

    def test_add_fact_and_cosine_deduplication(self) -> None:
        """Verify adding a fact works, and near-duplicate fact (>= 0.9 similarity) is skipped."""
        fid1 = self.memory.add_fact("I love drinking Earl Grey tea in the morning")
        self.assertIsNotNone(fid1)

        # Very similar phrase should be skipped
        fid2 = self.memory.add_fact("I love drinking Earl Grey tea in the morning.")
        self.assertIsNone(fid2)

        # Another highly similar phrase
        fid3 = self.memory.add_fact("I really love drinking Earl Grey tea in the morning")
        self.assertIsNone(fid3)

        # Distinct fact should be added
        fid4 = self.memory.add_fact("My favorite sport is tennis")
        self.assertIsNotNone(fid4)

        all_facts = self.db.get_all_facts()
        self.assertEqual(len(all_facts), 2)

    def test_search_facts_and_last_used_update(self) -> None:
        """Verify semantic search returns top matches and updates last_used."""
        self.memory.add_fact("User is studying quantum computing")
        self.memory.add_fact("User enjoys baking sourdough bread")
        self.memory.add_fact("User lives in Tokyo Japan")

        results = self.memory.search("baking bread and croissants", k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("sourdough", results[0]["text"].lower())
        self.assertIn("similarity", results[0])

    def test_rule_capture_regexes(self) -> None:
        """Verify deterministic rule capture without LLM."""
        # 1. remember that ...
        res1 = self.memory.check_rule_capture("Please remember that I work as an astrophysicist.")
        self.assertIsNotNone(res1)
        assert res1 is not None
        self.assertIn("remember that", res1[0])

        # 2. my name is ...
        res2 = self.memory.check_rule_capture("My name is Sophia!")
        self.assertIsNotNone(res2)
        assert res2 is not None
        self.assertIn("Sophia", res2[0])

        # 3. my birthday is ...
        res3 = self.memory.check_rule_capture("My birthday is December 25th")
        self.assertIsNotNone(res3)
        assert res3 is not None
        self.assertIn("December 25th", res3[0])

        # 4. I like / hate ...
        res4 = self.memory.check_rule_capture("I really like chocolate gelato")
        self.assertIsNotNone(res4)
        assert res4 is not None
        self.assertIn("chocolate gelato", res4[0])

        # 5. call me ...
        res5 = self.memory.check_rule_capture("You can call me Commander")
        self.assertIsNotNone(res5)
        assert res5 is not None
        self.assertIn("Commander", res5[0])

        # Non-rule query returns None
        self.assertIsNone(self.memory.check_rule_capture("How is the weather today?"))

    def test_direct_commands_what_remember_and_forget(self) -> None:
        """Verify 'what do you remember about me' and 'forget X' require no LLM."""
        self.memory.add_fact("User likes classical music")
        self.memory.add_fact("User's dog is named Luna")

        # Query memories
        rem_resp = self.memory.handle_direct_command("What do you remember about me?")
        self.assertIsNotNone(rem_resp)
        assert rem_resp is not None
        self.assertIn("classical music", rem_resp)
        self.assertIn("Luna", rem_resp)

        # Forget specific fact
        forget_resp = self.memory.handle_direct_command("Forget my dog Luna")
        self.assertIsNotNone(forget_resp)
        assert forget_resp is not None
        self.assertIn("forgotten", forget_resp.lower())

        remaining = self.db.get_all_facts()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["text"], "User likes classical music")

        # Forget everything
        forget_all_resp = self.memory.handle_direct_command("Forget everything")
        self.assertIsNotNone(forget_all_resp)
        self.assertEqual(len(self.db.get_all_facts()), 0)

    def test_summarize_session_idle_trigger_and_json(self) -> None:
        """Verify summarize_session only runs when idle 5+ min and executes one LLM call."""
        self.memory.add_turn("user", "Hello Anya, I am moving to Seattle next week.")
        self.memory.add_turn("assistant", "How exciting! Seattle is a great city.")

        # If not idle for 5 min (300s), returns None
        mock_llm_called = False

        def mock_llm(prompt: str) -> str:
            nonlocal mock_llm_called
            mock_llm_called = True
            return json.dumps({
                "summary": "User mentioned moving to Seattle next week. Anya congratulated them.",
                "facts": ["User is moving to Seattle"],
            })

        # Not idle yet
        res_not_idle = self.memory.summarize_session(mock_llm, force=False)
        self.assertIsNone(res_not_idle)
        self.assertFalse(mock_llm_called)

        # Force idle summarization (or simulated elapsed time >= 300s)
        res_idle = self.memory.summarize_session(mock_llm, force=True)
        self.assertIsNotNone(res_idle)
        self.assertTrue(mock_llm_called)
        assert res_idle is not None
        self.assertIn("Seattle", res_idle["summary"])
        self.assertEqual(res_idle["facts"], ["User is moving to Seattle"])

        # Check episode and fact stored in DB
        episodes = self.db.get_all_episodes()
        self.assertEqual(len(episodes), 1)
        self.assertIn("Seattle", episodes[0]["summary"])

        facts = self.db.get_all_facts()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["text"], "User is moving to Seattle")

        # Summarizing again without new turns should return None
        res_repeat = self.memory.summarize_session(mock_llm, force=False)
        self.assertIsNone(res_repeat)

    def test_context_for_prompt_token_and_count_limits(self) -> None:
        """Verify context_for_prompt limits: <= 5 facts, <= 1 episode, strictly < 150 tokens."""
        # Insert 7 facts
        for i in range(7):
            self.memory.add_fact(f"User preference #{i}: interest in topic {i}")

        # Insert 2 episodes
        self.db.add_episode(
            summary="User and Anya had a discussion about general preferences.",
            embedding=embed("discussion about preferences")[0],
        )
        self.db.add_episode(
            summary="User asked about weather in Paris.",
            embedding=embed("weather in Paris")[0],
        )

        context = self.memory.context_for_prompt("What are my preferences?")
        self.assertIsInstance(context, str)
        self.assertIn("[Relevant Memory]", context)

        # Token count check
        tokens = estimate_tokens(context)
        self.assertLess(tokens, 150)

        # Count facts in context: should be at most 5
        fact_bullet_count = context.count("\n- ") + (1 if context.startswith("- ") else 0)
        self.assertLessEqual(fact_bullet_count, 5)


if __name__ == "__main__":
    unittest.main()
