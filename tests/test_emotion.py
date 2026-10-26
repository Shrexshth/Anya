"""Unit tests for emotion engine, decay, dominant emotion, appraisal, modes, VIP rule, and templates."""

from pathlib import Path
import tempfile
import time
import unittest

from core.appraisal import AppraisalEngine, get_appraisal_engine
from core.db import Database
from core.emotion import (
    MAX_ANGER_LIMIT,
    EmotionEngine,
    MoodVector,
    get_emotion_engine,
)
from core.persona import (
    MODE_CHILL,
    MODE_FRIEND_ROAST,
    MODE_PROFESSIONAL,
    MODE_ROMANTIC,
    MODE_SASSY,
    MODE_STUDY,
    PersonaManager,
    check_vip_presence,
    match_mode_command,
)
from core.templates import TemplateManager, is_guilt_tripping, pick


class TestMoodVectorAndDecay(unittest.TestCase):
    """Test MoodVector clamping, anger cap, and exponential decay."""

    def test_mood_clamping_and_anger_cap(self) -> None:
        """Verify dimensions clamp [0, 1] and anger is strictly capped at 0.85."""
        mv = MoodVector(
            happiness=1.5,
            anger=0.99,
            sadness=-0.5,
            affection=0.8,
            energy=2.0,
            playfulness=1.0,
            boredom=0.0,
            anxiety=-1.0,
        )
        mv.clamp()
        self.assertEqual(mv.happiness, 1.0)
        self.assertEqual(mv.anger, MAX_ANGER_LIMIT)  # strictly capped at 0.85
        self.assertEqual(mv.sadness, 0.0)
        self.assertEqual(mv.energy, 1.0)
        self.assertEqual(mv.anxiety, 0.0)

    def test_exponential_decay_towards_baseline(self) -> None:
        """Verify exponential half-life decay moves mood toward baseline."""
        tmp = tempfile.TemporaryDirectory()
        db_path = Path(tmp.name) / "test_decay.db"
        db = Database(db_path)

        engine = EmotionEngine(db=db, auto_restore=False)
        engine.mood.anger = 0.80

        # Anger half life is 300 seconds; target baseline for friend_roast is 0.10
        # After 300s, anger gap should halve: 0.10 + (0.80 - 0.10) * 0.5 = 0.45
        engine.apply_decay(300.0)
        self.assertAlmostEqual(engine.mood.anger, 0.45, places=2)

        # After another 300s, gap halves again: 0.10 + (0.45 - 0.10) * 0.5 = 0.275
        engine.apply_decay(300.0)
        self.assertAlmostEqual(engine.mood.anger, 0.275, places=2)

        db.close()
        tmp.cleanup()

    def test_save_and_restore_with_decay(self) -> None:
        """Verify mood saves to SQLite and restores with elapsed decay applied."""
        tmp = tempfile.TemporaryDirectory()
        db_path = Path(tmp.name) / "test_restore.db"
        db = Database(db_path)

        engine1 = EmotionEngine(db=db, auto_restore=False)
        engine1.mood.anger = 0.80
        engine1.save_state()

        # Simulate 300 seconds elapsed before engine2 starts up
        # Modify the timestamp of the saved log entry
        with db.connection() as conn:
            conn.execute("UPDATE mood_log SET ts = ts - 300.0")
            conn.commit()

        engine2 = EmotionEngine(db=db, auto_restore=True)
        # Anger should be restored and decayed from 0.80 towards 0.10 (halved gap ~ 0.45)
        self.assertAlmostEqual(engine2.mood.anger, 0.45, places=2)

        db.close()
        tmp.cleanup()


class TestDominantEmotionAndHysteresis(unittest.TestCase):
    """Test dominant emotion scoring, intensity, blend, and 4s hysteresis."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test_dom.db"
        self.db = Database(self.db_path)
        self.engine = EmotionEngine(db=self.db, auto_restore=False)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_dominant_emotion_scoring_and_blend(self) -> None:
        """Verify highest scored emotion becomes primary with secondary blend."""
        # High affection + high happiness -> romantic
        self.engine.mood.affection = 0.90
        self.engine.mood.happiness = 0.80
        self.engine.mood.anger = 0.0
        self.engine.mood.boredom = 0.0

        primary, intensity, blend = self.engine.recalculate_dominant(force=True)
        self.assertEqual(primary, "romantic")
        self.assertGreater(intensity, 0.5)
        self.assertIsInstance(blend, dict)

    def test_hysteresis_dwell_time(self) -> None:
        """Verify current primary holds for at least 4.0 seconds before switching."""
        self.engine.primary_emotion = "happy"
        self.engine.primary_start_ts = time.time()  # 0 seconds elapsed

        # Mood shifts drastically to angry
        self.engine.mood.anger = 0.85
        self.engine.mood.happiness = 0.0
        self.engine.mood.energy = 0.80

        # Should NOT switch immediately because 4s dwell time has not elapsed
        primary, _, _ = self.engine.recalculate_dominant(force=False)
        self.assertEqual(primary, "happy")

        # Simulate 4.5 seconds elapsed
        self.engine.primary_start_ts = time.time() - 4.5
        primary, _, _ = self.engine.recalculate_dominant(force=False)
        self.assertEqual(primary, "angry")

    def test_hysteresis_margin_requirement(self) -> None:
        """Verify new candidate must beat current score by the hysteresis margin."""
        self.engine.primary_emotion = "happy"
        self.engine.primary_start_ts = time.time() - 10.0  # Dwell time satisfied

        # Set scores close to each other
        self.engine.mood.happiness = 0.50
        self.engine.mood.energy = 0.52
        self.engine.mood.playfulness = 0.50

        # Candidate slightly leads but by less than margin (0.15)
        primary, _, _ = self.engine.recalculate_dominant(force=False)
        # Should stay happy because gap < margin
        self.assertEqual(primary, "happy")


class TestAppraisalEngine(unittest.TestCase):
    """Test sentiment lexicon, keywords, and discrete event mood deltas."""

    def setUp(self) -> None:
        self.appraisal = get_appraisal_engine()

    def test_vader_positive_and_compliments(self) -> None:
        """Verify compliments and positive sentiment increase happiness and affection."""
        deltas, meta = self.appraisal.appraise_user_text("You are so cute and smart! Great job!")
        self.assertGreater(deltas.get("happiness", 0), 0.10)
        self.assertGreater(deltas.get("affection", 0), 0.05)
        self.assertIn("compliments", meta["categories"])

    def test_vader_insults_increase_anger(self) -> None:
        """Verify insults increase anger and decrease happiness."""
        deltas, meta = self.appraisal.appraise_user_text("You are stupid and useless, you suck!")
        self.assertGreater(deltas.get("anger", 0), 0.20)
        self.assertLess(deltas.get("happiness", 0), 0.0)
        self.assertIn("insults", meta["categories"])

    def test_apology_cools_anger_faster(self) -> None:
        """Verify apologies decrease anger significantly (guardrail cooling)."""
        deltas, meta = self.appraisal.appraise_user_text("I am so sorry, please forgive me, my bad.")
        self.assertLess(deltas.get("anger", 0), -0.20)
        self.assertIn("apologies", meta["categories"])

    def test_discrete_events(self) -> None:
        """Verify system events produce expected deltas."""
        d_success = self.appraisal.appraise_event("task_success")
        self.assertGreater(d_success.get("happiness", 0), 0)
        self.assertLess(d_success.get("boredom", 0), 0)

        d_fail = self.appraisal.appraise_event("task_fail")
        self.assertGreater(d_fail.get("anxiety", 0), 0)

        d_return = self.appraisal.appraise_event("returned_after_absence")
        self.assertGreater(d_return.get("affection", 0), 0)
        self.assertGreater(d_return.get("happiness", 0), 0)


class TestPersonalityModesAndVIP(unittest.TestCase):
    """Test personality modes, switching rules, and VIP professional override."""

    def setUp(self) -> None:
        self.persona = PersonaManager()

    def test_voice_commands_mode_switching(self) -> None:
        """Verify voice command regexes detect mode changes."""
        self.assertEqual(match_mode_command("switch to romantic mode"), MODE_ROMANTIC)
        self.assertEqual(match_mode_command("let's study mode"), MODE_STUDY)
        self.assertEqual(match_mode_command("sassy mode please"), MODE_SASSY)
        self.assertEqual(match_mode_command("chill mode"), MODE_CHILL)
        self.assertEqual(match_mode_command("professional mode"), MODE_PROFESSIONAL)

    def test_friend_mode_leaves_romantic(self) -> None:
        """Verify 'friend mode' always leaves romantic mode."""
        self.persona.set_mode(MODE_ROMANTIC)
        self.assertEqual(self.persona.current_mode, MODE_ROMANTIC)

        # Trigger friend mode
        mode, is_vip, _ = self.persona.evaluate_turn_mode("switch to friend mode")
        self.assertEqual(mode, MODE_FRIEND_ROAST)
        self.assertEqual(self.persona.current_mode, MODE_FRIEND_ROAST)
        self.assertFalse(is_vip)

    def test_style_hints_for_llm(self) -> None:
        """Verify every mode has a distinct LLM style hint."""
        for m in [MODE_FRIEND_ROAST, MODE_ROMANTIC, MODE_STUDY, MODE_SASSY, MODE_CHILL, MODE_PROFESSIONAL]:
            hint = self.persona.get_style_hint(m)
            self.assertIsInstance(hint, str)
            self.assertGreater(len(hint), 15)

    def test_vip_rule_in_code(self) -> None:
        """Verify presence of VIP name forces professional mode for the turn and clamps anger/playfulness to 0."""
        text = "Hello Professor, can you check this report for me?"
        mode, is_vip, vip_name = self.persona.evaluate_turn_mode(text)
        self.assertTrue(is_vip)
        self.assertEqual(vip_name, "Professor")
        self.assertEqual(mode, MODE_PROFESSIONAL)

        # Verify emotion engine VIP clamp
        tmp = tempfile.TemporaryDirectory()
        db = Database(Path(tmp.name) / "test_vip.db")
        engine = EmotionEngine(db=db, auto_restore=False)
        engine.mood.anger = 0.60
        engine.mood.playfulness = 0.70
        engine.apply_vip_override()

        self.assertEqual(engine.mood.anger, 0.0)
        self.assertEqual(engine.mood.playfulness, 0.0)

        db.close()
        tmp.cleanup()

    def test_guardrails_safe_skills_never_blocked(self) -> None:
        """Verify guardrail: no mood ever blocks a safe skill or kill switch."""
        tmp = tempfile.TemporaryDirectory()
        db = Database(Path(tmp.name) / "test_guard.db")
        engine = EmotionEngine(db=db, auto_restore=False)

        # Even at max anger
        engine.mood.anger = MAX_ANGER_LIMIT
        self.assertTrue(engine.can_execute_skill("open_app", is_safe=True))
        self.assertTrue(engine.can_execute_skill("kill_switch", is_safe=True))

        db.close()
        tmp.cleanup()


class TestTemplates(unittest.TestCase):
    """Test canned responses, 3-pick non-repeat, and guilt-tripping guardrail."""

    def setUp(self) -> None:
        self.tm = TemplateManager()

    def test_pick_common_kinds_and_slots(self) -> None:
        """Verify template selection and slot formatting for common kinds."""
        kinds = ["ack_done", "ack_fail", "confirm_ask", "greet", "goodnight", "miss_you", "unknown", "brain_offline"]
        for k in kinds:
            resp = self.tm.pick(k, emotion="happy", mode="friend_roast")
            self.assertIsInstance(resp, str)
            self.assertGreater(len(resp), 3)

    def test_at_least_4_variants_per_emotion(self) -> None:
        """Verify at least 4 variants per emotion for common kinds in templates.yaml."""
        common_kinds = ["ack_done", "ack_fail", "confirm_ask", "greet", "goodnight", "miss_you"]
        emotions = ["happy", "excited", "smug", "romantic", "shy", "sad", "angry", "neutral"]

        for k in common_kinds:
            kind_dict = self.tm.templates.get(k, {})
            for emo in emotions:
                variants = kind_dict.get(emo, [])
                self.assertGreaterEqual(
                    len(variants),
                    4,
                    f"Kind '{k}' emotion '{emo}' has only {len(variants)} variants (minimum 4 required)",
                )

    def test_never_repeat_last_3_picks(self) -> None:
        """Verify pick never repeats the immediate last 3 picks when choices exist."""
        history = []
        for _ in range(7):
            chosen = self.tm.pick("greet", emotion="happy", mode="friend_roast")
            if len(history) >= 1:
                # Can't be any of the last 3 items in history
                forbidden = set(history[-3:])
                self.assertNotIn(chosen, forbidden)
            history.append(chosen)

    def test_no_guilt_tripping_in_templates(self) -> None:
        """Verify sadness and affection templates never include guilt-tripping text."""
        for kind in ["sad", "romantic", "miss_you"]:
            kind_group = self.tm.templates.get(kind, {})
            if isinstance(kind_group, dict):
                for emo, texts in kind_group.items():
                    if isinstance(texts, list):
                        for t in texts:
                            self.assertFalse(
                                is_guilt_tripping(t),
                                f"Template '{t}' in {kind}:{emo} violates guilt-tripping guardrail",
                            )


if __name__ == "__main__":
    unittest.main()
