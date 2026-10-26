"""SQLite schema definitions, connection lifecycle, migrations, and queries for Anya."""

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from config import get_settings

logger = logging.getLogger("anya.core.db")

MIGRATIONS: List[Tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'user',
            source TEXT NOT NULL DEFAULT 'conversation',
            created REAL NOT NULL,
            last_used REAL NOT NULL,
            embedding BLOB
        );

        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT NOT NULL,
            mood TEXT DEFAULT 'neutral',
            ts REAL NOT NULL,
            embedding BLOB
        );

        CREATE TABLE IF NOT EXISTS mood_log (
            ts REAL NOT NULL,
            mood_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS actions (
            ts REAL NOT NULL,
            skill TEXT NOT NULL,
            args TEXT,
            result TEXT
        );

        CREATE TABLE IF NOT EXISTS routines (
            name TEXT PRIMARY KEY,
            steps_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_facts_last_used ON facts(last_used);
        CREATE INDEX IF NOT EXISTS idx_facts_kind ON facts(kind);
        CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);
        CREATE INDEX IF NOT EXISTS idx_mood_log_ts ON mood_log(ts);
        CREATE INDEX IF NOT EXISTS idx_actions_ts ON actions(ts);
        """,
    )
]


def configure_connection(conn: sqlite3.Connection) -> None:
    """Apply performance pragmas including WAL mode and row factory."""
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.DatabaseError:
        pass
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row


def run_migrations(conn: sqlite3.Connection) -> None:
    """Execute pending schema migrations in an isolated transaction."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at REAL NOT NULL
        );
        """
    )
    cur = conn.cursor()
    cur.execute("SELECT version FROM schema_migrations")
    applied = {row[0] for row in cur.fetchall()}

    for version, sql in MIGRATIONS:
        if version not in applied:
            logger.info("Applying database migration v%d...", version)
            cur.executescript(sql)
            cur.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, time.time()),
            )
            conn.commit()
            logger.info("Database migration v%d applied successfully.", version)


def serialize_embedding(embedding: Optional[Union[np.ndarray, List[float]]]) -> Optional[bytes]:
    """Serialize numpy vector or float sequence to raw float32 bytes for BLOB storage."""
    if embedding is None:
        return None
    arr = np.asarray(embedding, dtype=np.float32)
    return arr.tobytes()


def deserialize_embedding(blob: Optional[bytes]) -> Optional[np.ndarray]:
    """Deserialize raw float32 BLOB bytes into a 1D numpy array."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


class Database:
    """Thread-safe SQLite database manager with WAL mode and migration support."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        if db_path is not None:
            self.db_path: Union[Path, str] = (
                Path(db_path) if str(db_path) != ":memory:" else ":memory:"
            )
        else:
            settings = get_settings()
            self.db_path = getattr(settings, "DB_PATH", Path("data/anya.db"))

        self._lock = threading.RLock()
        self._local = threading.local()
        self._is_memory = str(self.db_path) == ":memory:"
        self._memory_conn: Optional[sqlite3.Connection] = None

        if not self._is_memory and isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self.connection() as conn:
            run_migrations(conn)

    def get_connection(self) -> sqlite3.Connection:
        """Get or create an active connection."""
        if self._is_memory:
            with self._lock:
                if self._memory_conn is None:
                    conn = sqlite3.connect(":memory:", check_same_thread=False)
                    configure_connection(conn)
                    self._memory_conn = conn
                return self._memory_conn

        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            configure_connection(conn)
            self._local.conn = conn
        return conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a thread-safe connection with rollback on error."""
        conn = self.get_connection()
        with self._lock:
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise

    # ---------------- Facts CRUD ----------------

    def add_fact(
        self,
        text: str,
        kind: str = "user",
        source: str = "conversation",
        embedding: Optional[Union[np.ndarray, List[float]]] = None,
        created: Optional[float] = None,
        last_used: Optional[float] = None,
    ) -> int:
        """Insert a fact and return its ID."""
        now = time.time()
        c_time = created if created is not None else now
        u_time = last_used if last_used is not None else now
        blob = serialize_embedding(embedding)

        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO facts (text, kind, source, created, last_used, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (text, kind, source, c_time, u_time, blob),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_fact(self, fact_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single fact by ID."""
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, text, kind, source, created, last_used, embedding FROM facts WHERE id = ?",
                (fact_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "text": row["text"],
                "kind": row["kind"],
                "source": row["source"],
                "created": row["created"],
                "last_used": row["last_used"],
                "embedding": deserialize_embedding(row["embedding"]),
            }

    def get_all_facts(self) -> List[Dict[str, Any]]:
        """Retrieve all facts ordered by last_used descending."""
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, text, kind, source, created, last_used, embedding FROM facts ORDER BY last_used DESC"
            )
            rows = cur.fetchall()
            return [
                {
                    "id": row["id"],
                    "text": row["text"],
                    "kind": row["kind"],
                    "source": row["source"],
                    "created": row["created"],
                    "last_used": row["last_used"],
                    "embedding": deserialize_embedding(row["embedding"]),
                }
                for row in rows
            ]

    def update_fact_last_used(self, fact_id: int, last_used: Optional[float] = None) -> None:
        """Update last_used timestamp for a fact."""
        u_time = last_used if last_used is not None else time.time()
        with self.connection() as conn:
            conn.execute("UPDATE facts SET last_used = ? WHERE id = ?", (u_time, fact_id))
            conn.commit()

    def delete_fact(self, fact_id: int) -> bool:
        """Delete a fact by ID. Return True if deleted."""
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            conn.commit()
            return cur.rowcount > 0

    def delete_facts_by_text(self, text_pattern: str) -> int:
        """Delete facts containing text_pattern (case-insensitive). Return count deleted."""
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM facts WHERE LOWER(text) LIKE LOWER(?)",
                (f"%{text_pattern}%",),
            )
            conn.commit()
            return cur.rowcount

    def clear_facts(self) -> None:
        """Delete all facts."""
        with self.connection() as conn:
            conn.execute("DELETE FROM facts")
            conn.commit()

    # ---------------- Episodes CRUD ----------------

    def add_episode(
        self,
        summary: str,
        mood: str = "neutral",
        ts: Optional[float] = None,
        embedding: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> int:
        """Insert an episodic memory summary and return its ID."""
        event_ts = ts if ts is not None else time.time()
        blob = serialize_embedding(embedding)
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO episodes (summary, mood, ts, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (summary, mood, event_ts, blob),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_episodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent episodes ordered by ts descending."""
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, summary, mood, ts, embedding FROM episodes ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
            return [
                {
                    "id": row["id"],
                    "summary": row["summary"],
                    "mood": row["mood"],
                    "ts": row["ts"],
                    "embedding": deserialize_embedding(row["embedding"]),
                }
                for row in rows
            ]

    def get_all_episodes(self) -> List[Dict[str, Any]]:
        """Retrieve all episodes ordered by ts descending."""
        return self.get_episodes(limit=1000000)

    # ---------------- Mood Log CRUD ----------------

    def log_mood(
        self,
        mood_json: Union[str, Dict[str, Any]],
        ts: Optional[float] = None,
    ) -> None:
        """Record a timestamped mood state."""
        event_ts = ts if ts is not None else time.time()
        payload = mood_json if isinstance(mood_json, str) else json.dumps(mood_json)
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO mood_log (ts, mood_json) VALUES (?, ?)",
                (event_ts, payload),
            )
            conn.commit()

    def get_mood_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent mood log entries ordered by ts descending."""
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT ts, mood_json FROM mood_log ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
            return [{"ts": row["ts"], "mood_json": row["mood_json"]} for row in rows]

    # ---------------- Actions CRUD ----------------

    def log_action(
        self,
        skill: str,
        args: Any,
        result: Any,
        ts: Optional[float] = None,
    ) -> None:
        """Record an executed skill action."""
        event_ts = ts if ts is not None else time.time()
        args_str = args if isinstance(args, str) else json.dumps(args, default=str)
        result_str = result if isinstance(result, str) else json.dumps(result, default=str)
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO actions (ts, skill, args, result) VALUES (?, ?, ?, ?)",
                (event_ts, skill, args_str, result_str),
            )
            conn.commit()

    def get_actions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent actions ordered by ts descending."""
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT ts, skill, args, result FROM actions ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
            return [
                {
                    "ts": row["ts"],
                    "skill": row["skill"],
                    "args": row["args"],
                    "result": row["result"],
                }
                for row in rows
            ]

    # ---------------- Routines CRUD ----------------

    def save_routine(
        self,
        name: str,
        steps_json: Union[str, List[Any], Dict[str, Any]],
    ) -> None:
        """Upsert a routine definition by name."""
        payload = steps_json if isinstance(steps_json, str) else json.dumps(steps_json)
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO routines (name, steps_json)
                VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET steps_json = excluded.steps_json
                """,
                (name, payload),
            )
            conn.commit()

    def get_routine(self, name: str) -> Optional[Union[Dict[str, Any], List[Any], str]]:
        """Fetch a routine by name and parse JSON steps if applicable."""
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT steps_json FROM routines WHERE name = ?", (name,))
            row = cur.fetchone()
            if not row:
                return None
            steps_raw = row["steps_json"]
            try:
                return json.loads(steps_raw)
            except json.JSONDecodeError:
                return steps_raw

    def list_routines(self) -> Dict[str, Any]:
        """Fetch all routines as a name -> steps mapping."""
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name, steps_json FROM routines ORDER BY name ASC")
            results: Dict[str, Any] = {}
            for row in cur.fetchall():
                try:
                    results[row["name"]] = json.loads(row["steps_json"])
                except json.JSONDecodeError:
                    results[row["name"]] = row["steps_json"]
            return results

    def delete_routine(self, name: str) -> bool:
        """Delete a routine by name. Return True if deleted."""
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM routines WHERE name = ?", (name,))
            conn.commit()
            return cur.rowcount > 0

    # ---------------- Settings CRUD ----------------

    def set_setting(self, key: str, value: str) -> None:
        """Upsert a key-value setting."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, str(value)),
            )
            conn.commit()

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Fetch a setting value by key."""
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cur.fetchone()
            if not row:
                return default
            return str(row["value"])

    def get_all_settings(self) -> Dict[str, str]:
        """Fetch all settings as key -> value mapping."""
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT key, value FROM settings ORDER BY key ASC")
            return {row["key"]: row["value"] for row in cur.fetchall()}

    def delete_setting(self, key: str) -> bool:
        """Delete a setting by key. Return True if deleted."""
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM settings WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        """Close thread connection or memory connection."""
        with self._lock:
            if self._is_memory and self._memory_conn is not None:
                try:
                    self._memory_conn.close()
                except Exception:
                    pass
                self._memory_conn = None

            conn = getattr(self._local, "conn", None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
                self._local.conn = None


_DB_INSTANCE: Optional[Database] = None


def get_db(db_path: Optional[Union[str, Path]] = None) -> Database:
    """Return default singleton Database instance or custom instance if db_path specified."""
    global _DB_INSTANCE
    if db_path is not None:
        return Database(db_path)
    if _DB_INSTANCE is None:
        _DB_INSTANCE = Database()
    return _DB_INSTANCE
