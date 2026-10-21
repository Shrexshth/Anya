"""Configuration and environment settings loader for Anya."""

from dataclasses import dataclass, field
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from typing import List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path: Optional[Path | str] = None, override: bool = False) -> bool:  # type: ignore[misc]
        """Fallback lightweight .env loader when python-dotenv is not installed."""
        target = Path(dotenv_path) if dotenv_path else BASE_DIR / ".env"
        if not target.exists():
            return False
        try:
            with open(target, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if override or key not in os.environ:
                        os.environ[key] = val
            return True
        except Exception:
            return False

BASE_DIR = Path(__file__).resolve().parent


def _parse_bool(val: Optional[str], default: bool) -> bool:
    """Parse a boolean value from environment string."""
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(val: Optional[str], default: int) -> int:
    """Parse an integer value from environment string."""
    if val is None or not val.strip():
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _parse_list(val: Optional[str], default: List[str]) -> List[str]:
    """Parse a list of strings from JSON or comma-separated string."""
    if val is None or not val.strip():
        return default
    val = val.strip()
    if val.startswith("[") and val.endswith("]"):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in val.split(",") if item.strip()]


@dataclass
class Settings:
    """Central application configuration for Anya loaded from environment."""

    # Models & Voice
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct"
    OLLAMA_VISION_MODEL: str = "qwen2.5vl:3b"
    WHISPER_MODEL: str = "base.en"
    WAKE_MODEL: str = "tiny.en"
    WAKE_PHRASES: List[str] = field(default_factory=lambda: ["hey anya", "anya"])
    MODEL_PATH: str = "assets/model"
    PIPER_VOICE: str = "assets/voices/en_US-lessac-medium.onnx"

    # Conversational & Audio Settings
    CONVERSATION_WINDOW_S: int = 20
    BARGE_IN: bool = False
    PUSH_TO_TALK_KEY: str = "space"
    HISTORY_TURNS: int = 6
    NUM_CTX: int = 4096
    LLM_IDLE_UNLOAD_S: int = 300

    # UI & Performance Settings
    SERVER_PORT: int = 8000
    FPS_ACTIVE: int = 30
    FPS_IDLE: int = 10

    # Window geometry (logical pixels; capped at 90 % of available screen at runtime)
    WINDOW_W: int = 500
    WINDOW_H: int = 700

    # Live2D model fit (passed to JS via /api/config)
    MODEL_SCALE: float = 0.9      # fraction of the canvas to fill
    MODEL_OFFSET_Y: int = 0       # vertical centre offset in CSS pixels (positive = down)

    # Global hotkey to re-centre the window (pynput format)
    CENTER_HOTKEY: str = "<ctrl>+<alt>+a"

    # macOS: show on every Space / above fullscreen apps (requires pyobjc)
    SHOW_ON_ALL_SPACES: bool = True

    # Runtime Modes
    DRY_RUN: bool = False
    DEBUG: bool = False

    # Proactive & Focus Policies
    QUIET_HOURS: str = "23:00-07:00"
    MAX_PROACTIVE_PER_HOUR: int = 3
    BLOCKED_FOCUS_APPS: List[str] = field(default_factory=lambda: ["Discord"])

    # Optional External Integrations
    TELEGRAM_TOKEN: Optional[str] = None
    TELEGRAM_ALLOWED_ID: Optional[str] = None
    SPOTIFY_CLIENT_ID: Optional[str] = None
    SPOTIFY_CLIENT_SECRET: Optional[str] = None

    # Calculated directory paths
    BASE_DIR: Path = field(default_factory=lambda: BASE_DIR)
    LOGS_DIR: Path = field(default_factory=lambda: BASE_DIR / "logs")
    DATA_DIR: Path = field(default_factory=lambda: BASE_DIR / "data")
    ASSETS_DIR: Path = field(default_factory=lambda: BASE_DIR / "assets")
    DB_PATH: Path = field(default_factory=lambda: BASE_DIR / "data" / "anya.db")


def load_settings(env_file: Optional[Path | str] = None) -> Settings:
    """Load settings from .env file and environment variables."""
    dotenv_target = Path(env_file) if env_file else BASE_DIR / ".env"
    if dotenv_target.exists():
        load_dotenv(dotenv_path=dotenv_target, override=False)
    else:
        load_dotenv(override=False)

    return Settings(
        OLLAMA_MODEL=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        OLLAMA_VISION_MODEL=os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:3b"),
        WHISPER_MODEL=os.getenv("WHISPER_MODEL", "base.en"),
        WAKE_MODEL=os.getenv("WAKE_MODEL", "tiny.en"),
        WAKE_PHRASES=_parse_list(os.getenv("WAKE_PHRASES"), ["hey anya", "anya"]),
        MODEL_PATH=os.getenv("MODEL_PATH", "assets/model"),
        PIPER_VOICE=os.getenv("PIPER_VOICE", "assets/voices/en_US-lessac-medium.onnx"),
        CONVERSATION_WINDOW_S=_parse_int(os.getenv("CONVERSATION_WINDOW_S"), 20),
        BARGE_IN=_parse_bool(os.getenv("BARGE_IN"), False),
        PUSH_TO_TALK_KEY=os.getenv("PUSH_TO_TALK_KEY", "space"),
        HISTORY_TURNS=_parse_int(os.getenv("HISTORY_TURNS"), 6),
        NUM_CTX=_parse_int(os.getenv("NUM_CTX"), 4096),
        LLM_IDLE_UNLOAD_S=_parse_int(os.getenv("LLM_IDLE_UNLOAD_S"), 300),
        SERVER_PORT=_parse_int(os.getenv("SERVER_PORT"), 8000),
        FPS_ACTIVE=_parse_int(os.getenv("FPS_ACTIVE"), 30),
        FPS_IDLE=_parse_int(os.getenv("FPS_IDLE"), 10),
        WINDOW_W=_parse_int(os.getenv("WINDOW_W"), 500),
        WINDOW_H=_parse_int(os.getenv("WINDOW_H"), 700),
        MODEL_SCALE=float(os.getenv("MODEL_SCALE", "0.9")),
        MODEL_OFFSET_Y=_parse_int(os.getenv("MODEL_OFFSET_Y"), 0),
        CENTER_HOTKEY=os.getenv("CENTER_HOTKEY", "<ctrl>+<alt>+a"),
        SHOW_ON_ALL_SPACES=_parse_bool(os.getenv("SHOW_ON_ALL_SPACES"), True),
        DRY_RUN=_parse_bool(os.getenv("DRY_RUN"), False),
        DEBUG=_parse_bool(os.getenv("DEBUG"), False),
        QUIET_HOURS=os.getenv("QUIET_HOURS", "23:00-07:00"),
        MAX_PROACTIVE_PER_HOUR=_parse_int(os.getenv("MAX_PROACTIVE_PER_HOUR"), 3),
        BLOCKED_FOCUS_APPS=_parse_list(os.getenv("BLOCKED_FOCUS_APPS"), ["Discord"]),
        TELEGRAM_TOKEN=os.getenv("TELEGRAM_TOKEN") or None,
        TELEGRAM_ALLOWED_ID=os.getenv("TELEGRAM_ALLOWED_ID") or None,
        SPOTIFY_CLIENT_ID=os.getenv("SPOTIFY_CLIENT_ID") or None,
        SPOTIFY_CLIENT_SECRET=os.getenv("SPOTIFY_CLIENT_SECRET") or None,
    )


_SETTINGS_INSTANCE: Optional[Settings] = None


def get_settings() -> Settings:
    """Return cached global Settings instance, initializing on first access."""
    global _SETTINGS_INSTANCE
    if _SETTINGS_INSTANCE is None:
        _SETTINGS_INSTANCE = load_settings()
    return _SETTINGS_INSTANCE


def setup_logging(
    settings: Optional[Settings] = None, log_file: Optional[Path | str] = None
) -> logging.Logger:
    """Configure unified logging to console and logs/anya.log."""
    active_settings = settings or get_settings()
    log_path = Path(log_file) if log_file else active_settings.LOGS_DIR / "anya.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if active_settings.DEBUG else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root_logger.handlers
    )
    has_file = any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == str(log_path.resolve())
        for h in root_logger.handlers
    )

    if not has_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if not has_file:
        file_handler = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger = logging.getLogger("anya")
    logger.debug("Logging initialized at %s level (file: %s)", logging.getLevelName(level), log_path)
    return logger
