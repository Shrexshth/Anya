PROJECT: Anya, a local, emotional, agentic voice companion for macOS (Apple Silicon, 16 GB RAM).
Stack: Python 3.11, PySide6 (QWebEngineView + QWebChannel), Ollama (qwen2.5:7b-instruct),
faster-whisper, sounddevice, Piper TTS, fastembed (ONNX), SQLite, PixiJS 6.5 + pixi-live2d-display
(Cubism 4), AppleScript via osascript.

CORE PRINCIPLE: use the LLM only when necessary.
- Tier 0: exact regex rules. Tier 1: fuzzy intent match (embeddings over data/intents.yaml).
  Tier 2: LLM. A request stops at the first tier that handles it confidently.
- The emotion engine, routines, proactive messages and routine confirmations are plain code and
  templates. They never call the LLM.
- Stream LLM text sentence by sentence into TTS. Unload models when idle. Never keep two big
  models loaded. Cap frame rates when idle.

RULES:
- Follow the folder structure in the prompt exactly. Do not create extra top-level files.
- All settings live in config.py (from .env). No hardcoded ports, paths, model names or thresholds.
- Never use shell=True. Never build shell or AppleScript text by string formatting. Pass values
  through argv (`on run argv`) or arg lists.
- LLM output is untrusted: validate every tool call with pydantic before acting.
- Only the main thread touches Qt widgets. Other threads use Qt signals (core/events.py).
- Every skill has a risk level (safe or confirm), a pydantic args model, and a log line.
- Use the logging module. Type hints and short docstrings. Small functions.
- Add pytest tests for every pure-logic module (router, emotion, appraisal, templates, safety).
- After each task, tell me exactly how to run and test what you built.
- Only modify files named in the current task.