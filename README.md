# Anya

A local, emotional, agentic voice companion for macOS (Apple Silicon).

## Core Principles

- **LLM as Last Resort**:
  - **Tier 0**: Exact regex matching.
  - **Tier 1**: Fuzzy intent semantic matching via local embeddings over `data/intents.yaml`.
  - **Tier 2**: Ollama local LLM stream + Pydantic validated tool dispatch.
- **Pure-Code Emotion & Routines**:
  - Emotion dynamics, routines, proactive scheduling, and canned template responses are deterministic code and never invoke the LLM.
- **Resource Discipline**:
  - Lazy loading with automatic idle timeout unloading.
  - RAM monitoring via `psutil`.
  - Sentence-by-sentence streaming to Piper TTS.

## Tech Stack

- **Runtime**: Python 3.11, PySide6 (`QWebEngineView` + `QWebChannel`)
- **Intelligence**: Ollama (`qwen2.5:7b-instruct`, `qwen2.5vl:3b`)
- **Speech**: `faster-whisper`, `sounddevice`, Piper TTS
- **Embeddings & Vector**: `fastembed` (ONNX)
- **UI / Avatar**: PixiJS 6.5 + `pixi-live2d-display` (Live2D Cubism 4)
- **Storage**: SQLite (`data/anya.db`)
- **macOS Automation**: Safe AppleScript execution via `osascript` (argv only)

## Quick Start

1. Clone and navigate to the project directory:
   ```bash
   cd Anya
   ```

2. Copy `.env.example` to `.env` and review your settings:
   ```bash
   cp .env.example .env
   ```

3. Launch Anya via startup script:
   ```bash
   ./run.sh
   ```

## Running Tests

Execute pytest suite:
```bash
pytest
```
