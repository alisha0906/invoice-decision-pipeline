"""Centralized configuration.

Loads environment variables from a local .env (via python-dotenv) and exposes
the vision-reader settings plus tunable threshold constants used across the
pipeline. No secrets are hard-coded.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from PS-1/.env if present. Absent file is fine (defaults apply).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# --- Extraction mode -------------------------------------------------------
# 'vision_only' (default): EVERY invoice (PDF or image) is rendered to page
# images and read by a vision LLM. If the vision reader is unavailable or
# fails, the run degrades gracefully to REVIEW (never crashes).
EXTRACTION_MODE = os.getenv("EXTRACTION_MODE", "vision_only")

# --- Vision reader selection -----------------------------------------------
# Which vision reader reads the invoice images:
#   'ollama' (default) - a local vision model. No API key, no rate limits,
#                        works fully offline.
#   'gemini'           - Google Gemini (free AI Studio tier). Hosted; rate-limited.
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "ollama")

# --- Ollama (local vision - default) ---------------------------------------
# Pull a vision model first, e.g.:
#   ollama pull llava-phi3        (~2.9 GB, good balance)
#   ollama pull moondream         (~1.7 GB, smallest)
#   ollama pull llama3.2-vision   (~7.9 GB, best accuracy)
# The model name must contain a vision hint (llava/vision/moondream/...).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava-phi3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# --- Google Gemini (optional hosted vision reader) -------------------------
# Free API key from Google AI Studio (https://aistudio.google.com), NOT Vertex.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Empty/absent = auto-detect the newest usable flash model from the live
# /models list (Google gates some older models to existing users only).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")
GEMINI_HOST = os.getenv("GEMINI_HOST", "https://generativelanguage.googleapis.com")
# Seconds to pause between vision calls in batch eval (free-tier rate limit).
GEMINI_RATE_DELAY = _get_float("GEMINI_RATE_DELAY", 4.5)

# --- Threshold constants (tunable) -----------------------------------------
# Relative tolerance for invoice-vs-PO amount matching (2%).
TOLERANCE_PCT = _get_float("TOLERANCE_PCT", 0.02)
# Absolute tolerance floor (currency units) for small amounts.
TOLERANCE_ABS = _get_float("TOLERANCE_ABS", 1.00)
# Overall extraction confidence below which a run routes to REVIEW.
LOW_CONFIDENCE_THRESHOLD = _get_float("LOW_CONFIDENCE_THRESHOLD", 0.60)
# Band around the tolerance edge that is treated as borderline -> REVIEW.
TOLERANCE_BORDERLINE_PCT = _get_float("TOLERANCE_BORDERLINE_PCT", 0.01)

# --- Persistence ------------------------------------------------------------
# SQLite database file. Runtime-generated and gitignored.
DB_PATH = os.getenv("DB_PATH", str(_PROJECT_ROOT / "runs.db"))
