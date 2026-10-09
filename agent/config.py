"""
Configuration management for the AI Code Investigation Agent.
Loads settings from environment variables and .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (parent of the agent/ package)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)

# ── LLM ──────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Optional thinking level: minimal | low | medium | high.  Empty = do not send a
# thinking config (the model's own default applies).  Unsupported values/models
# raise an error rather than being silently ignored.
GEMINI_THINKING_LEVEL: str = os.getenv("GEMINI_THINKING_LEVEL", "").strip().lower()

# ── Agent loop ────────────────────────────────────────────────────────────────
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "30"))
LLM_REQUEST_BUDGET: int = int(os.getenv("LLM_REQUEST_BUDGET", "15"))

# ── Repository scanning ───────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_BYTES", str(512 * 1024)))  # 512 KB
MAX_READ_LINES: int = int(os.getenv("MAX_READ_LINES", "300"))

# ── Runs / observability ──────────────────────────────────────────────────────
RUNS_DIR: Path = Path(os.getenv("RUNS_DIR", "runs"))

# ── LLM retry ────────────────────────────────────────────────────────────────
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "5"))
LLM_RETRY_INITIAL_WAIT: float = float(os.getenv("LLM_RETRY_INITIAL_WAIT", "2.0"))

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))


def validate() -> None:
    """Raise if required settings are missing."""
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example → .env and add your key."
        )
