"""
Shared utilities for the Urdu Speech AI pipeline.

Provides configuration loading, logging setup, path resolution,
and common helper functions used across all pipeline scripts.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

import yaml
from dotenv import load_dotenv


# =============================================================================
# Project Root Detection
# =============================================================================

def get_project_root() -> Path:
    """Walk up from this file until we find the config/ directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not find project root (directory containing 'config/'). "
        "Make sure you run scripts from within the project tree."
    )


PROJECT_ROOT = get_project_root()


# =============================================================================
# Configuration Loading
# =============================================================================

def load_settings() -> dict:
    """Load and return the global settings from config/settings.yaml."""
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    with open(settings_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_emotions() -> list[dict]:
    """Load the emotion taxonomy from config/emotions.yaml."""
    emotions_path = PROJECT_ROOT / "config" / "emotions.yaml"
    if not emotions_path.exists():
        raise FileNotFoundError(f"Emotions file not found: {emotions_path}")
    with open(emotions_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("emotions", [])


def load_channels() -> list[dict]:
    """Load the poet/channel list from config/channels.yaml."""
    channels_path = PROJECT_ROOT / "config" / "channels.yaml"
    if not channels_path.exists():
        raise FileNotFoundError(f"Channels file not found: {channels_path}")
    with open(channels_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("poets", [])


def get_emotion_ids() -> list[str]:
    """Return a flat list of valid emotion IDs from the taxonomy."""
    return [e["id"] for e in load_emotions()]


# =============================================================================
# Environment
# =============================================================================

def load_env():
    """Load .env file from the project root."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logging.warning(
            "No .env file found at %s. "
            "Copy .env.example to .env and fill in your API keys.",
            env_path,
        )


def get_gemini_api_key() -> str:
    """Return the Gemini API key from the environment."""
    load_env()
    key = os.getenv("GEMINI_API_KEY", "")
    if not key or key == "your_gemini_api_key_here":
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file (see .env.example)."
        )
    return key


# =============================================================================
# Path Helpers
# =============================================================================

def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT and ensure it exists."""
    full = PROJECT_ROOT / relative_path
    full.mkdir(parents=True, exist_ok=True)
    return full


def get_data_paths(settings: dict | None = None) -> dict[str, Path]:
    """Return a dict of resolved data directory paths from settings."""
    if settings is None:
        settings = load_settings()
    paths_cfg = settings.get("paths", {})
    return {
        key: resolve_path(rel)
        for key, rel in paths_cfg.items()
    }


# =============================================================================
# Logging
# =============================================================================

def setup_logging(
    name: str = "urdu_speech_ai",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Configure and return a named logger with console + optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        log_path = PROJECT_ROOT / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# =============================================================================
# Manifest / Metadata Helpers
# =============================================================================

def load_manifest(manifest_path: Path) -> dict:
    """Load a JSON manifest file, returning an empty dict if missing."""
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest_path: Path, data: dict):
    """Atomically write a JSON manifest file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(manifest_path)


def make_segment_id(poet_name: str, video_id: str, segment_num: int) -> str:
    """
    Generate a deterministic segment ID.

    Example: rahat_indori_dQw4w9WgXcQ_007
    """
    safe_name = poet_name.lower().replace(" ", "_").replace("'", "")
    return f"{safe_name}_{video_id}_{segment_num:03d}"


def timestamp() -> str:
    """Return an ISO-8601 timestamp string."""
    return datetime.now().isoformat(timespec="seconds")
