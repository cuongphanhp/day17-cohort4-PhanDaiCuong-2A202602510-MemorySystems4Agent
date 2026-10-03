from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

# Ensure local imports work when running from any cwd
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from model_provider import ProviderConfig, normalize_provider

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _load_env_file(dotenv_path: Path) -> None:
    """Helper to load environment variables from a .env file even without python-dotenv."""
    if load_dotenv is not None and dotenv_path.exists():
        load_dotenv(dotenv_path)
        return
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'\"")
        if key not in os.environ:
            os.environ[key] = val


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a complete LabConfig."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    _load_env_file(root / ".env")

    raw_provider = os.getenv("LLM_PROVIDER", "gemini")
    provider = normalize_provider(raw_provider)
    model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Resolve API key and base URL based on provider
    api_key = None
    base_url = None

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    elif provider == "custom":
        api_key = os.getenv("CUSTOM_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")

    model_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    # Judge configuration
    raw_judge_provider = os.getenv("JUDGE_PROVIDER", raw_provider)
    judge_provider = normalize_provider(raw_judge_provider)
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)
    judge_temperature = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))

    judge_api_key = os.getenv("JUDGE_API_KEY") or api_key
    judge_base_url = os.getenv("JUDGE_BASE_URL") or base_url

    judge_config = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=judge_temperature,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model_config,
        judge_model=judge_config,
    )
