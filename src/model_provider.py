from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`."""
    if not value or not isinstance(value, str):
        raise ValueError("Provider must be a non-empty string.")

    cleaned = value.strip().lower()
    mapping = {
        "openai": "openai",
        "custom": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "google-genai": "gemini",
        "google_genai": "gemini",
        "anthropic": "anthropic",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "ollama": "ollama",
        "openrouter": "openrouter",
        "open-router": "openrouter",
        "open_router": "openrouter",
    }

    if cleaned in mapping:
        return mapping[cleaned]

    raise ValueError(
        f"Unsupported provider: '{value}'. "
        f"Supported providers: openai, custom, gemini, anthropic, ollama, openrouter"
    )


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the real chat model for the selected provider.

    Mapping:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter` or `ChatOpenAI` with OpenRouter base URL
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    elif provider == "custom":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or "EMPTY",
            base_url=config.base_url,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    elif provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter

            return ChatOpenRouter(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )
        except ImportError:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url or "https://openrouter.ai/api/v1",
            )

    raise ValueError(f"Unknown provider: {provider}")
