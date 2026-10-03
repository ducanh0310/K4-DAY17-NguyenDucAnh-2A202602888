from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by agents.

    Supported providers:
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
    """Map aliases like `anthorpic` -> `anthropic`, `google` -> `gemini`, etc."""
    val = (value or "").strip().lower()
    mapping = {
        "anthorpic": "anthropic",
        "google": "gemini",
        "google-genai": "gemini",
        "gemini-api": "gemini",
        "openai-custom": "custom",
    }
    return mapping.get(val, val)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Providers:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenAI` with OpenRouter base URL
    """
    import importlib

    provider = normalize_provider(config.provider)

    if provider == "openai":
        mod = importlib.import_module("langchain_openai")
        chat_cls = getattr(mod, "ChatOpenAI")
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        return chat_cls(model=config.model_name, temperature=config.temperature, api_key=api_key)

    elif provider == "custom":
        mod = importlib.import_module("langchain_openai")
        chat_cls = getattr(mod, "ChatOpenAI")
        api_key = config.api_key or os.getenv("CUSTOM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = config.base_url or os.getenv("CUSTOM_BASE_URL")
        return chat_cls(model=config.model_name, temperature=config.temperature, api_key=api_key, base_url=base_url)

    elif provider == "gemini":
        mod = importlib.import_module("langchain_google_genai")
        chat_cls = getattr(mod, "ChatGoogleGenerativeAI")
        api_key = config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return chat_cls(model=config.model_name, temperature=config.temperature, google_api_key=api_key)

    elif provider == "anthropic":
        mod = importlib.import_module("langchain_anthropic")
        chat_cls = getattr(mod, "ChatAnthropic")
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        return chat_cls(model=config.model_name, temperature=config.temperature, api_key=api_key)

    elif provider == "ollama":
        mod = importlib.import_module("langchain_ollama")
        chat_cls = getattr(mod, "ChatOllama")
        base_url = config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return chat_cls(model=config.model_name, temperature=config.temperature, base_url=base_url)

    elif provider == "openrouter":
        mod = importlib.import_module("langchain_openai")
        chat_cls = getattr(mod, "ChatOpenAI")
        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        base_url = config.base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return chat_cls(model=config.model_name, temperature=config.temperature, api_key=api_key, base_url=base_url)

    else:
        raise ValueError(f"Unsupported provider: {config.provider} (normalized: {provider})")


