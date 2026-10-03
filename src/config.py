from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the memory system lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _load_env_file(env_path: Path) -> None:
    """Load environment variables from file with python-dotenv or fallback parser."""
    try:
        import importlib
        dotenv_mod = importlib.import_module("dotenv")
        load_fn = getattr(dotenv_mod, "load_dotenv")
        load_fn(env_path)
    except Exception:
        # Fallback manual parser if python-dotenv is not installed in the interpreter
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
        except Exception:
            pass


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig instance."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    env_file = root / ".env"
    if env_file.exists():
        _load_env_file(env_file)

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)


    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "500"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    provider_str = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")

    model_cfg = ProviderConfig(
        provider=provider_str,
        model_name=model_name,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider_str))
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)
    judge_api_key = os.getenv("JUDGE_API_KEY", api_key)
    judge_base_url = os.getenv("JUDGE_BASE_URL", base_url)

    judge_cfg = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_cfg,
        judge_model=judge_cfg,
    )

