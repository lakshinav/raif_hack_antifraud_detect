from __future__ import annotations

import typing

import pydantic_settings

DEFAULT_OPENROUTER_MODEL: typing.Final = "google/gemini-3.1-flash-lite"
DEFAULT_OPENROUTER_TIMEOUT_SECONDS: typing.Final = 4.2
DEFAULT_RISK_CACHE_SIZE: typing.Final = 1024


@typing.final
class AppSettings(pydantic_settings.BaseSettings):
    openrouter_api_key: str = ""
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_timeout_seconds: float = DEFAULT_OPENROUTER_TIMEOUT_SECONDS
    risk_cache_size: int = DEFAULT_RISK_CACHE_SIZE

    model_config = pydantic_settings.SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_settings() -> AppSettings:
    return AppSettings()
