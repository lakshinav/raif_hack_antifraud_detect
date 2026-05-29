from __future__ import annotations

import typing

import pydantic_settings

DEFAULT_OPENROUTER_MODEL: typing.Final = "google/gemini-3.1-flash-lite"
DEFAULT_OPENROUTER_TIMEOUT_SECONDS: typing.Final = 4.2


@typing.final
class AppSettings(pydantic_settings.BaseSettings):
    openrouter_api_key: str = ""
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_timeout_seconds: float = DEFAULT_OPENROUTER_TIMEOUT_SECONDS

    model_config = pydantic_settings.SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_settings() -> AppSettings:
    return AppSettings()
