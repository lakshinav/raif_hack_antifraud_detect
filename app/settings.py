from __future__ import annotations

import typing

import pydantic_settings

DEFAULT_OPENROUTER_MODEL: typing.Final = "google/gemini-3.1-flash-lite"
DEFAULT_SECONDARY_OPENROUTER_MODEL: typing.Final = "openai/gpt-5.4-mini"
DEFAULT_JUDGE_OPENROUTER_MODEL: typing.Final = "openai/gpt-5.5"
DEFAULT_OPENROUTER_TIMEOUT_SECONDS: typing.Final = 4.2
DEFAULT_RISK_CACHE_SIZE: typing.Final = 1024
DEFAULT_RISK_CONFIDENCE_THRESHOLD: typing.Final = 0.80
DEFAULT_CLEAN_CONFIDENCE_THRESHOLD: typing.Final = 0.55
DEFAULT_RISK_FAST_ACCEPT_CONFIDENCE_THRESHOLD: typing.Final = 0.87
DEFAULT_RISK_SECONDARY_CONFIDENCE_THRESHOLD: typing.Final = 0.75
DEFAULT_RISK_AGREEMENT_CONFIDENCE_THRESHOLD: typing.Final = 0.78
DEFAULT_ENABLE_CLEAN_ROLE_DISTRIBUTION_CHECK: typing.Final = True
DEFAULT_ENABLE_LOCAL_STATISTICAL_RULES_CHECK: typing.Final = True
DEFAULT_ENABLE_LOCAL_REGEX_RULES_CHECK: typing.Final = True
DEFAULT_CLEAN_USER_MESSAGE_FREQUENCY_THRESHOLD: typing.Final = 0.35
DEFAULT_RECOVERY_REVIEW_USER_MESSAGE_FREQUENCY: typing.Final = 0.80
DEFAULT_RECOVERY_REVIEW_MIN_MESSAGES: typing.Final = 35
DEFAULT_LOG_LEVEL: typing.Final = "WARNING"


@typing.final
class AppSettings(pydantic_settings.BaseSettings):
    openrouter_api_key: str = ""
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_primary_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_secondary_model: str = DEFAULT_SECONDARY_OPENROUTER_MODEL
    openrouter_judge_model: str = DEFAULT_JUDGE_OPENROUTER_MODEL
    openrouter_timeout_seconds: float = DEFAULT_OPENROUTER_TIMEOUT_SECONDS
    risk_cache_size: int = DEFAULT_RISK_CACHE_SIZE
    risk_confidence_threshold: float = DEFAULT_RISK_CONFIDENCE_THRESHOLD
    clean_confidence_threshold: float = DEFAULT_CLEAN_CONFIDENCE_THRESHOLD
    risk_fast_accept_confidence_threshold: float = DEFAULT_RISK_FAST_ACCEPT_CONFIDENCE_THRESHOLD
    risk_secondary_confidence_threshold: float = DEFAULT_RISK_SECONDARY_CONFIDENCE_THRESHOLD
    risk_agreement_confidence_threshold: float = DEFAULT_RISK_AGREEMENT_CONFIDENCE_THRESHOLD
    enable_clean_role_distribution_check: bool = DEFAULT_ENABLE_CLEAN_ROLE_DISTRIBUTION_CHECK
    enable_local_statistical_rules_check: bool = DEFAULT_ENABLE_LOCAL_STATISTICAL_RULES_CHECK
    enable_local_regex_rules_check: bool = DEFAULT_ENABLE_LOCAL_REGEX_RULES_CHECK
    clean_user_message_frequency_threshold: float = DEFAULT_CLEAN_USER_MESSAGE_FREQUENCY_THRESHOLD
    recovery_review_user_message_frequency: float = DEFAULT_RECOVERY_REVIEW_USER_MESSAGE_FREQUENCY
    recovery_review_min_messages: int = DEFAULT_RECOVERY_REVIEW_MIN_MESSAGES
    log_level: str = DEFAULT_LOG_LEVEL

    model_config = pydantic_settings.SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_settings() -> AppSettings:
    return AppSettings()
