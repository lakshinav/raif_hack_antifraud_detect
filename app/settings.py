from __future__ import annotations

import typing

import pydantic_settings

DEFAULT_OPENROUTER_MODEL: typing.Final = "anthropic/claude-opus-4.8-fast"
DEFAULT_SECONDARY_OPENROUTER_MODEL: typing.Final = "anthropic/claude-sonnet-4.6"
DEFAULT_JUDGE_OPENROUTER_MODEL: typing.Final = "anthropic/claude-opus-4.8"
DEFAULT_OPENROUTER_TIMEOUT_SECONDS: typing.Final = 4.2
DEFAULT_RISK_CACHE_SIZE: typing.Final = 1024
DEFAULT_RISK_CONFIDENCE_THRESHOLD: typing.Final = 0.78
DEFAULT_CLEAN_CONFIDENCE_THRESHOLD: typing.Final = 0.84
DEFAULT_RISK_FAST_ACCEPT_CONFIDENCE_THRESHOLD: typing.Final = 0.88
DEFAULT_RISK_SECONDARY_CONFIDENCE_THRESHOLD: typing.Final = 0.86
DEFAULT_RISK_AGREEMENT_CONFIDENCE_THRESHOLD: typing.Final = 0.74
DEFAULT_ENABLE_CLEAN_ROLE_DISTRIBUTION_CHECK: typing.Final = True
DEFAULT_ENABLE_LOCAL_STATISTICAL_RULES_CHECK: typing.Final = False
DEFAULT_ENABLE_LOCAL_REGEX_RULES_CHECK: typing.Final = False
DEFAULT_CLEAN_USER_MESSAGE_FREQUENCY_THRESHOLD: typing.Final = 0.30
DEFAULT_RECOVERY_REVIEW_USER_MESSAGE_FREQUENCY: typing.Final = 0.50
DEFAULT_RECOVERY_REVIEW_MIN_MESSAGES: typing.Final = 12
DEFAULT_LOG_LEVEL: typing.Final = "INFO"


@typing.final
class AppSettings(pydantic_settings.BaseSettings):
    openrouter_api_key: str = ""
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_primary_model: str = ""
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
