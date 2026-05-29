from __future__ import annotations

import typing

import pydantic_settings

DEFAULT_OPENROUTER_MODEL: typing.Final = "openai/gpt-4o-mini" #"google/gemini-3.1-flash-lite"
DEFAULT_SECONDARY_OPENROUTER_MODEL: typing.Final = "qwen/qwen3.7-max"
DEFAULT_JUDGE_OPENROUTER_MODEL: typing.Final = "openai/gpt-5.5"
DEFAULT_OPENROUTER_TIMEOUT_SECONDS: typing.Final = 8
DEFAULT_RISK_CACHE_SIZE: typing.Final = 1024
DEFAULT_RISK_CONFIDENCE_THRESHOLD: typing.Final = 0.70
DEFAULT_CLEAN_CONFIDENCE_THRESHOLD: typing.Final = 0.70
DEFAULT_RISK_FAST_ACCEPT_CONFIDENCE_THRESHOLD: typing.Final = 0.70
DEFAULT_RISK_SECONDARY_CONFIDENCE_THRESHOLD: typing.Final = 0.70
DEFAULT_RISK_AGREEMENT_CONFIDENCE_THRESHOLD: typing.Final = 0.65
DEFAULT_ENABLE_CLEAN_ROLE_DISTRIBUTION_CHECK: typing.Final = True
DEFAULT_ENABLE_LOCAL_STATISTICAL_RULES_CHECK: typing.Final = True
DEFAULT_ENABLE_LOCAL_REGEX_RULES_CHECK: typing.Final = True
DEFAULT_CLEAN_USER_MESSAGE_FREQUENCY_THRESHOLD: typing.Final = 0.30
DEFAULT_RECOVERY_REVIEW_USER_MESSAGE_FREQUENCY: typing.Final = 0.90
DEFAULT_RECOVERY_REVIEW_MIN_MESSAGES: typing.Final = 50
DEFAULT_LOG_LEVEL: typing.Final = "WARNING"

# Category-specific confidence thresholds for improved precision/recall balance
# Higher thresholds reduce false positives, lower thresholds improve recall
DEFAULT_IDENTITY_DECEPTION_CONFIDENCE_THRESHOLD: typing.Final = 0.80  # Higher to reduce false positives
DEFAULT_ADVERSARIAL_ATTACK_CONFIDENCE_THRESHOLD: typing.Final = 0.75  # Higher to reduce false positives
DEFAULT_TRANSACTION_COERCION_CONFIDENCE_THRESHOLD: typing.Final = 0.60  # Lower to improve recall
DEFAULT_POLICY_MANIPULATION_CONFIDENCE_THRESHOLD: typing.Final = 0.60  # Lower to improve recall
DEFAULT_INFORMATION_EXTRACTION_CONFIDENCE_THRESHOLD: typing.Final = 0.65  # Lower to improve recall
DEFAULT_SCOPE_VIOLATION_CONFIDENCE_THRESHOLD: typing.Final = 0.65

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

    # Category-specific confidence thresholds
    identity_deception_confidence_threshold: float = DEFAULT_IDENTITY_DECEPTION_CONFIDENCE_THRESHOLD
    adversarial_attack_confidence_threshold: float = DEFAULT_ADVERSARIAL_ATTACK_CONFIDENCE_THRESHOLD
    transaction_coercion_confidence_threshold: float = DEFAULT_TRANSACTION_COERCION_CONFIDENCE_THRESHOLD
    policy_manipulation_confidence_threshold: float = DEFAULT_POLICY_MANIPULATION_CONFIDENCE_THRESHOLD
    information_extraction_confidence_threshold: float = DEFAULT_INFORMATION_EXTRACTION_CONFIDENCE_THRESHOLD
    scope_violation_confidence_threshold: float = DEFAULT_SCOPE_VIOLATION_CONFIDENCE_THRESHOLD

    model_config = pydantic_settings.SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_settings() -> AppSettings:
    return AppSettings()
