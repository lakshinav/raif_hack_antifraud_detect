# ruff: noqa: RUF001
from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import typing

from app.logging_config import app_logger

RiskCategory = typing.Literal[
    "policy_manipulation",
    "adversarial_attack",
    "identity_deception",
    "transaction_coercion",
    "information_extraction",
    "scope_violation",
]

LOCAL_RULE_MIN_SCORE: typing.Final = 4
LOCAL_RULE_MIN_MARGIN: typing.Final = 2

STATISTICAL_RULES_PATH: typing.Final = pathlib.Path(__file__).with_name("statistical_rules.json")


@typing.final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LocalRulePattern:
    risk_category: RiskCategory
    compiled_pattern: re.Pattern[str]
    score_weight: int


@typing.final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StatisticalRulePhrase:
    risk_category: RiskCategory
    normalized_phrase: str
    score_weight: int


def compile_pattern(pattern_text: str) -> re.Pattern[str]:
    return re.compile(pattern_text, flags=re.IGNORECASE)


CATEGORY_PRIORITY: typing.Final[tuple[RiskCategory, ...]] = (
    "adversarial_attack",
    "identity_deception",
    "information_extraction",
    "transaction_coercion",
    "policy_manipulation",
    "scope_violation",
)

NEGATED_ATTACK_PATTERNS: typing.Final[tuple[re.Pattern[str], ...]] = (
    compile_pattern(r"не\s+прошу\s+отключа\w*\s+protection"),
    compile_pattern(r"не\s+прошу\s+убирать\s+проверк"),
    compile_pattern(r"не\s+предлагаю\s+убрать\s+проверк"),
)

LOCAL_RULE_PATTERNS: typing.Final[tuple[LocalRulePattern, ...]] = (
    LocalRulePattern(
        risk_category="adversarial_attack",
        compiled_pattern=compile_pattern(r"\binternal\s+(?:note|debug|reasoning)\b|внутренн\w+\s+reasoning"),
        score_weight=4,
    ),
    LocalRulePattern(
        risk_category="adversarial_attack",
        compiled_pattern=compile_pattern(r"\btrusted\s+mode\b|режим\s+без\s+дополнительн\w+\s+провер"),
        score_weight=4,
    ),
    LocalRulePattern(
        risk_category="adversarial_attack",
        compiled_pattern=compile_pattern(r"ignore\s+(?:previous\s+)?instructions|system\s+prompt|guardrails"),
        score_weight=4,
    ),
    LocalRulePattern(
        risk_category="adversarial_attack",
        compiled_pattern=compile_pattern(r"отключ\w+\s+(?:провер|защит|protection)|disable\s+(?:security\s+)?checks"),
        score_weight=3,
    ),
    LocalRulePattern(
        risk_category="adversarial_attack",
        compiled_pattern=compile_pattern(r"risk\s+score|сигнал\w+\s+повыша\w+\s+risk\s+score"),
        score_weight=3,
    ),
    LocalRulePattern(
        risk_category="identity_deception",
        compiled_pattern=compile_pattern(r"чуж\w+\s+аккаунт|восстановить\s+чуж"),
        score_weight=5,
    ),
    LocalRulePattern(
        risk_category="identity_deception",
        compiled_pattern=compile_pattern(
            r"доступ\s+к\s+почте.*последн\w+\s+операц|последн\w+\s+операц.*кодовое\s+слово"
        ),
        score_weight=4,
    ),
    LocalRulePattern(
        risk_category="identity_deception",
        compiled_pattern=compile_pattern(r"паспортн\w+\s+данн\w+.*кодовое\s+слово|селфи\s+с\s+паспорт"),
        score_weight=4,
    ),
    LocalRulePattern(
        risk_category="identity_deception",
        compiled_pattern=compile_pattern(
            r"временно\s+отключить\s+подтверждение|доступ\s+буквально\s+на\s+пару\s+минут"
        ),
        score_weight=3,
    ),
    LocalRulePattern(
        risk_category="information_extraction",
        compiled_pattern=compile_pattern(r"операци\w+\s+другого\s+клиента|данн\w+\s+другого\s+клиента"),
        score_weight=5,
    ),
    LocalRulePattern(
        risk_category="information_extraction",
        compiled_pattern=compile_pattern(r"последн\w+\s+(?:активность|операц)|когда.*последн\w+\s+раз"),
        score_weight=4,
    ),
    LocalRulePattern(
        risk_category="information_extraction",
        compiled_pattern=compile_pattern(r"даже\s+без\s+деталей|чисто\s+теоретически.*pending"),
        score_weight=3,
    ),
    LocalRulePattern(
        risk_category="information_extraction",
        compiled_pattern=compile_pattern(r"номер\s+карт\w+\s+(?:жен|муж|клиент)|cvv|код\s+подтверждения"),
        score_weight=4,
    ),
    LocalRulePattern(
        risk_category="transaction_coercion",
        compiled_pattern=compile_pattern(r"требует\s+срочно\s+перевести|продавец\s+очень\s+торопит|перевести\s+аванс"),
        score_weight=4,
    ),
    LocalRulePattern(
        risk_category="transaction_coercion",
        compiled_pattern=compile_pattern(r"оператор\s+подтвердил|фиксируйте.*оператор|вы\s+сказали.*нормально"),
        score_weight=4,
    ),
    LocalRulePattern(
        risk_category="transaction_coercion",
        compiled_pattern=compile_pattern(r"вы\s+бы\s+рискнули|это\s+выглядит\s+как\s+обычная\s+сделка"),
        score_weight=3,
    ),
    LocalRulePattern(
        risk_category="transaction_coercion",
        compiled_pattern=compile_pattern(
            r"одобр\w+\s+операц|(?:подтверди|подтвердите)\s+перевод|ускор\w+\s+(?:платеж|операц|перевод)"
        ),
        score_weight=4,
    ),
    LocalRulePattern(
        risk_category="policy_manipulation",
        compiled_pattern=compile_pattern(r"сдела\w+\s+исключение|нестандартн\w+\s+решени|premium-клиент"),
        score_weight=4,
    ),
    LocalRulePattern(
        risk_category="policy_manipulation",
        compiled_pattern=compile_pattern(r"изменить\s+интерпретац|условно\s+подтвержден|трактовать\s+статус"),
        score_weight=4,
    ),
    LocalRulePattern(
        risk_category="policy_manipulation",
        compiled_pattern=compile_pattern(r"обойти\s+(?:правил|policy)|bypass\s+(?:policy|rules)"),
        score_weight=4,
    ),
    LocalRulePattern(
        risk_category="scope_violation",
        compiled_pattern=compile_pattern(r"жалоб\w+\s+в\s+цб|помочь\s+с\s+(?:python|код|csv)|debug\w*\s+etl"),
        score_weight=5,
    ),
    LocalRulePattern(
        risk_category="scope_violation",
        compiled_pattern=compile_pattern(r"mobile\s+dev|apns|fcm|backend\s+retries|composite\s+key"),
        score_weight=4,
    ),
)


def prepare_normalized_dictionary_phrase(phrase_text: str) -> str:
    return phrase_text.casefold().replace("ё", "е")


def parse_statistical_phrase(
    risk_category: RiskCategory,
    phrase_payload: object,
) -> StatisticalRulePhrase | None:
    if not isinstance(phrase_payload, dict):
        return None

    phrase_text = phrase_payload.get("phrase")
    score_weight = phrase_payload.get("weight")
    if not isinstance(phrase_text, str) or not isinstance(score_weight, int):
        return None

    return StatisticalRulePhrase(
        risk_category=risk_category,
        normalized_phrase=prepare_normalized_dictionary_phrase(phrase_text),
        score_weight=score_weight,
    )


def parse_statistical_rules(raw_payload: object) -> tuple[StatisticalRulePhrase, ...]:
    if not isinstance(raw_payload, dict):
        return ()

    categories_payload = raw_payload.get("categories")
    if not isinstance(categories_payload, dict):
        return ()

    parsed_phrases: list[StatisticalRulePhrase] = []
    for one_risk_category in CATEGORY_PRIORITY:
        phrases_payload = categories_payload.get(one_risk_category)
        if not isinstance(phrases_payload, list):
            continue
        for one_phrase_payload in phrases_payload:
            parsed_phrase = parse_statistical_phrase(one_risk_category, one_phrase_payload)
            if parsed_phrase is not None:
                parsed_phrases.append(parsed_phrase)

    return tuple(parsed_phrases)


def load_statistical_rules() -> tuple[StatisticalRulePhrase, ...]:
    try:
        raw_payload: object = json.loads(STATISTICAL_RULES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as load_error:
        app_logger.warning("Statistical rules load failed: {}", load_error)
        return ()

    parsed_phrases = parse_statistical_rules(raw_payload)
    app_logger.info("Statistical rules loaded: phrases={}", len(parsed_phrases))
    return parsed_phrases


STATISTICAL_RULE_PHRASES: typing.Final = load_statistical_rules()


def prepare_normalized_dialogue_text(dialogue_text: str) -> str:
    return dialogue_text.casefold().replace("ё", "е")


def check_negated_attack(dialogue_text: str) -> bool:
    return any(one_compiled_pattern.search(dialogue_text) for one_compiled_pattern in NEGATED_ATTACK_PATTERNS)


def build_initial_scores() -> dict[RiskCategory, int]:
    return dict.fromkeys(CATEGORY_PRIORITY, 0)


def extract_user_dialogue_text(dialogue_text: str) -> str:
    user_lines = [
        one_line.removeprefix("user:").strip()
        for one_line in dialogue_text.splitlines()
        if one_line.casefold().startswith("user:")
    ]
    if user_lines:
        return "\n".join(user_lines)

    return dialogue_text


def build_statistical_scores(normalized_dialogue: str) -> dict[RiskCategory, int]:
    category_scores = build_initial_scores()
    for one_rule_phrase in STATISTICAL_RULE_PHRASES:
        if one_rule_phrase.normalized_phrase in normalized_dialogue:
            category_scores[one_rule_phrase.risk_category] += one_rule_phrase.score_weight

    return category_scores


def process_dialogue_with_statistical_dictionary(normalized_dialogue: str) -> RiskCategory | None:
    category_scores = build_statistical_scores(normalized_dialogue)
    top_category = choose_top_category(category_scores)
    if top_category is None:
        app_logger.info("Statistical dictionary result: fallback_to_regex scores={}", category_scores)
        return None

    app_logger.info("Statistical dictionary result: category={} scores={}", top_category, category_scores)
    return top_category


def choose_top_category(category_scores: dict[RiskCategory, int]) -> RiskCategory | None:
    ranked_categories = sorted(
        category_scores.items(),
        key=lambda category_item: (
            -category_item[1],
            CATEGORY_PRIORITY.index(category_item[0]),
        ),
    )
    top_category, top_score = ranked_categories[0]
    second_score = ranked_categories[1][1]

    if top_score < LOCAL_RULE_MIN_SCORE:
        return None

    if second_score > 0 and top_score - second_score < LOCAL_RULE_MIN_MARGIN:
        return None

    return top_category


def process_dialogue_with_local_rules(dialogue_text: str) -> RiskCategory | None:
    normalized_dialogue = prepare_normalized_dialogue_text(extract_user_dialogue_text(dialogue_text))

    statistical_category = process_dialogue_with_statistical_dictionary(normalized_dialogue)
    if statistical_category is not None:
        return statistical_category

    category_scores = build_initial_scores()

    for one_local_pattern in LOCAL_RULE_PATTERNS:
        if one_local_pattern.risk_category == "adversarial_attack" and check_negated_attack(normalized_dialogue):
            continue
        if one_local_pattern.compiled_pattern.search(normalized_dialogue):
            category_scores[one_local_pattern.risk_category] += one_local_pattern.score_weight

    top_category = choose_top_category(category_scores)
    if top_category is None:
        app_logger.info("Local rules result: fallback_to_llm scores={}", category_scores)
        return None

    app_logger.info("Local rules result: category={} scores={}", top_category, category_scores)
    return top_category
