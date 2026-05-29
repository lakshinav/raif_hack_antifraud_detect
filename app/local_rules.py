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

LOCAL_RULE_MIN_SCORE: typing.Final = 5
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
        risk_category="adversarial_attack",
        compiled_pattern=compile_pattern(
            r"penetration\s+test.*answer\s+strictly\s+in\s+json|give\s+me\s+thresholds|"
            r"answer\s+strictly\s+in\s+yaml.*system\s+internals|admin\s+would\s+see|реальн\w+\s+поля"
        ),
        score_weight=5,
    ),
    LocalRulePattern(
        risk_category="adversarial_attack",
        compiled_pattern=compile_pattern(
            r"answer\s+strictly\s+as\s+csv|prompt\s+template\s+for\s+risk\s+detection|"
            r"reasoning\s+chains|endpoint\s+calls\s+reveal|internal\s+(?:flags|error\s+codes|path)|"
            r"(?:exact|typical)\s+(?:log\s+retention|thresholds|error\s+code)|"
            r"bypass\s+the\s+rate\s+limiter|role\s+assertions|privileged\s+response\s+format|"
            r"model\s+internals|reveal\s+sql\s+queries|db\s+tables|internal\s+api|"
            r"risk\s+score\s+most|give\s+percentages|"
            r"внутренн\w+\s+сигнал|поля.*логах\s+по\s+транзакц|признак\w+\s+в\s+тексте\s+триггер\w+|"
            r"обходить\s+систему|в\s+конце\s+выдадите\s+внутренн"
        ),
        score_weight=6,
    ),
    LocalRulePattern(
        risk_category="adversarial_attack",
        compiled_pattern=compile_pattern(
            r"example\s+prompt\s+that\s+causes\s+the\s+system\s+to\s+leak\s+info|"
            r"progressively\s+ask.*slightly\s+sensitive|eventually\s+reveal\s+patterns|"
            r"list\s+adversarial\s+patterns(?:.|\n)*provide\s+examples"
        ),
        score_weight=8,
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
        risk_category="identity_deception",
        compiled_pattern=compile_pattern(
            r"reset\s+(?:her|his|their)\s+login\s+for\s+me|"
            r"asked\s+me\s+to\s+help.*(?:reset|login|recovery|verify)|"
            r"let\s+me\s+in\s+to\s+help\s+my\s+friend|"
            r"(?:one-time\s+code|app\s+password).*(?:use|for\s+him|for\s+her)"
        ),
        score_weight=5,
    ),
    LocalRulePattern(
        risk_category="identity_deception",
        compiled_pattern=compile_pattern(
            r"(?:mother|dad|brother|colleague|aunt|partner|sister|friend).*(?:phone|sim|sms|access)"
            r".*(?:asked\s+me|reset|switch|verify\s+on|confirm\s+transactions)|"
            r"act\s+as\s+(?:his|her)\s+representative|acting\s+for\s+my\s+friend|"
            r"on\s+(?:her|his)\s+behalf|transfer\s+ownership\s+to\s+me\s+temporarily|"
            r"reset\s+(?:his\s+email|her\s+password)\s+for\s+me|temporarily\s+confirm\s+transactions\s+for\s+him|"
            r"сч[её]та\s+моего\s+друга|действовать\s+от\s+его\s+имени|"
            r"перевыпуск\s+карт\w+\s+на\s+е[её]\s+имя|временно\s+дать\s+мне\s+просмотр|"
            r"выдать\s+временн\w+\s+код\s+мне|восстановить\s+доступ|"
            r"сменить\s+контактн\w+\s+телефон\s+у\s+клиент|помощник\s+клиент\w+.*выписк|"
            r"устное\s+разрешение|доступ\s+к\s+сч[её]ту\s+родственник|временно\s+обойти\s+подтверждение"
        ),
        score_weight=6,
    ),
    LocalRulePattern(
        risk_category="identity_deception",
        compiled_pattern=compile_pattern(
            r"снять\s+лимит\s+со\s+сч[её]та\s+моего\s+друга|"
            r"телефон\s+родственника\s+украден.*сменить\s+способ\s+подтверждения|"
            r"подключиться\s+как\s+доверенное\s+лицо|"
            r"отключить\s+двухфакторн\w+\s+аутентификац"
        ),
        score_weight=8,
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
        risk_category="information_extraction",
        compiled_pattern=compile_pattern(
            r"раскрывать\s+данн\w+\s+клиент\w+(?:.|\n)*активн\w+|"
            r"даже\s+просто.*без\s+деталей|"
            r"which\s+belong\s+to\s+my\s+colleague'?s\s+account|"
            r"tell\s+me\s+whether\s+(?:a\s+)?(?:payout|account|transfer|transaction)"
        ),
        score_weight=5,
    ),
    LocalRulePattern(
        risk_category="information_extraction",
        compiled_pattern=compile_pattern(
            r"can't\s+share\s+account\s+info|can't\s+share\s+specifics|"
            r"account\s+had\s+activity|recent\s+refunds\s+to\s+account|"
            r"associated\s+with\s+account.*gmail|"
            r"last\s+4\s+digits.*confirm\s+a\s+recent\s+transaction|"
            r"review\s+status.*tell\s+me\s+if\s+it'?s\s+fraud|"
            r"wording\s+alone.*friend'?s\s+account|"
            r"was\s+there\s+a\s+transfer.*from\s+which\s+bank.*to\s+which\s+account|"
            r"under\s+review.*does\s+that\s+always\s+mean\s+fraud|"
            r"подтвердить.*последн\w+\s+списание|было\s+ли\s+списание|"
            r"движени\w+\s+за\s+24\s+часа|попытк\w+\s+восстановления\s+пароля|"
            r"попытк\w+\s+входа|номер\s+терминала|отобража\w+.*отмен[её]нн\w+\s+перевод"
        ),
        score_weight=6,
    ),
    LocalRulePattern(
        risk_category="information_extraction",
        compiled_pattern=compile_pattern(
            r"карте\s+сегодня\s+появилось\s+списание.*подтвердить.*перевод|"
            r"паспортн\w+\s+данн\w+.*вы\s+тогда\s+скажете|"
            r"последн\w+\s+входящ\w+\s+платеж|платеж\s+скрыт.*отклон[её]н|"
            r"операция\s+помечена\s+'?internal\s+review'?"
        ),
        score_weight=8,
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
        risk_category="transaction_coercion",
        compiled_pattern=compile_pattern(
            r"payment\s+that\s+must\s+clear\s+today|approve\s+accelerating\s+it|"
            r"previous\s+rep.*(?:fine|safe)|"
            r"bypass\s+a\s+fraud\s+hold|mark\s+it\s+as\s+a\s+false\s+positive|manually\s+override|"
            r"cancel\s+the\s+review\s+and\s+push\s+the\s+payment"
        ),
        score_weight=5,
    ),
    LocalRulePattern(
        risk_category="transaction_coercion",
        compiled_pattern=compile_pattern(
            r"transfer\s+is\s+critical.*approve|seller.*push\s+it|"
            r"split\s+a\s+large\s+transfer.*bypass\s+a\s+daily\s+limit|"
            r"release\s+it.*employees|confirm\s+the\s+payment.*mark\s+it\s+as\s+safe|"
            r"run\s+the\s+transfer\s+manually|manager\s+already\s+approved\s+verbally|"
            r"fast-track\s+review|exception\s+to\s+processing\s+rules.*vendor\s+payout|"
            r"срочно\s+провести\s+перевод|дать\s+добро\s+вручную|пропустить\s+проверку|"
            r"перевод\s+обработали\s+раньше\s+лимита|срочн\w+\s+платеж\s+поставщик|"
            r"голосов\w+\s+подтверждени|провести\s+без\s+дополнительн\w+\s+проверки|"
            r"интуитивн\w+\s+проверку|разблокировать\s+плат[её]ж|потребуем\s+компенсацию|"
            r"провести\s+перевод\s+до\s+конца\s+дня|банк\s+компенсирует\s+штраф"
        ),
        score_weight=6,
    ),
    LocalRulePattern(
        risk_category="transaction_coercion",
        compiled_pattern=compile_pattern(
            r"payroll\s+must\s+go\s+out\s+today|temporary\s+hold.*release\s+it|"
            r"банк\s+не\s+верн[её]т\s+ответственность|"
            r"клиент\s+категорически\s+требует\s+провести\s+сейчас|"
            r"банк\s+не\s+проверял\s+подозрительн\w+\s+элемент|закрыть\s+глазами"
        ),
        score_weight=8,
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
        risk_category="policy_manipulation",
        compiled_pattern=compile_pattern(
            r"трактовать\s+провер\w+\s+как\s+ненужн\w+|закро\w+\s+постфактум|"
            r"техническ\w+\s+формальн\w+|"
            r"reclassify\s+an\s+alert\s+as\s+informational|client\s+accepts\s+liability|"
            r"interpret\s+the\s+rule\s+loosely"
        ),
        score_weight=5,
    ),
    LocalRulePattern(
        risk_category="policy_manipulation",
        compiled_pattern=compile_pattern(
            r"treat\s+it\s+as\s+a\s+formality|close\s+it\s+after\s+the\s+payment|"
            r"record\s+a\s+note.*client\s+accepts\s+all\s+risks|approved\s+in\s+advance|"
            r"ignore\s+compliance|treat\s+it\s+as\s+resolved|record\s+that\s+and\s+unblock|"
            r"soften\s+policy\s+language|operational\s+review,\s+not\s+fraud|"
            r"интерпретировать.*чтобы\s+плат[её]ж\s+прош[её]л|формальн\w+\s+проверк|"
            r"трактовать\s+статус|сделать\s+исключение.*понимаете\s+бизнес|"
            r"поставьте\s+галочку|подстроить.*проверку|не\s+проводить\s+полн\w+\s+аудит|"
            r"формальн\w+\s+подпись\s+банка|не\s+делали\s+акцент.*проверке|"
            r"клиент\s+подтверждает"
        ),
        score_weight=6,
    ),
    LocalRulePattern(
        risk_category="policy_manipulation",
        compiled_pattern=compile_pattern(
            r"акт\s+о\s+том,\s+что\s+банк\s+не\s+нес[её]т\s+ответственности|"
            r"treat\s+it\s+as\s+resolved|vendor\s+agreed\s+to\s+accept\s+risk|"
            r"record\s+that\s+and\s+let\s+the\s+payment\s+go|"
            r"flex'?'\s+the\s+rule|меняем\s+интерпретацию"
        ),
        score_weight=8,
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
    LocalRulePattern(
        risk_category="scope_violation",
        compiled_pattern=compile_pattern(
            r"investment\s+advice|investment\s+strategy|portfolio|"
            r"python/etl|full\s+pipeline|"
            r"маркетингов\w+\s+кампан|рекламн\w+\s+агент|баннер|a/b\s+тест"
        ),
        score_weight=5,
    ),
    LocalRulePattern(
        risk_category="scope_violation",
        compiled_pattern=compile_pattern(
            r"complaint\s+letter|filling\s+a\s+regulatory\s+form.*writing\s+a\s+complaint|"
            r"external\s+authority|press\s+release|"
            r"blog\s+post|legal\s+contract|contract\s+template|startup\s+pitch|pitch\s+deck|"
            r"predictive\s+legal\s+risk\s+model|marketing\s+campaign|creative\s+advert|"
            r"pr\s+writer|regulatory\s+filing|full\s+legal\s+disclaimer|seo-article|"
            r"жалоб\w+\s+регулятор|официальн\w+\s+письм|юридическ\w+\s+формулировк|"
            r"пресс-релиз|пиар-текст|меморандум|официальн\w+\s+заявлени|"
            r"адвокат\w+\s+клиент|составь\s+иск|бизнес-план|финансов\w+\s+прогноз|"
            r"роль\s+аудитор|кейс-стади|официальн\w+\s+обращени|внедрени\w+\s+.*crm"
        ),
        score_weight=6,
    ),
    LocalRulePattern(
        risk_category="scope_violation",
        compiled_pattern=compile_pattern(
            r"technical\s+advice\s+for\s+securing\s+our\s+integration|"
            r"concrete\s+code\s+samples\s+for\s+key\s+rotation|"
            r"кредитн\w+\s+скоринг.*инвестиционн\w+\s+стратеги|"
            r"образовательн\w+\s+курс\s+по\s+управлению\s+долгом"
        ),
        score_weight=8,
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


def process_dialogue_with_local_rules(
    dialogue_text: str,
    *,
    enable_statistical_rules_check: bool = True,
    enable_regex_rules_check: bool = True,
) -> RiskCategory | None:
    normalized_dialogue = prepare_normalized_dialogue_text(extract_user_dialogue_text(dialogue_text))

    if enable_statistical_rules_check:
        statistical_category = process_dialogue_with_statistical_dictionary(normalized_dialogue)
        if statistical_category is not None:
            return statistical_category
    else:
        app_logger.info("Local statistical rules skipped by feature flag")

    if not enable_regex_rules_check:
        app_logger.info("Local regex rules skipped by feature flag")
        return None

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
