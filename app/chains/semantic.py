"""Семантический классификатор-подсказка для detection_chain.

Кодирует эталонные диалоги из train.json через OpenAI-совместимый /embeddings эндпоинт,
строит два центроида (attack / clean) и для входного диалога возвращает метку близости:
positive (похоже на clean) / negative (похоже на attack) / neutral (непонятно).

Любая ошибка (нет ключа, провайдер не отдаёт /embeddings, сетевой сбой) приводит к None —
вызывающая сторона трактует это как "подсказки нет" и работает как раньше.
"""

from __future__ import annotations

import json
import typing

import httpx
import numpy as np

from app.chains.prompts import TRAIN_DATA_PATH, format_dialogue
from app.logging_config import app_logger
from app.settings import load_settings

SemanticHint = typing.Literal["positive", "negative", "neutral"]


@typing.final
class Centroids(typing.NamedTuple):
    """L2-нормированные центроиды классов attack/clean."""

    attack: np.ndarray
    clean: np.ndarray


# Кэш центроидов на процесс. None означает "ещё не считали"; неудачу не кэшируем.
_CENTROIDS_CACHE: Centroids | None = None


async def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Батч-запрос эмбеддингов на OpenAI-совместимый /embeddings. None при любой ошибке."""
    app_settings = load_settings()
    if not app_settings.openrouter_api_key:
        app_logger.warning("embeddings_request_skipped", reason="empty_api_key")
        return None
    if not texts:
        return []

    embeddings_url = f"{app_settings.embeddings_base_url.rstrip('/')}/embeddings"
    request_payload = {"model": app_settings.embeddings_model, "input": texts}
    app_logger.info(
        "embeddings_request_started",
        model=app_settings.embeddings_model,
        input_count=len(texts),
    )
    try:
        async with httpx.AsyncClient(timeout=app_settings.openrouter_timeout_seconds) as httpx_connection:
            response = await httpx_connection.post(
                embeddings_url,
                headers={
                    "Authorization": f"Bearer {app_settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
            )
        response.raise_for_status()
    except httpx.HTTPError as http_error:
        app_logger.warning("embeddings_request_failed", model=app_settings.embeddings_model, error=str(http_error))
        return None

    try:
        response_payload: object = response.json()
    except ValueError:
        app_logger.warning("embeddings_response_invalid_json", model=app_settings.embeddings_model)
        return None

    return parse_embeddings_payload(response_payload, expected_count=len(texts))


def parse_embeddings_payload(response_payload: object, *, expected_count: int) -> list[list[float]] | None:
    if not isinstance(response_payload, dict):
        return None

    data_payload = response_payload.get("data")
    if not isinstance(data_payload, list) or len(data_payload) != expected_count:
        app_logger.warning("embeddings_response_unexpected_shape")
        return None

    vectors: list[list[float]] = []
    for item in data_payload:
        if not isinstance(item, dict):
            return None
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not all(isinstance(value, int | float) for value in embedding):
            return None
        vectors.append([float(value) for value in embedding])

    return vectors


def load_reference_dialogues() -> tuple[list[str], list[str]]:
    """Возвращает (attack_dialogues, clean_dialogues) из train.json."""
    sessions = json.loads(TRAIN_DATA_PATH.read_text(encoding="utf-8"))
    attack_dialogues: list[str] = []
    clean_dialogues: list[str] = []
    for session in sessions:
        dialogue = format_dialogue(session.get("messages", []))
        if session.get("expected_red_flags"):
            attack_dialogues.append(dialogue)
        else:
            clean_dialogues.append(dialogue)
    return attack_dialogues, clean_dialogues


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return np.asarray(vector / norm)


def build_class_centroid(vectors: list[list[float]]) -> np.ndarray:
    """Среднее L2-нормированных векторов, затем повторная нормировка центроида."""
    matrix = np.array(vectors, dtype=np.float64)
    normalized = np.array([normalize_vector(row) for row in matrix])
    return normalize_vector(normalized.mean(axis=0))


async def get_centroids() -> Centroids | None:
    """Lazy-построение центроидов из train.json с кэшем на процесс. None при ошибке эмбеддинга."""
    global _CENTROIDS_CACHE  # noqa: PLW0603
    if _CENTROIDS_CACHE is not None:
        return _CENTROIDS_CACHE

    attack_dialogues, clean_dialogues = load_reference_dialogues()
    if not attack_dialogues or not clean_dialogues:
        app_logger.warning("semantic_centroids_skipped", reason="missing_reference_class")
        return None

    embeddings = await embed_texts(attack_dialogues + clean_dialogues)
    if embeddings is None:
        return None

    attack_centroid = build_class_centroid(embeddings[: len(attack_dialogues)])
    clean_centroid = build_class_centroid(embeddings[len(attack_dialogues) :])
    _CENTROIDS_CACHE = Centroids(attack=attack_centroid, clean=clean_centroid)
    app_logger.info(
        "semantic_centroids_built",
        attack_count=len(attack_dialogues),
        clean_count=len(clean_dialogues),
    )
    return _CENTROIDS_CACHE


async def classify_semantic(dialogue: str) -> SemanticHint | None:
    """Метка близости диалога к attack/clean центроидам. None если эмбеддинги недоступны."""
    centroids = await get_centroids()
    if centroids is None:
        return None

    embeddings = await embed_texts([dialogue])
    if not embeddings:
        return None

    dialogue_vector = normalize_vector(np.array(embeddings[0], dtype=np.float64))
    attack_similarity = float(dialogue_vector @ centroids.attack)
    clean_similarity = float(dialogue_vector @ centroids.clean)

    margin = load_settings().semantic_margin
    difference = attack_similarity - clean_similarity
    if difference >= margin:
        hint: SemanticHint = "negative"
    elif difference <= -margin:
        hint = "positive"
    else:
        hint = "neutral"

    app_logger.info(
        "semantic_hint_computed",
        hint=hint,
        attack_similarity=attack_similarity,
        clean_similarity=clean_similarity,
    )
    return hint
