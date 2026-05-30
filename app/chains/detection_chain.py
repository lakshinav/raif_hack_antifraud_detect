from __future__ import annotations

import typing

from langchain_core.runnables import Runnable, RunnableLambda

from app.chains.llm import build_structured_llm
from app.chains.prompts import DETECTION_PROMPT
from app.chains.schemas import DetectionResult
from app.chains.semantic import classify_semantic
from app.settings import load_settings


async def prepare_detection_input(dialogue: str) -> dict[str, str]:
    """Готовит вход промпта; при включённом флаге добавляет семантическую подсказку."""
    if not load_settings().enable_semantic_classifier:
        return {"dialogue": dialogue}

    semantic_hint = await classify_semantic(dialogue)
    if semantic_hint is None:
        # Сигнал недоступен — оставляем partial-дефолт промпта ("нет данных").
        return {"dialogue": dialogue}

    return {"dialogue": dialogue, "semantic_hint": semantic_hint}


def build_detection_chain(model: str | None = None) -> Runnable[str, DetectionResult]:
    """Chain 1: по тексту диалога (`role: content` построчно) определяет, есть ли атака."""
    prepare: RunnableLambda[str, dict[str, str]] = RunnableLambda(prepare_detection_input)
    return typing.cast(
        "Runnable[str, DetectionResult]",
        prepare | DETECTION_PROMPT | build_structured_llm(DetectionResult, model),
    )
