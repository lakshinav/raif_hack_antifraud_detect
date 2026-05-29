from __future__ import annotations

import typing

from langchain_core.runnables import Runnable, RunnableLambda

from app.chains.llm import build_structured_llm
from app.chains.prompts import DETECTION_PROMPT
from app.chains.schemas import DetectionResult


def build_detection_chain(model: str | None = None) -> Runnable[str, DetectionResult]:
    """Chain 1: по тексту диалога (`role: content` построчно) определяет, есть ли атака."""
    prepare: RunnableLambda[str, dict[str, str]] = RunnableLambda(lambda dialogue: {"dialogue": dialogue})
    return typing.cast(
        "Runnable[str, DetectionResult]",
        prepare | DETECTION_PROMPT | build_structured_llm(DetectionResult, model),
    )
