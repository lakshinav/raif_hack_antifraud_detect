from __future__ import annotations

import typing

from langchain_core.runnables import Runnable, RunnableLambda

from app.chains.llm import build_structured_llm
from app.chains.prompts import CLASSIFICATION_PROMPT
from app.chains.schemas import ClassificationResult


def build_classification_chain(model: str | None = None) -> Runnable[str, ClassificationResult]:
    """Chain 2: по тексту диалога с подтверждённой атакой определяет её категорию."""
    prepare: RunnableLambda[str, dict[str, str]] = RunnableLambda(lambda dialogue: {"dialogue": dialogue})
    return typing.cast(
        "Runnable[str, ClassificationResult]",
        prepare | CLASSIFICATION_PROMPT | build_structured_llm(ClassificationResult, model),
    )
