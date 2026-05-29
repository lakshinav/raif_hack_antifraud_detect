from __future__ import annotations

import typing

import pydantic
from langchain_core.runnables import Runnable, RunnableLambda

from app.chains.classification_chain import build_classification_chain
from app.chains.detection_chain import build_detection_chain
from app.models import RiskCategory


@typing.final
class PipelineResult(pydantic.BaseModel):
    """Итог композиции: есть ли атака и какая (если есть)."""

    is_attack: bool
    category: RiskCategory | None = None
    detection_confidence: float
    classification_confidence: float | None = None
    reason: str


def build_pipeline(model: str | None = None) -> Runnable[str, PipelineResult]:
    """Композиция: detection chain -> (если атака) classification chain. Вход — текст диалога."""
    detection_chain = build_detection_chain(model)
    classification_chain = build_classification_chain(model)

    def _run(dialogue: str) -> PipelineResult:
        detection = detection_chain.invoke(dialogue)
        if not detection.is_attack:
            return PipelineResult(
                is_attack=False,
                category=None,
                detection_confidence=detection.confidence,
                classification_confidence=None,
                reason=detection.reason,
            )
        classification = classification_chain.invoke(dialogue)
        return PipelineResult(
            is_attack=True,
            category=classification.category,
            detection_confidence=detection.confidence,
            classification_confidence=classification.confidence,
            reason=classification.reason,
        )

    return RunnableLambda(_run)
