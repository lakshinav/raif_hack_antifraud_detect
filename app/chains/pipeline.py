from __future__ import annotations

import functools
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


@functools.lru_cache(maxsize=8)
def build_pipeline(model: str | None = None) -> Runnable[str, PipelineResult]:
    """Композиция: detection chain -> (если атака) classification chain. Вход — текст диалога.

    Мемоизируется по `model`: цепочки и ChatOpenAI-клиенты строятся один раз на процесс, а не на
    каждый запрос /check. Кэш безопасен — построенный Runnable без состояния между вызовами.
    """
    detection_chain = build_detection_chain(model)
    classification_chain = build_classification_chain(model)

    async def _run(dialogue: str) -> PipelineResult:
        detection = await detection_chain.ainvoke(dialogue)
        if not detection.is_attack:
            return PipelineResult(
                is_attack=False,
                category=None,
                detection_confidence=detection.confidence,
                classification_confidence=None,
                reason=detection.reason,
            )
        classification = await classification_chain.ainvoke(dialogue)
        # Классификатор может «передумать» и вернуть clean — это вето на ложное срабатывание детектора.
        is_attack = classification.category != "clean"
        return PipelineResult(
            is_attack=is_attack,
            category=typing.cast("RiskCategory", classification.category) if is_attack else None,
            detection_confidence=detection.confidence,
            classification_confidence=classification.confidence,
            reason=classification.reason,
        )

    return RunnableLambda(_run)
