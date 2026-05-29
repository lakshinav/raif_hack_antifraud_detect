from __future__ import annotations

import typing

import pydantic

from app.models import RiskCategory


@typing.final
class DetectionResult(pydantic.BaseModel):
    """Результат первого chain: есть ли в диалоге атака."""

    is_attack: bool = pydantic.Field(description="True, если в диалоге есть попытка атаки/манипуляции.")
    confidence: float = pydantic.Field(
        ge=0.0,
        le=1.0,
        description="Калиброванная уверенность в решении is_attack (0.0–1.0).",
    )
    reason: str = pydantic.Field(description="Краткое обоснование решения на русском языке.")


@typing.final
class ClassificationResult(pydantic.BaseModel):
    """Результат второго chain: к какой категории относится атака."""

    category: RiskCategory = pydantic.Field(description="Одна из шести категорий атаки.")
    confidence: float = pydantic.Field(
        ge=0.0,
        le=1.0,
        description="Калиброванная уверенность в выбранной категории (0.0–1.0).",
    )
    reason: str = pydantic.Field(description="Краткое обоснование выбора категории на русском языке.")
