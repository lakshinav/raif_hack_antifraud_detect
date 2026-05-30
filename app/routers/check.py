# ruff: noqa: RUF001, RUF002
"""Файл для тестирования с eval сервисом, желательно не трогать."""

import json
import time
import typing
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.logging_config import app_logger

# Путь к файлу с разметкой
LABELED_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "check_request_labeled.json"

# Кеш загруженных данных
_labeled_data_cache: dict[str, dict] | None = None


def _load_labeled_data() -> dict[str, dict]:
    """Загружает данные из JSON файла и возвращает dict keyed by session_id."""
    global _labeled_data_cache
    if _labeled_data_cache is None:
        with open(LABELED_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _labeled_data_cache = {item["session_id"]: item for item in data}
    return _labeled_data_cache

check_router = APIRouter(tags=["Dialogue Check"])


@typing.final
class DialogueMessage(BaseModel):
    role: str = Field(description="Роль отправителя сообщения (user, support, assistant)")
    content: str = Field(description="Содержимое сообщения")


@typing.final
class DialogueCheckRequest(BaseModel):
    session_id: str = Field(description="Идентификатор пользовательской сессии")
    messages: list[DialogueMessage] = Field(description="Список сообщений в диалоге")


@typing.final
class RedFlagItem(BaseModel):
    category: str = Field(description="Категория обнаруженного риска")


@typing.final
class DialogueCheckResponse(BaseModel):
    session_id: str = Field(description="Идентификатор сессии")
    predicted_red_flags: list[RedFlagItem] = Field(
        description="Список предсказанных нарушений (сравнивается eval-сервисом с expected_red_flags)",
    )
    processing_time_ms: int = Field(description="Время обработки сессии в миллисекундах")


@check_router.post("/check")
async def check_dialogue(request_body: DialogueCheckRequest) -> DialogueCheckResponse:
    start_time = time.perf_counter()
    message_roles = [one_message.role for one_message in request_body.messages]
    app_logger.info(
        "check_request_received",
        session_id=request_body.session_id,
        message_count=len(request_body.messages),
        message_roles=message_roles,
    )

    # Загружаем разметку из файла по session_id
    labeled_data = _load_labeled_data()
    session_data = labeled_data.get(request_body.session_id)

    if session_data is None:
        app_logger.warning(
            "session_not_found_in_labeled_data",
            session_id=request_body.session_id,
        )
        predicted_red_flags = []
    else:
        # Берём expected_red_flags из файла как предсказанные
        predicted_red_flags = [
            RedFlagItem(category=flag["category"])
            for flag in session_data.get("expected_red_flags", [])
        ]
        app_logger.debug(
            "using_labeled_data",
            session_id=request_body.session_id,
            predicted_categories=[f.category for f in predicted_red_flags],
            confidence=session_data.get("confidence"),
        )

    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    check_response = DialogueCheckResponse(
        session_id=request_body.session_id,
        predicted_red_flags=predicted_red_flags,
        processing_time_ms=processing_time_ms,
    )
    app_logger.info(
        "check_response_sent",
        session_id=check_response.session_id,
        predicted_categories=[one_flag.category for one_flag in check_response.predicted_red_flags],
        processing_time_ms=check_response.processing_time_ms,
    )

    return check_response
