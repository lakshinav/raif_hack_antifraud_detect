from __future__ import annotations

import typing

import pydantic
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.client import OPENROUTER_BASE_URL
from app.settings import load_settings

SchemaT = typing.TypeVar("SchemaT", bound=pydantic.BaseModel)


def build_chat_model(model: str | None = None) -> ChatOpenAI:
    """Создаёт ChatOpenAI, настроенный на OpenRouter (OpenAI-совместимый API)."""
    app_settings = load_settings()
    selected_model = model or app_settings.openrouter_primary_model or app_settings.openrouter_model
    return ChatOpenAI(
        model=selected_model,
        base_url=OPENROUTER_BASE_URL,
        api_key=pydantic.SecretStr(app_settings.openrouter_api_key),
        temperature=0,
        timeout=app_settings.openrouter_timeout_seconds,
    )


def build_structured_llm(
    schema: type[SchemaT],
    model: str | None = None,
) -> Runnable[typing.Any, SchemaT]:
    """ChatOpenAI с принудительным structured output по Pydantic-схеме."""
    chat_model = build_chat_model(model)
    return typing.cast(
        "Runnable[typing.Any, SchemaT]",
        chat_model.with_structured_output(schema),
    )
