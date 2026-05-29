from __future__ import annotations

import typing

import httpx

from app.logging_config import app_logger
from app.settings import AppSettings, load_settings

OPENROUTER_CHAT_COMPLETIONS_URL: typing.Final = "https://openrouter.ai/api/v1/chat/completions"


def parse_completion_content(response_payload: object) -> str | None:
    if not isinstance(response_payload, dict):
        return None

    choices_payload = response_payload.get("choices")
    if not isinstance(choices_payload, list) or not choices_payload or not isinstance(choices_payload[0], dict):
        return None

    message_payload = choices_payload[0].get("message")
    if not isinstance(message_payload, dict):
        return None

    content_payload = message_payload.get("content")
    if isinstance(content_payload, str):
        return content_payload

    return None


@typing.final
class OpenRouterClient:
    def __init__(self, app_settings: AppSettings | None = None) -> None:
        self.app_settings = app_settings or load_settings()
        app_logger.info(
            "OpenRouter client configured: primary_model={} secondary_model={} judge_model={} "
            "timeout={} api_key_configured={}",
            self.app_settings.openrouter_primary_model or self.app_settings.openrouter_model,
            self.app_settings.openrouter_secondary_model,
            self.app_settings.openrouter_judge_model,
            self.app_settings.openrouter_timeout_seconds,
            bool(self.app_settings.openrouter_api_key),
        )

    async def request_completion(
        self,
        prompt_text: str,
        *,
        json_mode: bool = True,
        model_name: str | None = None,
    ) -> str | None:
        if not self.app_settings.openrouter_api_key:
            app_logger.warning("OpenRouter request skipped: OPENROUTER_API_KEY is empty")
            return None

        selected_model = model_name or self.app_settings.openrouter_primary_model or self.app_settings.openrouter_model
        request_payload: dict[str, typing.Any] = {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0,
        }
        if json_mode:
            request_payload["response_format"] = {"type": "json_object"}

        app_logger.info("OpenRouter request started: model={}", selected_model)
        try:
            async with httpx.AsyncClient(timeout=self.app_settings.openrouter_timeout_seconds) as httpx_connection:
                response = await httpx_connection.post(
                    OPENROUTER_CHAT_COMPLETIONS_URL,
                    headers={
                        "Authorization": f"Bearer {self.app_settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                )
            app_logger.info("OpenRouter response received: status_code={}", response.status_code)
            response.raise_for_status()
        except httpx.HTTPError as http_error:
            app_logger.warning("OpenRouter request failed: {}", http_error)
            return None

        try:
            response_payload: object = response.json()
        except ValueError:
            app_logger.warning("OpenRouter response is not valid JSON")
            return None

        completion_content = parse_completion_content(response_payload)
        if completion_content is None:
            app_logger.warning("OpenRouter response does not contain completion content")
            return None

        app_logger.info("OpenRouter completion parsed")
        return completion_content
