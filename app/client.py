from __future__ import annotations

import typing

import httpx

from app.logging_config import app_logger
from app.settings import AppSettings, load_settings

OPENROUTER_BASE_URL: typing.Final = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_COMPLETIONS_URL: typing.Final = f"{OPENROUTER_BASE_URL}/chat/completions"


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
            "openrouter_client_configured",
            primary_model=self.app_settings.openrouter_primary_model or self.app_settings.openrouter_model,
            secondary_model=self.app_settings.openrouter_secondary_model,
            judge_model=self.app_settings.openrouter_judge_model,
            timeout_seconds=self.app_settings.openrouter_timeout_seconds,
            api_key_configured=bool(self.app_settings.openrouter_api_key),
        )

    async def request_completion(
        self,
        prompt_text: str,
        *,
        json_mode: bool = True,
        model_name: str | None = None,
    ) -> str | None:
        if not self.app_settings.openrouter_api_key:
            app_logger.warning("openrouter_request_skipped", reason="empty_api_key")
            return None

        selected_model = model_name or self.app_settings.openrouter_primary_model or self.app_settings.openrouter_model
        request_payload: dict[str, typing.Any] = {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0,
        }
        if json_mode:
            request_payload["response_format"] = {"type": "json_object"}

        app_logger.info(
            "openrouter_request_started",
            model=selected_model,
            json_mode=json_mode,
            prompt_text=prompt_text,
            prompt_length=len(prompt_text),
            request_payload=request_payload,
        )
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
            app_logger.info(
                "openrouter_response_received",
                model=selected_model,
                status_code=response.status_code,
            )
            response.raise_for_status()
        except httpx.HTTPError as http_error:
            app_logger.warning("openrouter_request_failed", model=selected_model, error=str(http_error))
            return None

        try:
            response_payload: object = response.json()
        except ValueError:
            app_logger.warning("openrouter_response_invalid_json", model=selected_model)
            return None
        app_logger.info(
            "openrouter_response_payload_received",
            model=selected_model,
            response_payload=response_payload,
        )

        completion_content = parse_completion_content(response_payload)
        if completion_content is None:
            app_logger.warning("openrouter_completion_missing", model=selected_model)
            return None

        app_logger.info(
            "openrouter_completion_received",
            model=selected_model,
            completion_content=completion_content,
            completion_length=len(completion_content),
        )
        return completion_content
