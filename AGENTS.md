# Agent Guide

This repository is a FastAPI service for detecting red-flag categories in bank-support dialogues.
Keep changes focused on the evaluator contract and avoid broad refactors during hackathon work.

## Core Commands

- Install/update dependencies: `just setup`
- Run local server: `just dev-local`
- Run Docker dev server: `just dev-docker`
- Run code audit: `just audit`
- Run tests: `just test`

`just audit` is the main quality gate. It runs Ruff fix, Ruff format, mypy, and flake8/COP for `app/`.
Run `just audit` and `just test` before considering implementation work complete.

If `uv` needs access to its cache outside the sandbox, run the same commands with the required approval instead of changing
project configuration.

## API Contract

- Do not break `POST /check`.
- Request shape:
  - `session_id: str`
  - `messages: list[{role: str, content: str}]`
- Response shape:
  - `session_id: str`
  - `predicted_red_flags: list[{category: str}]`
  - `processing_time_ms: int`
- Keep `app/routers/check.py` as a thin wrapper. The detection pipeline should live in `process_risk_detection`.
- Current task rule: one dialogue maps to at most one risk label.
- Internal `clean` means no public red flag. Return `predicted_red_flags: []` for clean dialogues.

## Labels

Allowed risk labels:

- `policy_manipulation`
- `adversarial_attack`
- `identity_deception`
- `transaction_coercion`
- `information_extraction`
- `scope_violation`

Internal non-risk label:

- `clean`

Do not emit labels outside this set. Do not return `clean` in the public API response.

## Dataset Notes

`train.json` has 50 sessions with this shape:

- `session_id`
- `messages`
- `expected_red_flags`

Observed from the current file:

- 26 sessions have empty `expected_red_flags` and should be treated as `clean`.
- Each risk class appears 4 times.
- No session has more than one expected label.
- Roles are `chatbot`, `user`, and `support`.
- Dialogue length ranges from 5 to 36 messages, average about 15 messages.

This supports the current single-label classifier design: `1 dialogue = 1 label`.

## LLM Pipeline

- OpenRouter settings are managed in `app/settings.py` with `pydantic-settings`.
- The OpenRouter client is in `app/client.py`.
- Runtime classification logic is in `app/models.py`.
- Prompt output must be structured JSON: `{"category": "label_name"}`.
- The prompt must handle Russian, English, and mixed-language dialogues.
- Keep average `/check` latency around the evaluator target, roughly 5 seconds.
- Prefer fast, deterministic classification settings. Current default model is `google/gemini-3.1-flash-lite`.
- Log outbound LLM requests, response status, and parse failures, but never log API keys.

## Coding Rules

- Use Python 3.11 and `uv`.
- Keep strict typing compatible with `mypy --strict`.
- Follow the repository's Ruff and flake8/COP rules.
- Prefer small functions with explicit names.
- Avoid changing evaluator-facing schema unless the task explicitly requires it.
- Do not add secrets to the repo. Use `.env` locally and `.env.example` for non-secret defaults.
- Preserve existing user changes in the working tree.

## Practical Workflow

1. Read `app/routers/check.py`, `app/models.py`, `app/client.py`, and `app/settings.py` before changing classifier behavior.
2. If modifying labels or prompt behavior, check `train.json` examples for the affected category.
3. Keep `/check` stable and put pipeline changes inside `process_risk_detection` or helpers it calls.
4. Run `just audit`.
5. Run `just test`.
