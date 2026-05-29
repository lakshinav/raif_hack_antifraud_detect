# https://just.systems/man/en/

set dotenv-load

default:
    @just --list

# Подготовка venv, установка зависимостей и pre-commit хуков
setup:
    uv sync
    uv run pre-commit install

# Запуск dev-сервера в Docker с автоматическим отслеживанием изменений (watch)
dev-docker:
    docker compose -f docker-compose.local.yml up --watch

# Запуск dev-сервера локально на хосте (без Docker)
dev-local:
    uv run uvicorn app.main:application --host 127.0.0.1 --port 8787 --reload

# Полный аудит кода с автоисправлением: Ruff check/fix + Ruff format + mypy + flake8 COP
audit:
    uv run ruff check app/ --fix
    uv run ruff format app/
    uv run mypy --ignore-missing-imports app/
    uv run flake8 app/

# Запуск автоматических тестов (pytest)
test:
    uv run pytest

# Локальная оценка качества на train-датасете через запущенный локальный /check endpoint
eval-local:
    uv run python scripts/evaluate.py --dataset-path data/train.json

# Оценка любого /check endpoint (пример: just eval-endpoint http://127.0.0.1:8787/check)
eval-endpoint endpoint_url:
    uv run python scripts/evaluate.py --dataset-path data/train.json --endpoint-url {{endpoint_url}}

# Валидация модели с расчетом macro F1 score (запускает pipeline локально)
validate:
    uv run python scripts/validate.py --dataset-path data/train.json

# Валидация с экспортом отчета в JSON
validate-with-report output_path="validation_report.json":
    uv run python scripts/validate.py --dataset-path data/train.json --output-path {{output_path}}

# Создание релизного тега и отправка в удаленный репозиторий (например, just release 1.0.0)
release version:
    git tag v{{version}}
    git push origin v{{version}}
