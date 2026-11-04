# Оценка релевантности компаний поисковому запросу

Проект по курсу ML System Design в ИТМО. Автор — Александр Сафронов.

Система для DealRocket оценивает компании из готовой поисковой выдачи
по запросу пользователя, названию и описанию. Результат — **«подходит»**,
**«не подходит»** или **«недостаточно данных»** с кратким обоснованием.
Цель — ускорить ручной отбор компаний для B2B-продаж.

**Этап:** подготовка данных. В ДЗ №1 подготовлены
[дизайн-документ](docs/ml_system_design_doc.md) и окружение разработки.
В ДЗ №2 собран снимок запросов и выдачи, проведён [EDA](reports/eda.md)
и описаны этапы подготовки данных и моделей.
Следующие этапы: критерии и разметка, бейзлайн, LLM-классификатор и пилот.
Первичный поиск, сбор контактов и изменение интерфейса не входят в проект.

## Структура

```text
docs/ml_system_design_doc.md  — требования, архитектура и план оценки
reports/eda.md               — EDA: выводы и графики (reports/figures/)
reports/eda_summary.json     — агрегаты EDA
scripts/                     — выгрузка данных и EDA
data/raw/                    — локальные выгрузки, не входят в Git
pyproject.toml               — версия Python и настройки Ruff
requirements-dev.txt         — инструменты разработки
requirements-eda.txt         — зависимости выгрузки и EDA
.pre-commit-config.yaml      — проверки перед коммитом
.markdownlint-cli2.yaml      — правила проверки Markdown
```

## Подготовка окружения

Python 3.11+ и Git. Команды для macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
pre-commit install
```

## Проверки

```bash
pre-commit run --all-files
ruff check .
ruff format --check .
```

Pre-commit проверяет Markdown, YAML, TOML, конфликтные маркеры, оформление файлов
и ограничивает размер новых файлов до 500 КБ. При первом запуске он устанавливает
изолированное окружение Markdown-линтера; требуется доступ к сети.
Ruff проверяет и форматирует Python-код. Автоисправление: `ruff check --fix .`
и `ruff format .`. Версии Ruff в `requirements-dev.txt`
и `.pre-commit-config.yaml` должны совпадать.

## Данные и EDA

Выгрузки содержат данные DealRocket и хранятся только локально в `data/raw/`.
Для доступа нужны токен API и сессия админки DealRocket; их выдаёт владелец сервиса.

```bash
python3.12 -m venv .venv-eda
.venv-eda/bin/pip install -r requirements-eda.txt
export DEALROCKET_ADMIN_COOKIE='...'   # cookie сессии админки
export DEALROCKET_TOKEN='...'          # токен API
.venv-eda/bin/python scripts/export_admin_dialogs.py
.venv-eda/bin/python scripts/fetch_dealrocket_results.py --source admin --queries 80
.venv-eda/bin/python scripts/eda.py --excel <книга запросов менеджеров>.xlsx
```

Выгрузка выдачи повторяет поиск заново и пишет результат после каждого запроса:
прерванный запуск продолжается с места остановки. Повторная выгрузка даёт
новый снимок, потому что база и справочники DealRocket меняются.

## ДЗ №1

[Условия задания](https://gist.github.com/pacifikus/47424379293e0874859354ecd130ba13).
Ветка — `hw_1`, PR — в `main`. Ментору предоставляется доступ к репозиторию.
После проверки автора — ревью ментора и двух студентов, исправления и слияние.
Для получения оценки также требуется проверить работы двух студентов.

## ДЗ №2

[Условия задания](https://gist.github.com/pacifikus/bada0e4b597852e4bb50512daf451929).
Ветка — `hw_2`. Результаты: [EDA](reports/eda.md) и раздел 3 дизайн-документа.
