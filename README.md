# compass

MVP сервис на FastAPI для рефлексивного ассистента.

Важно: сейчас чат отвечает «заглушкой» (без реальной LLM). Индексация документов и поиск по ним включаются автоматически, если задать ключи Embeddings и Qdrant (см. ниже). Это достаточно, чтобы проверить загрузку данных, пайплайны и стриминг.

## Что понадобится
- Установленный Python 3.11+
- Git
- Терминал (Windows: PowerShell или Git Bash)

## Где брать и куда платить (ключи)
- Embeddings (обязательно для индексации и поиска):
  - Вариант A: [OpenRouter](https://openrouter.ai) — зарегистрируйтесь, пополните баланс/привяжите карту, создайте API‑ключ.
  - Вариант B: [OpenAI](https://platform.openai.com) — включите оплату (billing), создайте API‑ключ.
  - Пример модели: `text-embedding-3-small` (OpenAI) или аналог у провайдера в формате OpenAI API.
- Qdrant (векторная БД, опционально):
  - [Qdrant Cloud](https://cloud.qdrant.io) — создайте серверless‑кластер (часто есть бесплатный тариф), получите URL и API‑ключ.

Оплата идёт провайдерам (OpenRouter/OpenAI и Qdrant). Без ключей всё запустится, но индексации/поиска не будет.

## Запуск — пошагово (Windows/macOS/Linux)
1) Клонируйте репозиторий и зайдите в папку
```
git clone <URL_ВАШЕГО_РЕПО>
cd compass
```

2) Создайте виртуальное окружение и активируйте его
- Windows (PowerShell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```
- macOS/Linux:
```
python -m venv .venv
source .venv/bin/activate
```

3) Установите зависимости
```
pip install -r requirements.txt
```

4) (Опционально, но рекомендовано) Создайте файл `.env` в корне проекта со своими переменными
Пример содержимого:
```
# БД (по умолчанию — SQLite файл в ./data)
DATABASE_URL=sqlite:///./data/compass.db

# Embeddings в стиле OpenAI API (для индексации и поиска)
EMBEDDINGS_API_BASE=https://api.openai.com/v1
EMBEDDINGS_API_KEY=sk-...ваш_ключ...
EMBEDDINGS_MODEL=text-embedding-3-small

# Qdrant Cloud (опционально)
QDRANT_URL=https://...your-qdrant-url...
QDRANT_API_KEY=...ваш_ключ...
QDRANT_COLLECTION=chunks

# LLM (OpenAI/OpenRouter совместимый Chat Completions)
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=sk-...ваш_ключ...
LLM_MODEL_FAST=gpt-3.5-turbo
LLM_MODEL_SMART=gpt-4o
ETHICS_MODE=experimental
```

5) Запустите сервер разработки
```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

6) Проверьте, что сервис жив
```
GET http://localhost:8000/api/health
```
Должно вернуть: `{ "ok": true }`

## Как пользоваться (примеры)
- HTTP чат (пока заглушка ответа):
```
curl -s -X POST http://localhost:8000/api/chat/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Привет"}
    ]
  }'
```

- Загрузка zip с данными (любой набор папок/файлов внутри):
```
curl -s -X POST http://localhost:8000/api/files/upload \
  -F zip_file=@/полный/путь/к/архиву.zip
```
Ответ содержит `job_id`. Статус обработки:
```
curl -s http://localhost:8000/api/ingest/<job_id>
```
Если настроены Embeddings и Qdrant — автоматически произойдёт индексация.

- WebSocket стриминг чата (стаб):
Установите любой WS‑клиент (например, [`websocat`](https://github.com/vi/websocat) или `wscat`). Пример с websocat:
```
websocat ws://localhost:8000/ws/chat
Привет
```
Вы увидите поток токенов и в конце `[DONE]`.

## Где данные
- База данных по умолчанию: `./data/compass.db` (SQLite)
- Загрузки: `./data/uploads/`
- Временные файлы ingestion/чанки: `./data/tmp/` (в т.ч. `chunks.json` для каждой задачи)

## Тесты
```
pytest -q
```

## Docker (опционально)
Собрать и запустить локально:
```
docker build -t compass .
docker run --rm -p 8000:8000 --env-file .env compass
```

## Типовые проблемы и решения
- «403/401 при индексации» — проверьте `EMBEDDINGS_API_KEY` и `EMBEDDINGS_API_BASE`.
- «Нет индексации» — не заданы переменные для Embeddings/Qdrant; это опционально, сервис всё равно работает.
- «Не активируется venv на Windows» — запускайте PowerShell от имени администратора или разрешите выполнение скриптов: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- «Падает парсинг PDF/HTML/DOCX» — убедитесь, что установлены зависимости (см. requirements), и файлы не защищены.

## Деплой в облако (Render / Railway)
### Render (с Docker)
1. Загрузите репозиторий в GitHub
2. На `render.com` создайте Web Service → выбор репо → Render прочитает `render.yaml`
3. Добавьте переменные окружения (секция Environment) из `.env`
4. Деплой стартует автоматически, проверка: `GET https://<your-app>.onrender.com/api/health`

### Railway (с Procfile)
1. Загрузите репозиторий в GitHub
2. На `railway.app` создайте проект, подключите репозиторий
3. Railway обнаружит `Procfile` и запустит веб процесс
4. Добавьте переменные окружения из `.env`

## Что готово сейчас и чего ждать дальше
- Готово: каркас FastAPI, REST/WS чат (стабы), загрузка zip, фоновая распаковка/чанкинг, опциональная индексация в Qdrant, БД для сессий/сообщений.
- В планах: подключение реальной LLM по API, RAG‑контекст в ответ, многоуровневая память и проактивность.

