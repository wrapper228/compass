# compass

MVP сервис на FastAPI для рефлексивного ассистента.

Важно: сейчас чат отвечает «заглушкой» (без реальной LLM). Retrieval полностью локальный: чанки индексируются в Faiss + BM25, rerank выполняет кросс-энкодер. Для работы HyDE/переформулирования и финального ответа по-прежнему нужен совместимый Chat Completions API (см. ниже).

## Что понадобится
- Установленный Python 3.11+
- Git
- Терминал (Windows: PowerShell или Git Bash)

## Где брать и куда платить (ключи)
- LLM (опционально, но нужен для HyDE/переформулирования и финального ответа):
  - Совместимый Chat Completions API — OpenRouter, OpenAI и т.п.
  - Нужен API‑ключ и хотя бы одна модель (`LLM_MODEL_FAST`/`LLM_MODEL_SMART`).
- Индексация и поиск работают локально и не требуют внешних сервисов — модели `BAAI/bge-small-en-v1.5` и `BAAI/bge-reranker-large` скачиваются автоматически с Hugging Face (интернет всё равно нужен при первом запуске).

Без LLM ключей сервис запустится, ingestion и поиск будут работать, но ответы останутся заглушкой.

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
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
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

# Retrieval (локальные индексы)
RETRIEVAL_ENABLED=true
INDICES_PATH=./data/indices
DENSE_MODEL_NAME=BAAI/bge-small-en-v1.5
RERANK_MODEL_NAME=BAAI/bge-reranker-large

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
- Веб‑интерфейс чата: открой в браузере http://localhost:8000/chat

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
Ответ `status=done` дополнен сводкой (`documents`, `chunks`, `indexed`). Индексация идёт в локальные индексы Faiss/BM25.

- WebSocket стриминг чата (стаб):
Установите любой WS‑клиент (например, [`websocat`](https://github.com/vi/websocat) или `wscat`). Пример с websocat:
```
websocat ws://localhost:8000/ws/chat
Привет
```
Вы увидите поток токенов и в конце `[DONE]`.

## Где данные
- База данных по умолчанию: `./data/compass.db` (SQLite)
- Локальные индексы (Faiss/BM25, метаданные): `./data/indices/`
- Загрузки исходных архивов: `./data/uploads/`
- Временные файлы ingestion: `./data/tmp/<job_id>/` (в т.ч. `ingest_result.json` с итогами)

## Управление индексами
- Полный пересчёт (например, раз в месяц или после правок в моделях):
  ```
  python3 -c "from app.services.retrieval.index_manager import HybridIndexManager; HybridIndexManager().rebuild()"
  ```
- Очистить индекс можно, удалив каталог `INDICES_PATH` — новые загрузки создадут его автоматически.

## Тесты
```
python3 -m pytest -q
```
Если `pytest` не установлен, выполните `pip install pytest`.

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
- Готово: каркас FastAPI, REST/WS чат (стабы), гибридный Retrieval (Faiss + BM25 + кросс-энкодер) с HyDE/self-check, загрузка zip, фоновая распаковка/чанкинг, локальная БД для сессий/сообщений.
- В планах: подключение реальной LLM по API, расширение памяти и проактивности.

