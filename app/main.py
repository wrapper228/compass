from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler

from app.core.config import get_settings
from app.api.routes_health import router as health_router
from app.api.routes_chat import router as chat_router
from app.api.routes_files import router as files_router
from app.api.routes_memory import router as memory_router
from app.ui.routes_ui import router as ui_router
from app.db.base import Base
from app.db.session import engine
from app.db.session import SessionLocal
from app.services.memory import update_session_summary
from app.services.preferences import extract_preferences


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)

    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(files_router)
    app.include_router(memory_router)
    app.include_router(ui_router)

    @app.get("/")
    def root():
        return {"service": settings.app_name, "ok": True}

    @app.on_event("startup")
    def on_startup() -> None:
        # Простейшая инициализация таблиц для MVP (заменим на Alembic позже)
        Base.metadata.create_all(bind=engine)
        # Ночной cron (по желанию) — ежедневно в 03:30
        scheduler = BackgroundScheduler()

        def consolidate_job():
            db = SessionLocal()
            try:
                sessions = db.query(app.db.models.SessionModel).all()
                for s in sessions:
                    try:
                        update_session_summary(db, s)
                        extract_preferences(db, s)
                    except Exception:
                        continue
            finally:
                db.close()

        scheduler.add_job(consolidate_job, "cron", hour=3, minute=30)
        scheduler.start()

    return app


app = create_app()


