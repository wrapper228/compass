import os
from pathlib import Path

# Используем отдельную тестовую БД, чтобы не трогать рабочую
TEST_DIR = Path("test_artifacts")
TEST_DIR.mkdir(exist_ok=True)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(TEST_DIR / 'test.db').as_posix()}")


