import os
from pathlib import Path

# Используем отдельную тестовую БД, чтобы не трогать рабочую
TEST_DIR = Path("test_artifacts")
TEST_DIR.mkdir(exist_ok=True)
db_path = TEST_DIR / "test.db"
if db_path.exists():
    db_path.unlink()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(TEST_DIR / 'test.db').as_posix()}")
os.environ.setdefault("RETRIEVAL_ENABLED", "false")
indices_dir = TEST_DIR / "indices"
if indices_dir.exists():
    import shutil

    shutil.rmtree(indices_dir)
os.environ.setdefault("INDICES_PATH", str(indices_dir))


