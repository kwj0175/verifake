from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Docker Compose environment 또는 .env에서 주입
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://user:password@db:3306/verifake_db",
)

# SQLite 사용 시(로컬 개발/테스트) check_same_thread=False 설정
# FastAPI 백그라운드 태스크는 별도 스레드에서 실행되므로 필수
_connect_args: dict = {}
if DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,   # 끊긴 커넥션 자동 감지
    pool_recycle=3600,    # MySQL wait_timeout 대응
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
