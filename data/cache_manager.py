"""캐시 관리자 (SQLite 기반, 선택적 Redis)"""
from __future__ import annotations

import json
import logging
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).parent / "cache" / "stock_cache.db"


class CacheManager:
    """SQLite + 선택적 인메모리 캐시"""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    dtype TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # ------------------------------------------------------------------ #
    # Dict / Any 캐시
    # ------------------------------------------------------------------ #

    def get(self, key: str) -> Any | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value, dtype, expires_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        value, dtype, expires_at = row
        if time.time() > expires_at:
            self.delete(key)
            return None
        if dtype == "json":
            return json.loads(value)
        if dtype == "pickle":
            return pickle.loads(value)
        return value

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        expires_at = time.time() + ttl
        try:
            serialized = json.dumps(value, default=str).encode()
            dtype = "json"
        except (TypeError, ValueError):
            serialized = pickle.dumps(value)
            dtype = "pickle"

        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, dtype, expires_at) VALUES (?, ?, ?, ?)",
                (key, serialized, dtype, expires_at),
            )

    def delete(self, key: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def clear_expired(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM cache WHERE expires_at < ?", (time.time(),))

    # ------------------------------------------------------------------ #
    # DataFrame 캐시 (Parquet)
    # ------------------------------------------------------------------ #

    def get_df(self, key: str) -> pd.DataFrame | None:
        data = self.get(f"df:{key}")
        if data is None:
            return None
        try:
            import io, json
            return pd.read_json(io.StringIO(data))
        except Exception:
            return None

    def set_df(self, key: str, df: pd.DataFrame, ttl: int = 300) -> None:
        try:
            self.set(f"df:{key}", df.to_json(), ttl=ttl)
        except Exception:
            pass
