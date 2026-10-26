"""SQLite database for prediction history logging."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data") / "glimpse.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Create the predictions table if it does not exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                risk_score REAL NOT NULL
            )
            """
        )
        conn.commit()


def log_prediction(risk_score: float) -> None:
    """Insert a single prediction into the history table."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO predictions (timestamp, risk_score) VALUES (?, ?)",
            (datetime.utcnow().isoformat(timespec="seconds"), float(risk_score)),
        )
        conn.commit()


def get_history(limit: int = 100) -> list[dict]:
    """Return the most recent predictions as a list of dicts."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, timestamp, risk_score FROM predictions ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]