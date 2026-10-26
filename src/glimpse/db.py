"""Database module for storing prediction history using SQLite."""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data") / "glimpse.db"

def init_db():
    """Create the predictions table if it doesn't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            risk_score REAL NOT NULL,
            risk_label TEXT NOT NULL,
            top_feature TEXT,
            shap_value REAL,
            feature_value REAL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(risk_score: float, risk_label: str, contributions: list):
    """Save a prediction to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # گرفتن مهم‌ترین فیچر (اولین مورد در لیست)
    top_feat = contributions[0]["feature"] if contributions else "N/A"
    shap_val = contributions[0]["shap_value"] if contributions else 0.0
    feat_val = contributions[0]["feature_value"] if contributions else 0.0

    cursor.execute("""
        INSERT INTO predictions (timestamp, risk_score, risk_label, top_feature, shap_value, feature_value)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        risk_score,
        risk_label,
        top_feat,
        shap_val,
        feat_val
    ))
    conn.commit()
    conn.close()

def get_predictions(limit: int = 10):
    """Retrieve the latest N predictions."""
    if not DB_PATH.exists():
        return []
        
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]