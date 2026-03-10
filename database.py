"""
SQLite backend for Restaurant Satisfaction Prediction app.
Handles connection, schema creation, and prediction logging.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import json
import sqlite3
from contextlib import contextmanager
from typing import Dict, Any, List

# Database file path - defaults to project folder
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "restaurant_satisfaction.db"))


def get_connection():
    """Create a new database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
    return conn


@contextmanager
def get_cursor(commit: bool = True):
    """Context manager for database cursor."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def init_schema():
    """Create tables if they don't exist."""
    with get_cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                input_data TEXT NOT NULL,
                predicted_probability REAL NOT NULL,
                predicted_class INTEGER NOT NULL,
                threshold_used REAL NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_class ON predictions(predicted_class)")


def log_prediction(
    model_name: str,
    input_data: Dict[str, Any],
    predicted_probability: float,
    predicted_class: int,
    threshold_used: float,
) -> int:
    """Insert a prediction record and return the new ID."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO predictions
                (model_name, input_data, predicted_probability, predicted_class, threshold_used)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                model_name,
                json.dumps(input_data),
                predicted_probability,
                predicted_class,
                threshold_used,
            ),
        )
        return cur.lastrowid


def get_recent_predictions(limit: int = 50) -> List[Dict]:
    """Fetch recent predictions for display."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, created_at, model_name, input_data,
                   predicted_probability, predicted_class, threshold_used
            FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def get_prediction_stats() -> Dict[str, Any]:
    """Get aggregate statistics."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) as total_predictions,
                SUM(CASE WHEN predicted_class = 1 THEN 1 ELSE 0 END) as high_satisfaction_count,
                SUM(CASE WHEN predicted_class = 0 THEN 1 ELSE 0 END) as low_satisfaction_count,
                AVG(predicted_probability) as avg_probability
            FROM predictions
        """)
        row = cur.fetchone()
        return dict(row) if row else {}
