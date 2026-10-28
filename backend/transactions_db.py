import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "artifacts" / "transactions.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_data TEXT NOT NULL,
                amount REAL NOT NULL,
                prediction INTEGER NOT NULL,
                label TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                threshold REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def save_transaction(transaction_data, result):
    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO transactions (
                transaction_data,
                amount,
                prediction,
                label,
                fraud_probability,
                threshold,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                json.dumps(transaction_data),
                float(transaction_data["Amount"]),
                int(result["prediction"]),
                str(result["label"]),
                float(result["fraud_probability"]),
                float(result["threshold"]),
                created_at,
            ),
        )
        return cursor.lastrowid


def get_transaction_history(limit=100):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM transactions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "transaction_data": json.loads(row["transaction_data"]),
            "amount": row["amount"],
            "prediction": row["prediction"],
            "label": row["label"],
            "fraud_probability": row["fraud_probability"],
            "threshold": row["threshold"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]