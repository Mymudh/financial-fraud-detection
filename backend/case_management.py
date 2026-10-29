import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "artifacts" / "fraud_cases.db"

router = APIRouter(prefix="/cases", tags=["Fraud Case Management"])


def get_connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fraud_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                prediction TEXT NOT NULL,
                case_status TEXT NOT NULL DEFAULT 'OPEN',
                analyst_feedback TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


class CaseCreate(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=100)
    fraud_probability: float = Field(ge=0, le=1)
    prediction: str = Field(min_length=1, max_length=50)


class CaseUpdate(BaseModel):
    case_status: str | None = None
    analyst_feedback: str | None = Field(default=None, max_length=100)
    notes: str | None = Field(default=None, max_length=2000)


ALLOWED_STATUSES = {
    "OPEN",
    "UNDER_REVIEW",
    "CONFIRMED_FRAUD",
    "LEGITIMATE",
    "CLOSED",
}


def serialize_case(row):
    return dict(row)


initialize_database()


@router.post("")
def create_case(data: CaseCreate):
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO fraud_cases (
                transaction_id,
                fraud_probability,
                prediction,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                data.transaction_id.strip(),
                data.fraud_probability,
                data.prediction.strip(),
                now,
                now,
            ),
        )

        case_id = cursor.lastrowid

        row = connection.execute(
            "SELECT * FROM fraud_cases WHERE id = ?",
            (case_id,),
        ).fetchone()

    return {"message": "Fraud case created.", "case": serialize_case(row)}


@router.get("")
def list_cases(
    case_status: str | None = None,
    limit: int = 100,
):
    if not 1 <= limit <= 500:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 500.",
        )

    query = "SELECT * FROM fraud_cases"
    parameters = []

    if case_status:
        normalized_status = case_status.upper()

        if normalized_status not in ALLOWED_STATUSES:
            raise HTTPException(
                status_code=400,
                detail="Invalid case status.",
            )

        query += " WHERE case_status = ?"
        parameters.append(normalized_status)

    query += " ORDER BY id DESC LIMIT ?"
    parameters.append(limit)

    with get_connection() as connection:
        rows = connection.execute(
            query,
            parameters,
        ).fetchall()

    return {
        "count": len(rows),
        "cases": [serialize_case(row) for row in rows],
    }


@router.get("/{case_id}")
def get_case(case_id: int):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM fraud_cases WHERE id = ?",
            (case_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Case not found.")

    return {"case": serialize_case(row)}


@router.patch("/{case_id}")
def update_case(case_id: int, data: CaseUpdate):
    updates = data.model_dump(exclude_unset=True)

    if not updates:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one field to update.",
        )

    if "case_status" in updates:
        if updates["case_status"] is None:
            raise HTTPException(
                status_code=400,
                detail="Case status cannot be null.",
            )

        updates["case_status"] = updates["case_status"].upper()

        if updates["case_status"] not in ALLOWED_STATUSES:
            raise HTTPException(
                status_code=400,
                detail="Invalid case status.",
            )

    if "analyst_feedback" in updates:
        feedback = updates["analyst_feedback"]

        if feedback is not None:
            feedback = feedback.strip()

            if not feedback:
                raise HTTPException(
                    status_code=400,
                    detail="Analyst feedback cannot be empty.",
                )

            updates["analyst_feedback"] = feedback

    updates["updated_at"] = datetime.now(timezone.utc).isoformat()

    assignments = ", ".join(
        f"{column} = ?" for column in updates
    )

    values = list(updates.values())
    values.append(case_id)

    with get_connection() as connection:
        existing = connection.execute(
            "SELECT id FROM fraud_cases WHERE id = ?",
            (case_id,),
        ).fetchone()

        if existing is None:
            raise HTTPException(
                status_code=404,
                detail="Case not found.",
            )

        connection.execute(
            f"UPDATE fraud_cases SET {assignments} WHERE id = ?",
            values,
        )

        row = connection.execute(
            "SELECT * FROM fraud_cases WHERE id = ?",
            (case_id,),
        ).fetchone()

    return {
        "message": "Fraud case updated.",
        "case": serialize_case(row),
    }


@router.get("/statistics/summary")
def case_statistics():
    with get_connection() as connection:
        total = connection.execute(
            "SELECT COUNT(*) FROM fraud_cases"
        ).fetchone()[0]

        rows = connection.execute(
            """
            SELECT case_status, COUNT(*) AS count
            FROM fraud_cases
            GROUP BY case_status
            """
        ).fetchall()

        feedback_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM fraud_cases
            WHERE analyst_feedback IS NOT NULL
            """
        ).fetchone()[0]

    return {
        "total_cases": total,
        "feedback_count": feedback_count,
        "status_counts": {
            row["case_status"]: row["count"]
            for row in rows
        },
    }