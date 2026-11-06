"""SQLite persistence layer (stdlib sqlite3).

Stores invoice processing runs plus an invoice ledger used for duplicate
detection and split-PO cumulative tracking. JSON-heavy fields are serialized
to TEXT columns.
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from app import config
from app.models import RunRecord


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Create tables if they do not already exist."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_filename TEXT NOT NULL,
                extracted_fields TEXT NOT NULL,
                matched_po TEXT,
                decision TEXT NOT NULL,
                reasoning_trail TEXT NOT NULL,
                stages TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS invoice_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor TEXT,
                invoice_number TEXT,
                amount REAL,
                date TEXT,
                po_number TEXT
            )
            """
        )
        conn.commit()


def insert_run(record: RunRecord, db_path: Optional[str] = None) -> int:
    """Persist a RunRecord and return its assigned id."""
    timestamp = record.timestamp or datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO runs (
                timestamp, input_filename, extracted_fields, matched_po,
                decision, reasoning_trail, stages
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                record.input_filename,
                json.dumps(record.extracted_fields),
                json.dumps(record.matched_po) if record.matched_po is not None else None,
                record.decision,
                json.dumps(record.reasoning_trail),
                json.dumps(record.stages),
            ),
        )
        conn.commit()
        record.id = cur.lastrowid
        record.timestamp = timestamp
        return cur.lastrowid


def _row_to_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=row["id"],
        timestamp=row["timestamp"],
        input_filename=row["input_filename"],
        extracted_fields=json.loads(row["extracted_fields"]),
        matched_po=json.loads(row["matched_po"]) if row["matched_po"] is not None else None,
        decision=row["decision"],
        reasoning_trail=json.loads(row["reasoning_trail"]),
        stages=json.loads(row["stages"]),
    )


def list_runs(decision_filter: Optional[str] = None, db_path: Optional[str] = None) -> list:
    """Return runs newest-first, optionally filtered by decision."""
    with _connect(db_path) as conn:
        if decision_filter:
            rows = conn.execute(
                "SELECT * FROM runs WHERE decision = ? ORDER BY id DESC",
                (decision_filter,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
        return [_row_to_record(r) for r in rows]


def get_run(run_id: int, db_path: Optional[str] = None) -> Optional[RunRecord]:
    """Return a single run by id, or None if missing."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _row_to_record(row) if row is not None else None


def record_ledger_entry(
    vendor: str,
    invoice_number: str,
    amount: float,
    date: str,
    po_number: Optional[str],
    db_path: Optional[str] = None,
) -> int:
    """Append an invoice to the ledger and return its id."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO invoice_ledger (vendor, invoice_number, amount, date, po_number)
            VALUES (?, ?, ?, ?, ?)
            """,
            (vendor, invoice_number, amount, date, po_number),
        )
        conn.commit()
        return cur.lastrowid


def cumulative_for_po(po_number: str, db_path: Optional[str] = None) -> float:
    """Sum of all ledger amounts billed against a PO (0.0 if none)."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0.0) AS total FROM invoice_ledger WHERE po_number = ?",
            (po_number,),
        ).fetchone()
        return float(row["total"])


def is_duplicate(
    vendor: str,
    invoice_number: str,
    amount: float,
    date: str,
    db_path: Optional[str] = None,
) -> bool:
    """True if an identical ledger entry already exists."""
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM invoice_ledger
            WHERE vendor = ? AND invoice_number = ? AND amount = ? AND date = ?
            LIMIT 1
            """,
            (vendor, invoice_number, amount, date),
        ).fetchone()
        return row is not None
