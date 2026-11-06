"""Tests for app.storage round-trips and ledger helpers."""

from app import storage
from app.models import Decision, RunRecord, StageStatus


def test_insert_list_get_roundtrip(tmp_path):
    db = str(tmp_path / "test_runs.db")
    storage.init_db(db)

    record = RunRecord(
        input_filename="invoice_001.pdf",
        extracted_fields={"vendor": "Acme", "amount": 100.5},
        matched_po={"po_number": "PO-1", "amount": 100.0},
        decision=Decision.APPROVE.value,
        reasoning_trail=["matched PO-1", "within tolerance"],
        stages=[{"name": "extract", "status": StageStatus.OK.value, "confidence": 0.9, "data": {}}],
    )

    run_id = storage.insert_run(record, db)
    assert run_id is not None
    assert record.id == run_id

    all_runs = storage.list_runs(db_path=db)
    assert len(all_runs) == 1

    fetched = storage.get_run(run_id, db)
    assert fetched is not None
    assert fetched.input_filename == "invoice_001.pdf"
    assert fetched.extracted_fields == {"vendor": "Acme", "amount": 100.5}
    assert fetched.matched_po == {"po_number": "PO-1", "amount": 100.0}
    assert fetched.decision == Decision.APPROVE.value
    assert fetched.reasoning_trail == ["matched PO-1", "within tolerance"]
    assert fetched.stages[0]["status"] == StageStatus.OK.value

    approved = storage.list_runs(decision_filter=Decision.APPROVE.value, db_path=db)
    assert len(approved) == 1
    rejected = storage.list_runs(decision_filter=Decision.REJECT.value, db_path=db)
    assert len(rejected) == 0


def test_is_duplicate(tmp_path):
    db = str(tmp_path / "test_runs.db")
    storage.init_db(db)

    assert storage.is_duplicate("Acme", "INV-1", 100.0, "2024-01-01", db) is False
    storage.record_ledger_entry("Acme", "INV-1", 100.0, "2024-01-01", "PO-1", db)
    assert storage.is_duplicate("Acme", "INV-1", 100.0, "2024-01-01", db) is True


def test_cumulative_for_po(tmp_path):
    db = str(tmp_path / "test_runs.db")
    storage.init_db(db)

    assert storage.cumulative_for_po("PO-1", db) == 0.0
    storage.record_ledger_entry("Acme", "INV-1", 40.0, "2024-01-01", "PO-1", db)
    storage.record_ledger_entry("Acme", "INV-2", 60.0, "2024-01-02", "PO-1", db)
    storage.record_ledger_entry("Acme", "INV-3", 10.0, "2024-01-03", "PO-2", db)

    assert storage.cumulative_for_po("PO-1", db) == 100.0
    assert storage.cumulative_for_po("PO-2", db) == 10.0
