"""Tests for app.pipeline.decision: one assertion per routing scenario.

Scenarios: APPROVE, missing-field REJECT, duplicate REJECT, out-of-tolerance
REJECT, borderline REVIEW, low-confidence/OCR-unavailable REVIEW, split-PO
REVIEW. Each asserts the reasoning_trail is populated with the relevant check.
"""

import importlib

import pytest

from app import storage
from app.models import Decision
from app.pipeline.decision import decide


@pytest.fixture(autouse=True)
def _clean_config(monkeypatch):
    """Reset config to defaults.

    test_config.py reloads app.config with monkeypatched thresholds; that
    reloaded module can leak stale values into later tests. Reload here against
    a clean environment so tolerance constants are the documented defaults.
    """
    for var in ("TOLERANCE_PCT", "TOLERANCE_BORDERLINE_PCT", "TOLERANCE_ABS", "LOW_CONFIDENCE_THRESHOLD"):
        monkeypatch.delenv(var, raising=False)
    import app.config as _config

    importlib.reload(_config)
    yield
    importlib.reload(_config)


def _store(tmp_path):
    db = str(tmp_path / "decide.db")
    storage.init_db(db)

    class _Store:
        def is_duplicate(self, *a, **k):
            return storage.is_duplicate(*a, db_path=db, **k)

        def cumulative_for_po(self, po_number):
            return storage.cumulative_for_po(po_number, db)

    return _Store(), db


def _trail_has(trail, check_name):
    return any(item["check"] == check_name for item in trail)


HIGH_CONF = {"confidence": 0.95, "path_used": "text", "ocr_unavailable": False}


def test_approve(tmp_path):
    store, _ = _store(tmp_path)
    fields = {
        "invoice_number": "INV-1",
        "date": "2024-01-01",
        "total": 1500.00,
        "vendor": "Acme Corporation",
        "po_reference": "PO-1001",
    }
    match = {
        "match_type": "EXACT",
        "po": {"po_number": "PO-1001"},
        "ambiguous": False,
        "po_amount": 1500.00,
        "cumulative_amount": 1500.00,
        "within_tolerance": True,
    }
    decision, trail = decide(fields, HIGH_CONF, match, store)
    assert decision == Decision.APPROVE
    assert _trail_has(trail, "approved")


def test_missing_field_reject(tmp_path):
    store, _ = _store(tmp_path)
    fields = {"invoice_number": "INV-2", "date": None, "total": 100.0}
    decision, trail = decide(fields, HIGH_CONF, {"match_type": "NONE"}, store)
    assert decision == Decision.REJECT
    assert _trail_has(trail, "critical_fields_present")
    assert trail[0]["observed"]["missing"] == ["date"]


def test_duplicate_reject(tmp_path):
    store, db = _store(tmp_path)
    storage.record_ledger_entry("Acme", "INV-DUP", 500.0, "2024-01-01", "PO-1001", db)
    fields = {
        "invoice_number": "INV-DUP",
        "date": "2024-01-01",
        "total": 500.0,
        "vendor": "Acme",
        "po_reference": "PO-1001",
    }
    match = {"match_type": "EXACT", "po_amount": 500.0, "within_tolerance": True}
    decision, trail = decide(fields, HIGH_CONF, match, store)
    assert decision == Decision.REJECT
    assert _trail_has(trail, "duplicate_invoice")


def test_out_of_tolerance_reject(tmp_path):
    store, _ = _store(tmp_path)
    fields = {
        "invoice_number": "INV-3",
        "date": "2024-01-01",
        "total": 1800.00,  # 20% over 1500
        "vendor": "Acme Corporation",
        "po_reference": "PO-1001",
    }
    match = {
        "match_type": "EXACT",
        "ambiguous": False,
        "po_amount": 1500.00,
        "cumulative_amount": 1800.00,
        "within_tolerance": False,
    }
    decision, trail = decide(fields, HIGH_CONF, match, store)
    assert decision == Decision.REJECT
    assert _trail_has(trail, "amount_within_tolerance")


def test_borderline_review(tmp_path):
    store, _ = _store(tmp_path)
    # 2.5% over: above TOLERANCE_PCT (2%) but within borderline band (+1%).
    fields = {
        "invoice_number": "INV-4",
        "date": "2024-01-01",
        "total": 1537.50,
        "vendor": "Acme Corporation",
        "po_reference": "PO-1001",
    }
    match = {
        "match_type": "EXACT",
        "ambiguous": False,
        "po_amount": 1500.00,
        "cumulative_amount": 1537.50,
        "within_tolerance": False,
    }
    decision, trail = decide(fields, HIGH_CONF, match, store)
    assert decision == Decision.REVIEW
    assert _trail_has(trail, "amount_borderline")


def test_low_confidence_ocr_unavailable_review(tmp_path):
    store, _ = _store(tmp_path)
    fields = {
        "invoice_number": "INV-5",
        "date": "2024-01-01",
        "total": 1500.00,
        "vendor": "Acme Corporation",
        "po_reference": "PO-1001",
    }
    match = {
        "match_type": "EXACT",
        "ambiguous": False,
        "po_amount": 1500.00,
        "cumulative_amount": 1500.00,
        "within_tolerance": True,
    }
    extraction = {"confidence": 0.0, "path_used": "ocr_unavailable", "ocr_unavailable": True}
    decision, trail = decide(fields, extraction, match, store)
    assert decision == Decision.REVIEW
    # Reader-unavailable short-circuits to REVIEW before any other check.
    assert _trail_has(trail, "extraction_available")


def test_extraction_unavailable_reviews_before_missing_field(tmp_path):
    """Reader-unavailable must REVIEW, not REJECT, even with no fields.

    An image-only/scanned PDF with no reader available extracts zero fields.
    That is a 'could not read' failure, not an 'invoice omits the field'
    failure, so it must route to REVIEW ahead of the missing-field REJECT.
    """
    store, _ = _store(tmp_path)
    fields = {"invoice_number": None, "date": None, "total": None, "vendor": None}
    extraction = {
        "confidence": 0.0,
        "path_used": "ocr_unavailable",
        "extraction_unavailable": True,
    }
    decision, trail = decide(fields, extraction, {"match_type": "NONE"}, store)
    assert decision == Decision.REVIEW
    assert _trail_has(trail, "extraction_available")
    assert not _trail_has(trail, "critical_fields_present")


def test_split_po_review(tmp_path):
    store, _ = _store(tmp_path)
    fields = {
        "invoice_number": "INV-6",
        "date": "2024-01-01",
        "total": 5000.00,
        "vendor": "Umbrella Supplies",
        "po_reference": "PO-2001",
    }
    match = {
        "match_type": "SPLIT",
        "ambiguous": True,
        "po_amount": 10000.00,
        "cumulative_amount": 10000.00,
        "within_tolerance": True,
    }
    decision, trail = decide(fields, HIGH_CONF, match, store)
    assert decision == Decision.REVIEW
    assert _trail_has(trail, "no_review_flags")
