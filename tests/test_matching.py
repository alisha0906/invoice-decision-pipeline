"""Tests for app.pipeline.matching: EXACT, FUZZY, SPLIT, and NONE.

The tests pass their own PO catalogue via ``pos=`` so they exercise the matching
LOGIC independently of the demo sample_data/purchase_orders.json.
"""

from app import storage
from app.pipeline.matching import match_invoice

# Self-contained PO catalogue for the matching logic tests.
_POS = [
    {"po_number": "PO-1001", "vendor": "Acme Corporation", "amount": 1500.00, "tolerance_pct": 0.02},
    {"po_number": "PO-1002", "vendor": "Globex Industries", "amount": 4200.00, "tolerance_pct": 0.05},
    {"po_number": "PO-2001", "vendor": "Umbrella Supplies", "amount": 10000.00, "tolerance_pct": 0.02},
]


def _init_db(tmp_path):
    db = str(tmp_path / "match.db")
    storage.init_db(db)
    return db


class _Store:
    """Thin storage adapter binding a fixed db_path for the matcher."""

    def __init__(self, db):
        self.db = db

    def cumulative_for_po(self, po_number):
        return storage.cumulative_for_po(po_number, self.db)

    def is_duplicate(self, *a, **k):
        return storage.is_duplicate(*a, db_path=self.db, **k)


def test_exact_match(tmp_path):
    store = _Store(_init_db(tmp_path))
    fields = {"po_reference": "PO-1001", "vendor": "Acme Corporation", "total": 1500.00}
    result = match_invoice(fields, store, pos=_POS)
    assert result.match_type == "EXACT"
    assert result.po["po_number"] == "PO-1001"
    assert result.within_tolerance is True


def test_fuzzy_match_vendor_amount(tmp_path):
    store = _Store(_init_db(tmp_path))
    # No po_reference; vendor name slightly off but amount within tolerance.
    fields = {"po_reference": None, "vendor": "Globex Industrie", "total": 4200.00}
    result = match_invoice(fields, store, pos=_POS)
    assert result.match_type == "FUZZY"
    assert result.po["po_number"] == "PO-1002"
    assert result.within_tolerance is True


def test_split_cumulative_within_and_over(tmp_path):
    db = _init_db(tmp_path)
    store = _Store(db)

    # Seed a prior invoice billing half of PO-2001 (10000 total).
    storage.record_ledger_entry("Umbrella Supplies", "INV-A", 5000.0, "2024-01-01", "PO-2001", db)

    # Second invoice of 5000 -> cumulative 10000, within tolerance.
    within = match_invoice(
        {"po_reference": "PO-2001", "vendor": "Umbrella Supplies", "total": 5000.0},
        store, pos=_POS,
    )
    assert within.match_type == "SPLIT"
    assert within.cumulative_amount == 10000.0
    assert within.within_tolerance is True

    # Second invoice of 6000 -> cumulative 11000, clearly over tolerance.
    over = match_invoice(
        {"po_reference": "PO-2001", "vendor": "Umbrella Supplies", "total": 6000.0},
        store, pos=_POS,
    )
    assert over.match_type == "SPLIT"
    assert over.cumulative_amount == 11000.0
    assert over.within_tolerance is False


def test_none_match(tmp_path):
    store = _Store(_init_db(tmp_path))
    fields = {"po_reference": "PO-DOES-NOT-EXIST", "vendor": "Unknown Vendor XYZ", "total": 1234.0}
    result = match_invoice(fields, store, pos=_POS)
    assert result.match_type == "NONE"
    assert result.po is None
