"""Offline tests for scripts/download_hf_invoices.py.

No network: parse_ground_truth is exercised on an inlined Donut target, and
save_row is exercised on a small generated PIL image into a tmp dir.
"""

import json
import sys
from pathlib import Path

from PIL import Image

# scripts/ is a sibling of tests/ under PS-1; make it importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from download_hf_invoices import parse_ground_truth, save_row  # noqa: E402


_DONUT_TARGET = {
    "gt_parse": {
        "invoice_no": "INV-9",
        "invoice_date": "2024-01-02",
        "seller": "Foo Ltd",
        "total": "$10.00",
        "items": [{"item_desc": "x", "item_gross_worth": "10.00"}],
    }
}


def test_parse_ground_truth_maps_donut_target():
    truth = parse_ground_truth(_DONUT_TARGET)

    assert truth["invoice_number"] == "INV-9"
    assert truth["date"] == "2024-01-02"
    assert truth["vendor"] == "Foo Ltd"
    assert truth["total"] == 10.0
    assert truth["po_reference"] is None
    assert len(truth["line_items"]) == 1


def test_parse_ground_truth_accepts_json_string():
    truth = parse_ground_truth(json.dumps(_DONUT_TARGET))
    assert truth["invoice_number"] == "INV-9"
    assert truth["total"] == 10.0


def test_parse_ground_truth_unwraps_gt_parses_list():
    raw = {"gt_parses": [{"invoice_number": "A-1", "total_gross_worth": "42.50"}]}
    truth = parse_ground_truth(raw)
    assert truth["invoice_number"] == "A-1"
    assert truth["total"] == 42.5


def test_save_row_writes_png_pdf_and_gt(tmp_path):
    img = Image.new("RGB", (40, 30), "white")

    png, pdf, gt = save_row(tmp_path, "sample_0001", img, _DONUT_TARGET)

    assert png.exists() and png.name == "sample_0001.png"
    assert gt.exists() and gt.name == "sample_0001.gt.json"
    # The PDF is rasterised via PyMuPDF; it should exist and be non-empty.
    assert pdf.exists() and pdf.name == "sample_0001.pdf"
    assert pdf.stat().st_size > 0

    with open(gt, "r", encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["invoice_number"] == "INV-9"
    assert saved["total"] == 10.0
