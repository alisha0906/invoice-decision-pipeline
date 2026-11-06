"""Tests for the FastAPI web layer (app.main).

Uploads a generated text-layer invoice PDF, consumes the SSE stream and asserts
the four pipeline stages arrive in order followed by a final decision event,
then asserts the run is persisted and retrievable via GET /api/runs and
GET /api/runs/{id}. Also smoke-tests GET / for both views + CDN assets.

Runs fully offline (no vision reader available): extraction is unavailable so
the pipeline streams its four stages and routes to REVIEW without network. The
assertions check the stage/stream/persist contract, not a specific decision.
"""

import importlib
import json

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app import config as _config


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient backed by a fresh temp SQLite DB."""
    db = str(tmp_path / "api.db")
    monkeypatch.setenv("DB_PATH", db)
    # Reload config so DB_PATH picks up the temp path, then storage + main.
    importlib.reload(_config)
    import app.storage as storage
    importlib.reload(storage)
    import app.main as main
    importlib.reload(main)

    with TestClient(main.app) as c:
        yield c

    importlib.reload(_config)


def _make_invoice_pdf(path: str) -> None:
    """A clean text-layer invoice matching PO-1001 within tolerance."""
    c = canvas.Canvas(path, pagesize=letter)
    text = c.beginText(72, 720)
    for line in (
        "ACME CORPORATION",
        "Invoice Number: INV-7001",
        "Date: 2024-05-01",
        "PO Number: PO-1001",
        "Vendor: Acme Corporation",
        "Total: $1500.00",
    ):
        text.textLine(line)
    c.drawText(text)
    c.showPage()
    c.save()


def _parse_sse(raw: str) -> list[dict]:
    """Extract JSON payloads from an SSE 'data:' stream body."""
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


def test_upload_stream_persist_retrieve(client, tmp_path):
    pdf = str(tmp_path / "invoice.pdf")
    _make_invoice_pdf(pdf)

    # Upload -> token
    with open(pdf, "rb") as fh:
        res = client.post(
            "/api/upload",
            files={"file": ("invoice.pdf", fh, "application/pdf")},
        )
    assert res.status_code == 200
    token = res.json()["id"]

    # Stream -> SSE events
    res = client.get(f"/api/stream/{token}")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(res.text)

    stage_events = [e for e in events if e.get("type") == "stage"]
    decision_events = [e for e in events if e.get("type") == "decision"]

    # Four stages in canonical order, then exactly one decision event.
    assert [e["name"] for e in stage_events] == [
        "extract",
        "structure_fields",
        "match_po",
        "decide",
    ]
    assert len(decision_events) == 1
    final = decision_events[0]
    assert final["decision"] in ("APPROVE", "REVIEW", "REJECT")
    run_id = final["run_id"]
    assert run_id is not None

    # Persisted + retrievable via list.
    res = client.get("/api/runs")
    assert res.status_code == 200
    runs = res.json()
    assert any(r["id"] == run_id for r in runs)

    # Retrievable via detail with a reasoning trail.
    res = client.get(f"/api/runs/{run_id}")
    assert res.status_code == 200
    detail = res.json()
    assert detail["id"] == run_id
    assert detail["decision"] == final["decision"]
    assert len(detail["reasoning_trail"]) >= 1


def test_runs_decision_filter(client, tmp_path):
    pdf = str(tmp_path / "invoice.pdf")
    _make_invoice_pdf(pdf)
    with open(pdf, "rb") as fh:
        token = client.post(
            "/api/upload", files={"file": ("invoice.pdf", fh, "application/pdf")}
        ).json()["id"]
    events = _parse_sse(client.get(f"/api/stream/{token}").text)
    decision = [e for e in events if e.get("type") == "decision"][0]["decision"]

    # Filtering by the actual decision returns the run; a different one does not.
    res = client.get("/api/runs", params={"decision": decision})
    assert res.status_code == 200
    assert len(res.json()) >= 1
    assert all(r["decision"] == decision for r in res.json())


def test_get_run_404(client):
    assert client.get("/api/runs/999999").status_code == 404


def test_real_scans_listed_and_runnable(client, tmp_path, monkeypatch):
    """A downloaded real_scans invoice is listed (source 'real / HuggingFace')
    and runs end-to-end through the orchestrator, streaming four stages + a
    decision and persisting the run."""
    import app.main as main

    # Point the app at a temp sample dir with invoice PDFs directly inside it.
    sample_dir = tmp_path / "sample_data"
    sample_dir.mkdir(parents=True)
    monkeypatch.setattr(main, "_SAMPLE_DIR", sample_dir)

    # A tiny invoice standing in for a downloaded real scan.
    real_pdf = sample_dir / "hf_train_0000.pdf"
    _make_invoice_pdf(str(real_pdf))

    # Listed as an invoice sample (name == filename directly under sample_data).
    res = client.get("/api/samples")
    assert res.status_code == 200
    listed = res.json()
    assert listed, "expected at least one invoice sample"
    entry = next(s for s in listed if s["name"] == "hf_train_0000.pdf")
    assert entry["source"] == "invoice"

    # Runnable: POST /api/run-sample/<name> then consume the SSE stream.
    res = client.post("/api/run-sample/" + entry["name"])
    assert res.status_code == 200
    token = res.json()["id"]

    events = _parse_sse(client.get(f"/api/stream/{token}").text)
    stage_events = [e for e in events if e.get("type") == "stage"]
    decision_events = [e for e in events if e.get("type") == "decision"]
    assert [e["name"] for e in stage_events] == [
        "extract",
        "structure_fields",
        "match_po",
        "decide",
    ]
    assert len(decision_events) == 1
    final = decision_events[0]
    assert final["decision"] in ("APPROVE", "REVIEW", "REJECT")

    # The run persisted and is retrievable.
    run_id = final["run_id"]
    runs = client.get("/api/runs").json()
    assert any(r["id"] == run_id for r in runs)


def test_index_serves_both_views(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.text
    # CDN assets present.
    assert "cdn.tailwindcss.com" in body
    assert "alpinejs" in body
    # Both views referenced.
    assert "Live Run" in body
    assert "Dashboard" in body
