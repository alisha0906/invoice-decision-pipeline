# Invoice Processing — from PDF to Decision

An automated, production-shaped pipeline that takes a vendor invoice (PDF or
image) and produces a clear, reasoned decision — **APPROVE / REVIEW / REJECT** —
with every step visible and every decision explained.

Built for Zamp's PS-1 (Finance/AP). The architecture mirrors a document-
digitisation system I built at Ericsson that processed 10,000+ scanned documents
with a parallel, resumable orchestrator.

---

## The engine: a phased orchestration pipeline

The core engine is `scripts/batch_orchestrator.py`. It runs every invoice in a
folder through four phases, **in parallel**, with a **completeness check** at each
boundary, **resumability**, and a **consolidated output**.

```
Folder of invoice PDFs / images
        │  ThreadPoolExecutor (N parallel workers)
        ▼
  ┌──────────────────── per invoice ────────────────────┐
  │ Phase 1  Pre-process  PDF → page image(s)            │
  │                       (PyMuPDF, 200→80 DPI size-cap) │
  │ Phase 2  Extract      image(s) + prompt → vision LLM │
  │                       → structured JSON fields       │
  │ Phase 3  Post-process normalize/coerce fields,       │
  │                       completeness QA flags          │
  │ Phase 4  Validate     match PO → ordered decision    │
  │                       rules → APPROVE/REVIEW/REJECT, │
  │                       persist run + seed ledger      │
  └──────────────────────────────────────────────────────┘
        │
        ▼  Phase 5  Consolidate
  batch_results.json + batch_summary.csv + batch_flags.csv
```

A web app (`app/main.py`) provides an interactive single-invoice view — live run
view + dashboard — over the same pipeline stages.

---

## Decision outcomes

- **APPROVE** — vendor approved, PO matched, amount within tolerance, no duplicate.
- **REVIEW** — needs a human: unreadable document, ambiguous/fuzzy/split PO match,
  low confidence, or amount near the tolerance edge.
- **REJECT** — hard failure: missing critical field, duplicate invoice, or amount
  clearly outside tolerance.

Every decision carries a **reasoning trail** — the ordered checks it ran and why
each passed or failed. That is the audit record an AP team needs.

## Edge cases handled

1. **Scanned / unreadable invoice** → REVIEW (routed to a human, never crashes).
2. **One PO split across multiple invoices** → cumulative amount tracked vs the PO total.
3. **Amount outside tolerance** → REJECT with the variance shown.
4. **Duplicate invoice** → REJECT (ledger-based detection).
5. **Missing critical field** (no invoice #, date, or total) → REJECT with reason.

---

## Vision reading (pluggable)

Reading is **vision-LLM-first**: every page is rendered to an image and read by a
vision model, the way a person reads a document. Chosen over OCR because OCR
collapses on varied, scanned layouts.

Pick the reader with `VISION_PROVIDER`:

- `ollama` (default) — a **local** vision model (`llava`, `llava-phi3`,
  `moondream`, `llama3.2-vision`). No API key, no rate limits, fully offline.
- `gemini` — Google Gemini (free AI Studio tier), with live model auto-detection.

If the reader is unavailable or errors, the run **degrades gracefully to REVIEW**
rather than crashing.

---

## Setup

From inside this folder:

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up the vision reader (local Ollama — default)
```bash
# Install Ollama from https://ollama.com, then pull a vision model:
ollama pull llava            # ~4.7 GB
# or: ollama pull llama3.2-vision   # ~7.9 GB, best accuracy
ollama list                  # confirm the exact model name
```

### 3. Create `.env`
```
EXTRACTION_MODE=vision_only
VISION_PROVIDER=ollama
OLLAMA_MODEL=llava:latest
OLLAMA_HOST=http://localhost:11434
```
(Match `OLLAMA_MODEL` exactly to what `ollama list` shows. To use Gemini instead,
set `VISION_PROVIDER=gemini` and `GEMINI_API_KEY=...`.)

### 4. Invoices to process
Sample invoices are already included in `sample_data/` (PDFs from the public
HuggingFace dataset
[`katanaml-org/invoices-donut-data-v1`](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1)).
To fetch more:
```bash
python scripts/download_hf_invoices.py --count 20 --out sample_data
```

### 5. Verify the reader (optional)
```bash
python scripts/verify_vision.py   # reads scripts/verify_vision.out.txt
```

---

## Run

### Batch engine (the main entry point)
```bash
# Process a whole folder in parallel:
python scripts/batch_orchestrator.py --input-dir sample_data --max-concurrent 4

# Preview without processing:
python scripts/batch_orchestrator.py --input-dir sample_data --dry-run

# Resume: re-running skips already-done invoices automatically.
# Re-run only the ones that failed:
python scripts/batch_orchestrator.py --input-dir sample_data --retry-failed
```

Outputs land in `<input-dir>/_batch_out/`:
- `batch_results.json` — full per-invoice result
- `batch_summary.csv` — one row per invoice (decision, fields, timing)
- `batch_flags.csv` — completeness / QA flags raised
- `batch_state.json` — durable progress (enables resume / retry)

### Interactive web app
```bash
uvicorn app.main:app
```
Open http://localhost:8000 — upload or pick an invoice, watch each stage run
live, and browse the dashboard of past runs.

---

## Orchestration features

- **Parallelism** — `ThreadPoolExecutor` processes N invoices concurrently.
- **Resumability** — `batch_state.json` records each outcome (written after every
  invoice); re-runs skip completed work, `--retry-failed` re-runs only errors.
- **Completeness / fault tolerance** — every invoice yields a result; unreadable
  ones route to REVIEW, a crash on one file never kills the batch, and per-invoice
  QA flags record missing fields / low confidence.
- **Thread-safe persistence** — a lock serialises SQLite writes.
- **Consolidation** — one combined JSON + a summary CSV + a QA-flags CSV.

---

## Project layout

```
app/
  main.py              FastAPI web layer (upload, SSE live run, dashboard API)
  config.py            settings (vision provider, thresholds) from .env
  models.py            RunRecord + enums
  storage.py           SQLite persistence (runs + invoice ledger)
  pipeline/
    extract.py         Phase 1+2: render PDF → vision LLM → fields
    providers.py       vision readers: Ollama (local), Gemini (hosted)
    llm.py             deterministic text fallback
    matching.py        Phase 4: PO matching (exact / fuzzy / split)
    decision.py        Phase 4: ordered decision rules + reasoning trail
    runner.py          single-invoice stage runner (drives the web live view)
  static/              Tailwind + Alpine UI (live run view + dashboard)
scripts/
  batch_orchestrator.py  the phased, parallel, resumable batch ENGINE
  download_hf_invoices.py fetch real invoices from HuggingFace
  verify_vision.py       health-check the configured vision reader
sample_data/
  purchase_orders.json   approved PO catalogue (matching targets)
  real_scans/            downloaded real invoices
tests/                   pytest: decision, matching, extraction, providers, api
```

## Tests
```bash
python -m pytest tests -q
```

## Tech stack
Python · FastAPI · Server-Sent Events · PyMuPDF (PDF→image) · Ollama / Gemini
vision models · SQLite · Tailwind + Alpine.js · `concurrent.futures` for the
parallel orchestrator.
