# Architecture — Invoice Processing (PS-1)

**Problem (Zamp PS-1, Finance/AP):** a mid-size company receives hundreds of
vendor invoices a month, all as PDFs. Today someone on the AP team opens each
one, finds the matching purchase order, checks whether the numbers line up, and
decides what to do. It's slow, repetitive, and error-prone.

**What this builds:** an automated pipeline that takes an invoice (PDF or image)
and produces a clear, reasoned decision — **APPROVE / REVIEW / REJECT** — with
every intermediate step visible and every decision explained.

The design is a **phased orchestration pipeline** — the same architecture as a
document-digitisation system I built at Ericsson that processed 10,000+ scanned
documents with parallel, resumable workers.

---

## 1. Two entry points, one pipeline

| Entry point | File | Use |
|---|---|---|
| **Batch engine** | `scripts/batch_orchestrator.py` | The core engine: process a folder of invoices in parallel, resumable, with consolidated output. |
| **Interactive web app** | `app/main.py` | Single-invoice live view + dashboard over the same pipeline stages. |

Both run the invoice through the same four phases.

---

## 2. The phased pipeline

```
Invoice (PDF or image)
   │
   ▼  Phase 1 — Pre-process        [pure Python]
   │   Render each PDF page to an image with PyMuPDF. Size-capped: start at
   │   150 DPI and step down (→ JPEG) until each page is under ~1.2 MB, so the
   │   vision model never receives an oversized input. Raw images pass through.
   │
   ▼  Phase 2 — Extract            [vision LLM]
   │   Send the page image(s) + a strict structured prompt to the vision model;
   │   it returns a JSON object of fields. Multi-page invoices send all pages in
   │   one call so figures spanning pages are combined.
   │
   ▼  Phase 3 — Post-process       [pure Python]
   │   Normalize and coerce the AI output: enforce the canonical 6-field schema,
   │   parse amounts (US and European formats), drop hallucinated keys, and raise
   │   completeness / QA flags for missing or low-confidence fields.
   │
   ▼  Phase 4 — Validate & decide  [rules]
   │   Match to a purchase order, then an ordered rules engine returns
   │   APPROVE / REVIEW / REJECT with a reasoning trail. Persist the run and seed
   │   the ledger for duplicate + split-PO tracking.
   │
   ▼  Decision + reasoning trail
```

**Cross-cutting principles**
- **Completeness check at every phase** — missing or unreadable data is flagged
  and routed, never silently dropped.
- **Graceful degradation** — a reader failure (no key, model down, bad image)
  becomes a REVIEW with a clear reason, never a crash.
- **LLM extracts, rules decide** — the model is non-deterministic, so it never
  makes the decision; a transparent, testable rule engine does.

---

## 3. Phase detail

### Phase 1 — Pre-process (`app/pipeline/extract.py`)
`_render_all_pages_png` renders every page with PyMuPDF and caps each page's byte
size (step DPI down, then JPEG). This is the single most important reliability
step — oversized images are the main cause of local-vision-model failures.

### Phase 2 — Extract (`app/pipeline/extract.py` + `providers.py`)
The configured vision reader turns the image(s) into a JSON object. The reader is
pluggable via `VISION_PROVIDER`:
- `ollama` (default) — a **local** vision model (`llava`, `llava-phi3`,
  `moondream`, `llama3.2-vision`). No API key, no rate limits, offline.
- `gemini` — Google Gemini (free AI Studio tier) with live model
  auto-detection (queries `/models` so it never calls a retired model name).

Both retry on transient server errors and downscale images before sending.
Output is the canonical field set: `invoice_number, date, total, vendor,
po_reference, currency`.

### Phase 3 — Post-process (`providers._coerce_fields`, orchestrator `validate_extraction`)
- `_coerce_fields` keeps only the 6 canonical keys, fills missing with `null`,
  and coerces `total` to a float (handling `$1,234.56` and `8,25` style numbers).
- `validate_extraction` raises QA flags (severity WARN / REVIEW / ERROR) for an
  unreadable document, missing critical fields, or a missing total — without
  failing the run.

### Phase 4 — Validate & decide (`matching.py`, `decision.py`)

**PO matching** against `sample_data/purchase_orders.json`:

| Match type | How | Notes |
|---|---|---|
| EXACT | `po_reference` equals a known PO number | confidence 1.0 |
| FUZZY | no PO ref → vendor-name similarity (difflib ≥ 0.6) + amount in tolerance | confidence = ratio |
| SPLIT | an EXACT PO that already has prior ledger entries → checks the **cumulative** amount vs the PO total | one PO across invoices |
| NONE | nothing matched | → REVIEW |

**Decision rules** (ordered, first decisive rule wins):

0. Extraction unavailable (reader couldn't read) → **REVIEW** — precedes the
   missing-field check, because "couldn't read" ≠ "field missing".
1. Missing critical field (invoice_number / date / total) → **REJECT**.
2. Duplicate invoice (ledger match) → **REJECT**.
3. Amount clearly outside tolerance → **REJECT** (borderline band → REVIEW).
4. Low confidence / ambiguous or split match → **REVIEW**.
5. Otherwise → **APPROVE**.

Tolerance is relative (`TOLERANCE_PCT`, default 2%) with an absolute floor
(`TOLERANCE_ABS`), configurable per-PO.

---

## 4. The batch orchestrator (the engine)

`scripts/batch_orchestrator.py` runs a whole folder through the four phases.

```
Folder of invoices
      │  ThreadPoolExecutor (N parallel workers)
      ▼
  process_invoice()  →  Phase 1 → 2 → 3 → 4  (thread-safe persist)
      │
      ▼  Phase 5 — Consolidate
  batch_results.json + batch_summary.csv + batch_flags.csv + batch_state.json
```

- **Parallelism** — `--max-concurrent` workers process invoices concurrently.
- **Resumability** — `batch_state.json` records each outcome after every invoice;
  re-runs skip completed invoices, `--retry-failed` re-runs only errors,
  `--dry-run` previews, `--summary-only` rebuilds outputs from state.
- **Fault tolerance** — a crash on one invoice is recorded as an error flag; the
  batch continues.
- **Thread-safe persistence** — one lock serialises SQLite writes so concurrent
  workers can't corrupt the runs / ledger tables.
- **Consolidation** — combined JSON, a per-invoice summary CSV, and a QA-flags CSV.

This is the production/scale path; the web app is the interactive single-invoice
view over the same stages.

---

## 5. Persistence (`app/storage.py`, SQLite)

- `runs` — every processed invoice: timestamp, filename, extracted fields,
  matched PO, decision, full reasoning trail, per-stage data.
- `invoice_ledger` — seeds duplicate detection and split-PO cumulative tracking
  across runs.

---

## 6. Web layer (`app/main.py` + `app/static/`)

- `POST /api/upload` — save an uploaded PDF, return a stream token.
- `GET  /api/stream/{token}` — **Server-Sent Events**: one event per stage as it
  runs, then a final decision event (drives the live-run animation).
- `GET  /api/runs`, `/api/runs/{id}` — dashboard history + run detail.
- `GET  /api/samples`, `POST /api/run-sample/{name}` — run invoices from
  `sample_data/`.

UI: Tailwind + Alpine.js (no build step). Two views — **Live Run** (stages light
up with a message, confidence, and which reader was used) and **Dashboard**
(filterable run history with status badges and a detail modal).

---

## 7. Data

- **Real invoices** — a sample from the HuggingFace dataset
  `katanaml-org/invoices-donut-data-v1` (~500 real invoice images with ground
  truth), fetched by `scripts/download_hf_invoices.py` into
  `sample_data/real_scans/`.
- **PO catalogue** — `sample_data/purchase_orders.json`, the approved vendors +
  purchase orders the matcher validates against.

---

## 8. Tech stack

- **Engine:** Python, `concurrent.futures.ThreadPoolExecutor`
- **Web:** FastAPI + Server-Sent Events
- **Pre-process:** PyMuPDF (PDF→image), Pillow (downscaling)
- **Vision reading:** Ollama (local, default) / Google Gemini (hosted option)
- **Rules:** pure Python matching + decision, difflib for fuzzy vendor matching
- **Storage:** SQLite (stdlib)
- **Frontend:** Tailwind CSS + Alpine.js
- **Tests:** pytest (decision, matching, extraction routing, providers, API)

---

## 9. Key design decisions (the judgment behind them)

- **Phased orchestration** — pre-process, extract, post-process, validate —
  with completeness checks at each boundary, so failures surface early and get
  routed, not buried. The same shape as my Ericsson production pipeline.
- **Vision-LLM-first, not OCR** — a vision model reads varied, scanned layouts
  the way a person does and returns structured fields in one step.
- **LLM extracts, rules decide** — the decision is deterministic, auditable, and
  testable; the non-deterministic model never decides.
- **Graceful degradation over hard failure** — anything uncertain or unreadable
  routes to human REVIEW with a reason; the process never just breaks.
- **"Couldn't read" ≠ "field missing"** — these are different failure modes with
  different right answers, and the decision engine distinguishes them.
- **Pluggable reader** — `VISION_PROVIDER` swaps local Ollama ↔ hosted Gemini
  with one env var (privacy/offline vs raw accuracy).
- **Built to scale** — the orchestrator is parallel, resumable, and fault-
  tolerant, so the single-invoice demo is the foundation of a batch system.
