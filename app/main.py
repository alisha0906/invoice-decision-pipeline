"""FastAPI web layer.

Serves the static Tailwind + Alpine UI and exposes the invoice-processing API:

    POST /api/upload           -> save a PDF, return a token id
    GET  /api/stream/{id}      -> SSE stream of pipeline stages + final decision
    GET  /api/runs             -> list runs (optional ?decision= filter)
    GET  /api/runs/{id}        -> full run detail incl. reasoning trail
    GET  /api/samples          -> list bundled sample PDFs
    POST /api/run-sample/{name}-> run a bundled sample (returns a stream id)

Streaming drives ``runner.run_streaming`` and persists the run at the end. The
SQLite schema is created on startup. Every invoice is read by the configured
vision reader (Ollama by default, Gemini optional); if the reader is
unavailable the run degrades to REVIEW.
"""

import json
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import storage
from app.pipeline import runner

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_SAMPLE_DIR = _PROJECT_ROOT / "sample_data"
# Uploads live inside the project (same drive as the app), NOT the system temp
# dir. This avoids failures when %TEMP% / C: is full or gets cleared.
_UPLOAD_DIR = _PROJECT_ROOT / ".uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Create the SQLite schema before serving requests.
    storage.init_db()
    yield


app = FastAPI(title="Invoice Decision Pipeline", lifespan=_lifespan)

# Maps an upload/sample token to the (file_path, display_filename) to process.
_PENDING: dict[str, tuple[str, str]] = {}


def _sse(event: dict) -> str:
    """Format a dict as a single SSE 'data:' event."""
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    """Save an uploaded PDF to the uploads dir and return a stream token."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")
    # Ensure the uploads dir exists at write time (temp dirs can be cleared
    # between startup and now, e.g. by a disk cleanup), so this self-heals.
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    dest = _UPLOAD_DIR / f"{token}.pdf"
    contents = await file.read()
    dest.write_bytes(contents)
    _PENDING[token] = (str(dest), file.filename)
    return JSONResponse({"id": token, "filename": file.filename})


@app.get("/api/stream/{token}")
def stream(token: str) -> StreamingResponse:
    """Drive the pipeline for a pending upload/sample and stream each stage."""
    pending = _PENDING.get(token)
    if pending is None:
        raise HTTPException(status_code=404, detail="unknown or expired token")
    file_path, filename = pending

    def _gen():
        try:
            # Ensure tables exist (defensive: the DB file may have been
            # recreated/moved between startup and now), then run the pipeline.
            storage.init_db()
            for event in runner.run_streaming(file_path, filename):
                yield _sse(event)
        except Exception as exc:  # never 500 silently - stream a clear error
            import traceback
            tb = traceback.format_exc()
            yield _sse({
                "type": "error",
                "message": f"{exc.__class__.__name__}: {exc}",
                "traceback": tb[-1500:],
            })
        finally:
            _PENDING.pop(token, None)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/api/runs")
def list_runs(decision: Optional[str] = Query(default=None)) -> JSONResponse:
    """List persisted runs, newest first, optionally filtered by decision."""
    records = storage.list_runs(decision_filter=decision)
    return JSONResponse([_run_summary(r) for r in records])


@app.get("/api/runs/{run_id}")
def get_run(run_id: int) -> JSONResponse:
    """Return a single run's full detail, including its reasoning trail."""
    record = storage.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    return JSONResponse(_run_detail(record))


# Suffixes accepted for a bundled/real sample run (images flow through the
# vision reader; PDFs through the normal text-layer/vision/OCR pipeline).
_SAMPLE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}


def _resolve_sample(name: str) -> Optional[Path]:
    """Resolve a sample name (optionally 'real_scans/<file>') safely.

    Returns the resolved path if it lives inside _SAMPLE_DIR and has an accepted
    suffix, else None. Rejects path traversal via .resolve() containment.
    """
    candidate = (_SAMPLE_DIR / name).resolve()
    sample_root = _SAMPLE_DIR.resolve()
    if sample_root not in candidate.parents and candidate != sample_root:
        return None
    if candidate.suffix.lower() not in _SAMPLE_SUFFIXES or not candidate.is_file():
        return None
    return candidate


@app.get("/api/samples")
def list_samples() -> JSONResponse:
    """List the invoice samples in sample_data/ (real HuggingFace invoices).

    Invoices live directly in sample_data/ as PDFs. When both <id>.pdf and an
    image exist for the same id, the PDF is listed (the pipeline processes PDFs
    end-to-end); a nested real_scans/ folder, if present, is also included.
    """
    samples = []
    if not _SAMPLE_DIR.exists():
        return JSONResponse(samples)

    by_name = {}  # display-name -> relative sample name

    def _collect(folder: Path, prefix: str):
        by_id = {}
        for f in folder.glob("*.pdf"):
            by_id[f.stem] = f
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            for f in folder.glob(pattern):
                by_id.setdefault(f.stem, f)
        for stem in sorted(by_id):
            f = by_id[stem]
            by_name[prefix + f.name] = f.stem

    _collect(_SAMPLE_DIR, "")
    real_dir = _SAMPLE_DIR / "real_scans"
    if real_dir.exists():
        _collect(real_dir, "real_scans/")

    for name in sorted(by_name):
        samples.append(
            {
                "name": name,
                "label": by_name[name].replace("_", " "),
                "source": "invoice",
            }
        )
    return JSONResponse(samples)


@app.post("/api/run-sample/{name:path}")
def run_sample(name: str) -> JSONResponse:
    """Register a bundled/real sample for streaming and return a token."""
    sample_path = _resolve_sample(name)
    if sample_path is None:
        raise HTTPException(status_code=404, detail="sample not found")
    token = uuid.uuid4().hex
    display = sample_path.name
    _PENDING[token] = (str(sample_path), display)
    return JSONResponse({"id": token, "filename": display})


def _run_summary(record) -> dict:
    """Compact shape for the dashboard table."""
    return {
        "id": record.id,
        "timestamp": record.timestamp,
        "input_filename": record.input_filename,
        "decision": record.decision,
        "vendor": record.extracted_fields.get("vendor"),
        "invoice_number": record.extracted_fields.get("invoice_number"),
        "total": record.extracted_fields.get("total"),
    }


def _run_detail(record) -> dict:
    """Full shape for a run's detail view."""
    return {
        "id": record.id,
        "timestamp": record.timestamp,
        "input_filename": record.input_filename,
        "decision": record.decision,
        "extracted_fields": record.extracted_fields,
        "matched_po": record.matched_po,
        "reasoning_trail": record.reasoning_trail,
        "stages": record.stages,
    }


# Mount the static UI at the root. Declared last so /api/* routes win.
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
