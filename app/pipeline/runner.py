"""Pipeline orchestrator.

Wires the four pipeline stages together:

    extract -> llm.extract_fields -> matching.match_invoice -> decision.decide

Each stage's status, confidence, and produced data are captured into an
ordered ``stages`` list so the UI can replay exactly what happened. The result
is persisted as a ``RunRecord`` and, for non-duplicate flows, the invoice is
appended to the ledger so duplicate + split-PO tracking accumulate.

Two entry points share the same stage logic:

    run(file_path, filename) -> RunRecord
        Run everything synchronously and return the persisted record.

    run_streaming(file_path, filename) -> generator
        Yield one dict per stage as it completes, then a final ``decision``
        event carrying the persisted run. Used by the SSE endpoint.
"""

from typing import Any, Iterator, Optional

from app import storage as storage_mod
from app.models import Decision, RunRecord, StageStatus
from app.pipeline import decision as decision_mod
from app.pipeline import llm as llm_mod
from app.pipeline import matching as matching_mod
from app.pipeline.extract import extract as extract_fn

# Human-readable stage identifiers, surfaced in SSE events and the UI.
STAGE_EXTRACT = "extract"
STAGE_STRUCTURE = "structure_fields"
STAGE_MATCH = "match_po"
STAGE_DECIDE = "decide"

STAGE_LABELS = {
    STAGE_EXTRACT: "Extract",
    STAGE_STRUCTURE: "Structure Fields",
    STAGE_MATCH: "Match PO",
    STAGE_DECIDE: "Validate / Decide",
}


def _stage(
    name: str, status: StageStatus, confidence: float, data: dict, message: str = ""
) -> dict[str, Any]:
    """Build one per-stage record (also the SSE event payload shape)."""
    return {
        "name": name,
        "label": STAGE_LABELS.get(name, name),
        "status": status.value if isinstance(status, StageStatus) else status,
        "confidence": round(float(confidence), 3),
        "message": message,
        "data": data,
    }


class _StorageAdapter:
    """Bind storage calls to a specific db_path for the decision/matching code.

    ``matching`` and ``decision`` expect an object exposing
    ``cumulative_for_po`` and ``is_duplicate`` without a db_path argument.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path

    def cumulative_for_po(self, po_number: str) -> float:
        return storage_mod.cumulative_for_po(po_number, self._db_path)

    def is_duplicate(self, vendor, invoice_number, amount, date) -> bool:
        return storage_mod.is_duplicate(
            vendor, invoice_number, amount, date, self._db_path
        )


def _execute(file_path: str, store: _StorageAdapter):
    """Run all four stages, yielding each stage dict, then a final summary.

    Yields stage dicts one at a time; the final yielded value is a summary dict
    with keys: decision (Decision), fields, match (MatchResult), extraction,
    reasoning_trail, stages.
    """
    stages: list[dict] = []

    # --- 1) Extract (vision LLM) -------------------------------------------
    extraction = extract_fn(file_path)
    extract_status = (
        StageStatus.DEGRADED if extraction.extraction_unavailable else StageStatus.OK
    )
    # The vision reader that produced this result ('ollama' | 'gemini-vision' |
    # 'none' when it could not be read).
    reader = extraction.meta.get("reader") or extraction.reader or "none"
    n_pages = extraction.meta.get("pages")
    if extraction.extraction_unavailable:
        extract_msg = (
            "Could not read the document with the vision model — routing to human review."
        )
    else:
        pages_txt = f"{n_pages} page(s)" if n_pages else "the document"
        extract_msg = f"Reading {pages_txt} with the {reader} vision model…"
    extract_stage = _stage(
        STAGE_EXTRACT,
        extract_status,
        extraction.confidence,
        {
            "reader": reader,
            "pages": n_pages,
            "extraction_unavailable": extraction.extraction_unavailable,
            # Surface the real reader error (e.g. Gemini 429/503, Ollama down)
            # so a failed extraction is visible instead of a silent REVIEW.
            "error": extraction.meta.get("error"),
        },
        message=extract_msg,
    )
    stages.append(extract_stage)
    yield extract_stage

    # --- 2) Structure fields ------------------------------------------------
    # The vision reader already returned structured fields. If extraction was
    # unavailable, fall back to empty fields (decision engine routes to REVIEW).
    vision_fields = extraction.meta.get("vision_fields")
    if vision_fields is not None:
        fields = vision_fields
        provider_used = reader
    else:
        fields, provider_used = llm_mod.extract_fields(extraction.raw_text)
    _inv = fields.get("invoice_number") or "?"
    _vendor = fields.get("vendor") or "?"
    _total = fields.get("total")
    _total_txt = f"{_total}" if _total is not None else "?"
    structure_msg = (
        f"Captured invoice {_inv} from {_vendor}, total {_total_txt}."
    )
    structure_stage = _stage(
        STAGE_STRUCTURE,
        StageStatus.OK,
        # No numeric confidence from the LLM layer; mirror extraction quality.
        extraction.confidence,
        {"provider_used": provider_used, "fields": fields},
        message=structure_msg,
    )
    stages.append(structure_stage)
    yield structure_stage

    # --- 3) Match PO --------------------------------------------------------
    match_result = matching_mod.match_invoice(fields, store)
    match_status = (
        StageStatus.OK if match_result.match_type != "NONE" else StageStatus.DEGRADED
    )
    _mtype = match_result.match_type or "NONE"
    _po = None
    try:
        if isinstance(match_result.po, dict):
            _po = match_result.po.get("po_number")
    except Exception:
        _po = None
    if _mtype == "NONE":
        match_msg = "No matching purchase order found for this vendor/amount."
    else:
        match_msg = f"Matched purchase order {_po} ({str(_mtype).lower()} match)."
    match_stage = _stage(
        STAGE_MATCH,
        match_status,
        match_result.confidence,
        match_result.as_dict(),
        message=match_msg,
    )
    stages.append(match_stage)
    yield match_stage

    # --- 4) Validate / Decide ----------------------------------------------
    decision, reasoning_trail = decision_mod.decide(
        fields, extraction, match_result, store
    )
    decide_status = StageStatus.OK if decision == Decision.APPROVE else StageStatus.DEGRADED
    _reason = reasoning_trail[-1]["check"] if reasoning_trail else ""
    _decision_msg = {
        "APPROVE": "Approved — vendor approved, PO matched, amount within tolerance.",
        "REVIEW": "Flagged for human review.",
        "REJECT": "Rejected.",
    }.get(decision.value, decision.value)
    if _reason and decision.value != "APPROVE":
        _decision_msg += f" (rule: {_reason})"
    decide_stage = _stage(
        STAGE_DECIDE,
        decide_status,
        extraction.confidence,
        {"decision": decision.value, "reasoning_trail": reasoning_trail},
        message=_decision_msg,
    )
    stages.append(decide_stage)
    yield decide_stage

    yield {
        "_summary": True,
        "decision": decision,
        "fields": fields,
        "match": match_result,
        "extraction": extraction,
        "reasoning_trail": reasoning_trail,
        "stages": stages,
    }


def _persist(summary: dict, filename: str, store: _StorageAdapter, db_path) -> RunRecord:
    """Build and persist the RunRecord, seeding the ledger for tracking."""
    decision: Decision = summary["decision"]
    fields: dict = summary["fields"]
    match_result = summary["match"]
    reasoning_trail: list = summary["reasoning_trail"]
    stages: list = summary["stages"]

    matched_po = match_result.po if match_result is not None else None

    record = RunRecord(
        input_filename=filename,
        extracted_fields=fields,
        matched_po=matched_po,
        decision=decision.value,
        reasoning_trail=reasoning_trail,
        stages=stages,
    )
    storage_mod.insert_run(record, db_path)

    # Seed the ledger so duplicate + split-PO tracking accumulate across runs.
    # Skip duplicates (they were REJECTed) to avoid double-counting.
    is_dupe = store.is_duplicate(
        fields.get("vendor"),
        fields.get("invoice_number"),
        fields.get("total"),
        fields.get("date"),
    )
    if not is_dupe:
        po_number = None
        if matched_po:
            po_number = matched_po.get("po_number")
        if not po_number:
            po_number = fields.get("po_reference")
        storage_mod.record_ledger_entry(
            fields.get("vendor"),
            fields.get("invoice_number"),
            fields.get("total"),
            fields.get("date"),
            po_number,
            db_path,
        )

    return record


def run(file_path: str, filename: str, db_path: Optional[str] = None) -> RunRecord:
    """Process a PDF synchronously and return the persisted RunRecord."""
    store = _StorageAdapter(db_path)
    summary: Optional[dict] = None
    for item in _execute(file_path, store):
        if item.get("_summary"):
            summary = item
    assert summary is not None  # _execute always yields a summary last
    return _persist(summary, filename, store, db_path)


def run_streaming(
    file_path: str, filename: str, db_path: Optional[str] = None
) -> Iterator[dict]:
    """Yield one dict per stage, then a final ``decision`` dict.

    Stage dicts carry ``type="stage"``; the terminal dict carries
    ``type="decision"`` with the persisted run (id, decision, fields,
    matched_po, reasoning_trail).
    """
    store = _StorageAdapter(db_path)
    summary: Optional[dict] = None
    for item in _execute(file_path, store):
        if item.get("_summary"):
            summary = item
            break
        event = dict(item)
        event["type"] = "stage"
        yield event

    assert summary is not None
    record = _persist(summary, filename, store, db_path)
    yield {
        "type": "decision",
        "run_id": record.id,
        "timestamp": record.timestamp,
        "input_filename": record.input_filename,
        "decision": record.decision,
        "extracted_fields": record.extracted_fields,
        "matched_po": record.matched_po,
        "reasoning_trail": record.reasoning_trail,
    }
