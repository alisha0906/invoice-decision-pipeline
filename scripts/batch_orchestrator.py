#!/usr/bin/env python3
"""
Batch Orchestrator for Invoice Processing (PS-1)
================================================
Processes every invoice (PDF or image) in a folder through the phased pipeline,
in parallel, with resumability, completeness checks, and a consolidated output.
This is the same orchestration pattern as the Ericsson POD document-digitisation
system (render -> AI -> post-process -> validate -> consolidate).

  Phase 1  Pre-process : PDF -> page image(s) (PyMuPDF, size-capped, parallel)
  Phase 2  Extract     : image(s) + structured prompt -> vision LLM -> JSON (parallel)
  Phase 3  Post-process: normalize/coerce fields, enforce schema, completeness check
  Phase 4  Validate    : match PO -> ordered decision rules -> APPROVE/REVIEW/REJECT,
                         raise QA flags, persist the run + seed the ledger
  Phase 5  Consolidate : combined results.json + summary.csv + flags.csv

Orchestration features (mirroring POD):
  * Parallelism     : ThreadPoolExecutor runs N invoices concurrently.
  * Resumability    : state.json records outcomes; re-runs skip done invoices;
                      --retry-failed re-runs only the ones that errored.
  * Completeness    : every invoice yields a result (REVIEW on unreadable);
                      per-invoice QA flags record missing fields / low confidence.
  * Fault tolerance : one bad invoice never kills the batch.
  * Consolidation   : one combined JSON, a summary CSV, and a QA flags CSV.

Usage:
    python scripts/batch_orchestrator.py --input-dir sample_data/real_scans
    python scripts/batch_orchestrator.py --input-dir sample_data/real_scans --max-concurrent 4
    python scripts/batch_orchestrator.py --input-dir sample_data/real_scans --dry-run
    python scripts/batch_orchestrator.py --input-dir sample_data/real_scans --retry-failed
    python scripts/batch_orchestrator.py --input-dir sample_data/real_scans --summary-only
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app import config, storage          # noqa: E402
from app.models import Decision          # noqa: E402
from app.pipeline import decision as decision_mod   # noqa: E402
from app.pipeline import matching as matching_mod   # noqa: E402
from app.pipeline.extract import extract as extract_fn  # noqa: E402

# ============================================================
# CONFIGURATION
# ============================================================
MAX_CONCURRENT = 4
_INPUT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}
_CRITICAL_FIELDS = ("invoice_number", "date", "total")

# sqlite3 + threads: serialise the write-side (ledger + runs) behind one lock.
_DB_LOCK = Lock()


# ============================================================
# STORAGE ADAPTER (binds storage calls to this batch's DB)
# ============================================================
class _StorageAdapter:
    def __init__(self, db_path):
        self._db = db_path

    def cumulative_for_po(self, po_number):
        return storage.cumulative_for_po(po_number, self._db)

    def is_duplicate(self, vendor, invoice_number, amount, date):
        return storage.is_duplicate(vendor, invoice_number, amount, date, self._db)


# ============================================================
# HELPERS
# ============================================================
def _find_inputs(input_dir: Path, limit=None) -> list:
    """Discover invoices; prefer the PDF when <id>.pdf and <id>.png both exist."""
    files = [p for p in sorted(input_dir.iterdir())
             if p.is_file() and p.suffix.lower() in _INPUT_SUFFIXES]
    by_stem = {}
    for f in files:
        by_stem.setdefault(f.stem, {})[f.suffix.lower()] = f
    chosen = [variants.get(".pdf") or next(iter(variants.values()))
              for _, variants in sorted(by_stem.items())]
    return chosen[:limit] if limit else chosen


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"done": {}, "failed": {}}


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ============================================================
# PHASE 3 COMPLETENESS CHECK (pre-decision QA flags)
# ============================================================
def validate_extraction(fields: dict, extraction, invoice: str) -> list:
    """Completeness / QA flags for one invoice's extracted fields.

    Mirrors POD's validate_extraction: records WARN/REVIEW/ERROR flags without
    failing the run, so a human sees exactly what was uncertain.
    """
    flags = []
    fields = fields or {}

    if getattr(extraction, "extraction_unavailable", False):
        flags.append({"severity": "ERROR", "invoice": invoice,
                      "check": "extraction", "detail": "document could not be read"})
        return flags

    for f in _CRITICAL_FIELDS:
        if not fields.get(f) and fields.get(f) != 0:
            flags.append({"severity": "REVIEW", "invoice": invoice,
                          "check": f, "detail": f"missing {f}"})

    if fields.get("total") in (None, ""):
        flags.append({"severity": "WARN", "invoice": invoice,
                      "check": "total", "detail": "no total captured"})

    return flags


# ============================================================
# PER-INVOICE PIPELINE (Phases 1-4)
# ============================================================
def process_invoice(path: Path, db_path: str) -> dict:
    """Run Phases 1-4 for one invoice. Never raises; returns a result dict.

      Phase 1: render (inside extract_fn)
      Phase 2: vision extraction (inside extract_fn)
      Phase 3: structure + completeness flags
      Phase 4: match PO -> decide -> persist
    """
    name = path.name
    t0 = time.time()
    result = {"invoice": name, "status": "error", "decision": None,
              "flags": [], "seconds": 0.0}
    try:
        # --- Phase 1 + 2: render + vision extraction ---
        extraction = extract_fn(str(path))
        fields = (extraction.meta or {}).get("vision_fields") or {}
        reader = (extraction.meta or {}).get("reader") or extraction.reader or "none"

        # --- Phase 3: completeness check / QA flags ---
        flags = validate_extraction(fields, extraction, name)

        # --- Phase 4: match + decide + persist (thread-safe) ---
        with _DB_LOCK:
            store = _StorageAdapter(db_path)
            match_result = matching_mod.match_invoice(fields, store)
            dec, trail = decision_mod.decide(fields, extraction, match_result, store)

            from app.models import RunRecord
            matched_po = match_result.po if match_result else None
            record = RunRecord(
                input_filename=name,
                extracted_fields=fields,
                matched_po=matched_po,
                decision=dec.value,
                reasoning_trail=trail,
                stages=[],
            )
            storage.insert_run(record, db_path)
            # seed ledger (skip duplicates to avoid double counting)
            if not store.is_duplicate(fields.get("vendor"), fields.get("invoice_number"),
                                      fields.get("total"), fields.get("date")):
                po_no = (matched_po or {}).get("po_number") if matched_po else fields.get("po_reference")
                storage.record_ledger_entry(fields.get("vendor"), fields.get("invoice_number"),
                                            fields.get("total"), fields.get("date"), po_no, db_path)

        result.update(
            status="ok", decision=dec.value, reader=reader,
            invoice_number=fields.get("invoice_number"), vendor=fields.get("vendor"),
            total=fields.get("total"),
            matched_po=(matched_po or {}).get("po_number") if matched_po else None,
            run_id=record.id, flags=flags,
        )
    except Exception as exc:  # fault tolerance: record, don't crash the batch
        result["error"] = f"{exc.__class__.__name__}: {exc}"
        result["flags"] = [{"severity": "ERROR", "invoice": name,
                            "check": "pipeline", "detail": str(exc)[:160]}]
    result["seconds"] = round(time.time() - t0, 1)
    return result


# ============================================================
# PHASE 5: CONSOLIDATION (results.json + summary.csv + flags.csv)
# ============================================================
def write_outputs(results: list, output_dir: Path) -> None:
    (output_dir / "batch_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    with open(output_dir / "batch_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["invoice", "status", "decision", "reader", "invoice_number",
                    "vendor", "total", "matched_po", "seconds", "error"])
        for r in results:
            w.writerow([r.get("invoice"), r.get("status"), r.get("decision"),
                        r.get("reader"), r.get("invoice_number"), r.get("vendor"),
                        r.get("total"), r.get("matched_po"), r.get("seconds"),
                        r.get("error", "")])

    # QA flags CSV (completeness/validation findings across the batch).
    with open(output_dir / "batch_flags.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Severity", "Invoice", "Check", "Detail"])
        for r in results:
            for fl in r.get("flags", []):
                w.writerow([fl.get("severity"), fl.get("invoice"),
                            fl.get("check"), fl.get("detail")])


# ============================================================
# MAIN ORCHESTRATION
# ============================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Batch invoice processing orchestrator")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="List invoices, don't process")
    ap.add_argument("--retry-failed", action="store_true", help="Re-run only failed invoices")
    ap.add_argument("--summary-only", action="store_true",
                    help="Rebuild outputs from the existing state, no processing")
    ap.add_argument("--db", default=str(_PROJECT_ROOT / "runs.db"))
    args = ap.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = _PROJECT_ROOT / input_dir
    if not input_dir.exists():
        print(f"ERROR: input dir not found: {input_dir}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "_batch_out")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "batch_state.json"

    storage.init_db(args.db)
    state = _load_state(state_path)
    inputs = _find_inputs(input_dir, args.limit)

    print("=" * 64)
    print("  BATCH INVOICE PROCESSING ORCHESTRATOR  (phased, parallel, resumable)")
    print("=" * 64)
    print(f"  Input:    {input_dir}")
    print(f"  Invoices: {len(inputs)} found")
    print(f"  Reader:   {getattr(config, 'VISION_PROVIDER', 'ollama')} "
          f"(EXTRACTION_MODE={config.EXTRACTION_MODE})")
    print(f"  Workers:  {args.max_concurrent}")
    print(f"  Output:   {output_dir}")
    print("=" * 64)

    if not inputs:
        print("No invoices found.")
        return 0

    # --- DRY RUN ---
    if args.dry_run:
        for i, p in enumerate(inputs, 1):
            tag = "DONE" if p.name in state["done"] else "PROCESS"
            print(f"  {i:4d}. [{tag:8s}] {p.name}")
        print(f"\n  {sum(1 for p in inputs if p.name in state['done'])} done, "
              f"{sum(1 for p in inputs if p.name not in state['done'])} to process")
        return 0

    # --- SUMMARY-ONLY: rebuild outputs from state, no processing ---
    if args.summary_only:
        results = [{"invoice": k, "status": "ok", "decision": v}
                   for k, v in state["done"].items()]
        results += [{"invoice": k, "status": "error", "error": v}
                    for k, v in state["failed"].items()]
        write_outputs(results, output_dir)
        print(f"  Rebuilt outputs from state ({len(results)} invoices).")
        return 0

    # Resumability: choose the work list.
    if args.retry_failed:
        todo = [p for p in inputs if p.name in state["failed"]]
        print(f"  Retry mode: {len(todo)} previously-failed invoice(s)\n")
    else:
        todo = [p for p in inputs if p.name not in state["done"]]
        print(f"  To process: {len(todo)} | already done: {len(state['done'])}\n")

    if not todo:
        print("Nothing to do. Use --retry-failed to re-run errors.")
        return 0

    results, counts = [], {"APPROVE": 0, "REVIEW": 0, "REJECT": 0, "error": 0}
    start = time.time()

    # Phases 1-4 across invoices, in parallel.
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
        futures = {pool.submit(process_invoice, p, args.db): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if r["status"] == "ok":
                counts[r["decision"]] = counts.get(r["decision"], 0) + 1
                state["done"][r["invoice"]] = r["decision"]
                state["failed"].pop(r["invoice"], None)
                nflags = len(r.get("flags", []))
                print(f"  [{i}/{len(todo)}] {r['invoice']:<26} -> {r['decision']:<8} "
                      f"flags={nflags}  ({r['seconds']}s)")
            else:
                counts["error"] += 1
                state["failed"][r["invoice"]] = r.get("error", "error")
                print(f"  [{i}/{len(todo)}] {r['invoice']:<26} -> ERROR  {r.get('error','')[:50]}")
            _save_state(state_path, state)  # durable progress after each invoice

    # Phase 5: consolidate.
    write_outputs(results, output_dir)
    elapsed = time.time() - start
    total_flags = sum(len(r.get("flags", [])) for r in results)

    print("=" * 64)
    print(f"  BATCH COMPLETE in {elapsed:.1f}s "
          f"({elapsed/max(len(todo),1):.1f}s/invoice, {args.max_concurrent} workers)")
    print(f"  APPROVE={counts.get('APPROVE',0)}  REVIEW={counts.get('REVIEW',0)}  "
          f"REJECT={counts.get('REJECT',0)}  ERROR={counts.get('error',0)}")
    print(f"  QA flags raised: {total_flags}")
    print(f"  Results : {output_dir / 'batch_results.json'}")
    print(f"  Summary : {output_dir / 'batch_summary.csv'}")
    print(f"  Flags   : {output_dir / 'batch_flags.csv'}")
    print(f"  State   : {state_path}  (resumable; --retry-failed to re-run errors)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
