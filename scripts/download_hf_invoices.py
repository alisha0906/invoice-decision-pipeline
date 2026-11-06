"""Download real invoice scans + ground truth from HuggingFace.

Target dataset: ``katanaml-org/invoices-donut-data-v1`` (Donut-style invoice
images with a JSON ground-truth target per row). For each row this writes three
sibling files into ``PS-1/sample_data/real_scans/``:

    <id>.png       the invoice image (so it flows through the vision reader)
    <id>.pdf       a single-page PDF wrapping the same image (so it flows
                   through the normal PDF pipeline exactly like the synthetic
                   scenarios do)
    <id>.gt.json   the canonical ground-truth fields parsed from the Donut
                   target (consumed by scripts/eval_extraction.py)

Two download paths are provided:

    * ``datasets`` (primary): ``from datasets import load_dataset`` - the
      official loader; prints the real row count of every split.
    * ``rows``     (fallback): the public datasets-server ``/rows`` HTTP API,
      which needs no auth and no heavy deps; prints per-split ``num_rows`` from
      the ``/size`` endpoint.

``--method auto`` (default) tries ``datasets`` first and falls back to ``rows``.

IMPORTANT: the network/dataset may be unavailable when this runs in CI/offline.
In that case the script prints a clear note and exits 0 - it never fails the
build solely because the dataset could not be fetched. The pure
``parse_ground_truth`` and ``save_row`` helpers are import-safe and unit-tested
offline on inlined samples.
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, Union

# PS-1/scripts/download_hf_invoices.py -> PS-1/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REAL_SCANS_DIR = _PROJECT_ROOT / "sample_data" / "real_scans"

_DATASET = "katanaml-org/invoices-donut-data-v1"
_ROWS_API = "https://datasets-server.huggingface.co/rows"
_SIZE_API = "https://datasets-server.huggingface.co/size"

_OFFLINE_NOTE = (
    "HF dataset unavailable (offline); script is correct, re-run with network "
    "to populate sample_data/real_scans/"
)

# Canonical field keys mirror app.pipeline.providers.FIELD_KEYS.
FIELD_KEYS = ("invoice_number", "date", "total", "vendor", "po_reference", "currency")

_AMOUNT_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _parse_amount(text: Any) -> Optional[float]:
    """Parse a money-ish value into a float, tolerating symbols and both the
    US (1,234.56) and European (1.234,56 / 8,25) number formats.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip()
    # keep only digits, separators and sign
    s = re.sub(r"[^0-9.,\-]", "", s)
    if not s:
        return None
    # Decide which separator is the decimal point.
    if "," in s and "." in s:
        # The rightmost of the two is the decimal separator.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")   # European: 1.234,56
        else:
            s = s.replace(",", "")                      # US: 1,234.56
    elif "," in s:
        # Only commas. If it looks like a decimal (one comma, <=2 trailing
        # digits) treat as decimal; else as thousands.
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            s = s.replace(",", ".")                      # 8,25 -> 8.25
        else:
            s = s.replace(",", "")                       # 1,234 -> 1234
    try:
        return float(s)
    except ValueError:
        return None


def _first_present(data: dict, *keys: str) -> Any:
    """Return the first key present (even if its value is falsy) in ``data``."""
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None


def parse_ground_truth(raw: Union[str, dict]) -> dict:
    """Map a Donut ground-truth target to the canonical invoice field dict.

    Accepts either a JSON string or an already-parsed dict. Unwraps the Donut
    envelopes ``{'gt_parse': {...}}`` and ``{'gt_parses': [{...}, ...]}`` (first
    element). Maps Donut keys to canonical keys:

        invoice_no / invoice_number        -> invoice_number
        invoice_date / date                -> date
        seller / vendor                    -> vendor
        total / total_gross_worth /
            summary.total_gross_worth      -> total   (parsed to float)
        currency                           -> currency (None when absent)
        po_reference                       -> None     (real invoices have none)

    Also returns ``line_items`` from an ``items`` list when present (kept in the
    .gt.json for richer eval). Pure and side-effect free for offline testing.
    """
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            data = {}
    elif isinstance(raw, dict):
        data = raw
    else:
        data = {}

    # Unwrap Donut envelopes.
    if isinstance(data, dict) and "gt_parse" in data and isinstance(data["gt_parse"], dict):
        data = data["gt_parse"]
    elif isinstance(data, dict) and "gt_parses" in data and isinstance(data["gt_parses"], list):
        data = data["gt_parses"][0] if data["gt_parses"] else {}

    if not isinstance(data, dict):
        data = {}

    # This dataset (katanaml-org/invoices-donut-data-v1) nests fields under a
    # 'header' object and totals under 'summary'. Search flat, header, summary.
    header = data.get("header") if isinstance(data.get("header"), dict) else {}
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}

    def _look(*keys):
        for src in (data, header, summary):
            v = _first_present(src, *keys)
            if v is not None:
                return v
        return None

    total_raw = _look("total", "total_gross_worth", "total_net_worth")

    out = {
        "invoice_number": _look("invoice_no", "invoice_number"),
        "date": _look("invoice_date", "date"),
        "total": _parse_amount(total_raw),
        "vendor": _look("seller", "vendor"),
        "po_reference": None,
        "currency": _look("currency"),
    }

    items = data.get("items")
    out["line_items"] = items if isinstance(items, list) else []
    return out


def _image_to_pdf_bytes(image: "Any") -> bytes:
    """Rasterise a PIL image into a single-page PDF (bytes).

    Uses PyMuPDF to wrap the PNG bytes in a page so the result flows through the
    same pipeline the synthetic scenarios use. Imported lazily.
    """
    import pymupdf  # PyMuPDF (fitz)

    png_buf = io.BytesIO()
    image.convert("RGB").save(png_buf, format="PNG")
    png_bytes = png_buf.getvalue()

    pix = pymupdf.Pixmap(io.BytesIO(png_bytes))
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=pix.width, height=pix.height)
        page.insert_image(page.rect, stream=png_bytes)
        return doc.tobytes()
    finally:
        doc.close()


def save_row(out_dir: Path, row_id: str, image: "Any", ground_truth_raw: Union[str, dict]) -> tuple:
    """Write ``<id>.png``, ``<id>.pdf`` and ``<id>.gt.json`` for one row.

    ``image`` is a PIL image. Returns the three written paths in that order.
    Pure of network effects so it is unit-testable offline.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{row_id}.png"
    pdf_path = out_dir / f"{row_id}.pdf"
    gt_path = out_dir / f"{row_id}.gt.json"

    image.convert("RGB").save(str(png_path), format="PNG")

    pdf_path.write_bytes(_image_to_pdf_bytes(image))

    truth = parse_ground_truth(ground_truth_raw)
    with open(gt_path, "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=2)
        fh.write("\n")

    return png_path, pdf_path, gt_path


def download_via_datasets(count: int, out_dir: Path) -> int:
    """Primary path: the official ``datasets`` loader. Returns rows written.

    Raises on ImportError / network / OSError so the caller can fall back.
    """
    from datasets import load_dataset

    ds = load_dataset(_DATASET)

    # Print the REAL per-split row counts.
    print(f"Dataset {_DATASET} splits:")
    for split in ds:
        print(f"  {split}: {len(ds[split])} rows")

    train = ds["train"]
    written = 0
    limit = min(count, len(train))
    for i in range(limit):
        row = train[i]
        row_id = f"hf_train_{i:04d}"
        image = row.get("image") if isinstance(row, dict) else row["image"]
        ground_truth = None
        for key in ("ground_truth", "text"):
            try:
                ground_truth = row[key]
            except (KeyError, TypeError):
                ground_truth = None
            if ground_truth:
                break
        png, pdf, gt = save_row(out_dir, row_id, image, ground_truth or {})
        print(f"  wrote {png.name}, {pdf.name}, {gt.name}")
        written += 1
    return written


def download_via_rows_api(count: int, out_dir: Path) -> int:
    """Fallback path: the public datasets-server ``/rows`` HTTP API.

    Needs no auth and no heavy deps. Returns the number of rows written. Raises
    ``requests.RequestException`` on network failure so the caller can degrade.
    """
    import requests
    from PIL import Image

    # Print per-split row counts from the /size endpoint.
    try:
        size_resp = requests.get(
            _SIZE_API, params={"dataset": _DATASET}, timeout=20
        )
        size_resp.raise_for_status()
        size_body = size_resp.json()
        splits = size_body.get("size", {}).get("splits", [])
        print(f"Dataset {_DATASET} splits:")
        for split in splits:
            print(f"  {split.get('split')}: {split.get('num_rows')} rows")
    except (requests.RequestException, KeyError, ValueError) as exc:
        # Size is informational only; keep going if it fails.
        print(f"  (could not fetch /size: {exc})")

    resp = requests.get(
        _ROWS_API,
        params={
            "dataset": _DATASET,
            "config": "default",
            "split": "train",
            "offset": 0,
            "length": count,
        },
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()

    written = 0
    for entry in body.get("rows", []):
        idx = entry.get("row_idx", written)
        row = entry.get("row", {})
        row_id = f"hf_train_{idx:04d}"

        image_field = row.get("image") or {}
        src = image_field.get("src") if isinstance(image_field, dict) else None
        if not src:
            continue
        img_resp = requests.get(src, timeout=30)
        img_resp.raise_for_status()
        image = Image.open(io.BytesIO(img_resp.content))

        ground_truth = row.get("ground_truth") or row.get("text") or {}
        png, pdf, gt = save_row(out_dir, row_id, image, ground_truth)
        print(f"  wrote {png.name}, {pdf.name}, {gt.name}")
        written += 1
    return written


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download HF invoice scans + ground truth into sample_data/real_scans/."
    )
    parser.add_argument("--count", type=int, default=12, help="number of rows to fetch (default 12)")
    parser.add_argument(
        "--method",
        choices=("auto", "datasets", "rows"),
        default="auto",
        help="download method (default auto: datasets then rows API)",
    )
    args = parser.parse_args(argv)

    out_dir = _REAL_SCANS_DIR

    try:
        if args.method == "datasets":
            written = download_via_datasets(args.count, out_dir)
        elif args.method == "rows":
            written = download_via_rows_api(args.count, out_dir)
        else:  # auto
            try:
                written = download_via_datasets(args.count, out_dir)
            except Exception as exc:  # noqa: BLE001 - any datasets failure -> fallback
                print(f"datasets loader unavailable ({exc}); falling back to rows API...")
                written = download_via_rows_api(args.count, out_dir)
    except Exception as exc:  # noqa: BLE001 - network/import/OS failure => clean no-op
        print(f"{_OFFLINE_NOTE}\n  (reason: {exc.__class__.__name__}: {exc})")
        return 0

    print(f"\nDone: wrote {written} invoice(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
