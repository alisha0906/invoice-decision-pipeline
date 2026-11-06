"""Purchase-order matching.

Loads the approved PO catalogue and matches an extracted invoice against it:

    EXACT  - the invoice po_reference equals a known PO number.
    FUZZY  - no usable po_reference: match by vendor name similarity
             (difflib ratio) plus amount within tolerance.
    SPLIT  - an EXACT-matched PO whose cumulative invoiced amount (prior
             ledger entries + this invoice) must be checked against the PO
             total; flagged when more than one invoice bills the same PO.
    NONE   - nothing matched.

``match_invoice(fields, storage)`` returns a ``MatchResult``; ``as_dict`` gives
a JSON-serializable shape for persistence and the API layer.
"""

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app import config

_PO_PATH = Path(__file__).resolve().parent.parent.parent / "sample_data" / "purchase_orders.json"

# Vendor-name similarity required for a FUZZY match.
_VENDOR_RATIO_THRESHOLD = 0.6


@dataclass
class MatchResult:
    match_type: str = "NONE"  # EXACT | FUZZY | SPLIT | NONE
    po: Optional[dict] = None
    ambiguous: bool = False
    confidence: float = 0.0
    cumulative_amount: Optional[float] = None  # SPLIT: total billed incl. current
    po_amount: Optional[float] = None
    within_tolerance: Optional[bool] = None
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "match_type": self.match_type,
            "po": self.po,
            "ambiguous": self.ambiguous,
            "confidence": self.confidence,
            "cumulative_amount": self.cumulative_amount,
            "po_amount": self.po_amount,
            "within_tolerance": self.within_tolerance,
            "notes": self.notes,
        }


def load_purchase_orders(path: Optional[str] = None) -> list[dict]:
    """Load the approved PO catalogue from JSON."""
    po_path = Path(path) if path else _PO_PATH
    with open(po_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("purchase_orders", [])


def _tolerance_for(po: dict) -> float:
    """Relative tolerance for a PO, defaulting to the global config value."""
    try:
        return float(po.get("tolerance_pct", config.TOLERANCE_PCT))
    except (TypeError, ValueError):
        return config.TOLERANCE_PCT


def _amount_within_tolerance(amount: float, target: float, tolerance_pct: float) -> bool:
    if amount is None or target is None:
        return False
    allowed = max(abs(target) * tolerance_pct, config.TOLERANCE_ABS)
    return abs(amount - target) <= allowed


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def match_invoice(fields: dict, storage, pos: Optional[list[dict]] = None) -> MatchResult:
    """Match extracted invoice fields against the PO catalogue.

    ``storage`` supplies cumulative_for_po for split-PO tracking.
    """
    if pos is None:
        pos = load_purchase_orders()

    po_ref = (fields.get("po_reference") or "").strip()
    amount = _to_float(fields.get("total"))
    vendor = (fields.get("vendor") or "").strip()

    # --- EXACT / SPLIT: po_reference matches a known PO number --------------
    if po_ref:
        for po in pos:
            if str(po.get("po_number", "")).strip().lower() == po_ref.lower():
                tol = _tolerance_for(po)
                po_amount = _to_float(po.get("amount"))
                prior = float(storage.cumulative_for_po(po["po_number"]))
                current = amount or 0.0
                cumulative = prior + current

                # Multiple invoices against one PO => SPLIT: check cumulative.
                if prior > 0:
                    within = _amount_within_tolerance(cumulative, po_amount, tol)
                    return MatchResult(
                        match_type="SPLIT",
                        po=po,
                        ambiguous=True,
                        confidence=0.9,
                        cumulative_amount=cumulative,
                        po_amount=po_amount,
                        within_tolerance=within,
                        notes=[
                            f"prior invoiced {prior:.2f} + current {current:.2f} "
                            f"= {cumulative:.2f} vs PO {po_amount:.2f}"
                        ],
                    )

                within = _amount_within_tolerance(current, po_amount, tol)
                return MatchResult(
                    match_type="EXACT",
                    po=po,
                    ambiguous=False,
                    confidence=1.0,
                    cumulative_amount=cumulative,
                    po_amount=po_amount,
                    within_tolerance=within,
                    notes=[f"po_reference {po_ref} matched PO {po['po_number']}"],
                )

    # --- FUZZY: no usable po_reference -> vendor + amount -------------------
    if vendor:
        best_po = None
        best_ratio = 0.0
        for po in pos:
            ratio = difflib.SequenceMatcher(
                None, vendor.lower(), str(po.get("vendor", "")).lower()
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_po = po

        if best_po is not None and best_ratio >= _VENDOR_RATIO_THRESHOLD:
            tol = _tolerance_for(best_po)
            po_amount = _to_float(best_po.get("amount"))
            within = _amount_within_tolerance(amount, po_amount, tol)
            if within:
                return MatchResult(
                    match_type="FUZZY",
                    po=best_po,
                    ambiguous=True,
                    confidence=round(best_ratio, 3),
                    cumulative_amount=amount,
                    po_amount=po_amount,
                    within_tolerance=True,
                    notes=[
                        f"vendor '{vendor}' ~ '{best_po['vendor']}' "
                        f"(ratio {best_ratio:.2f}), amount within tolerance"
                    ],
                )

    # --- NONE ---------------------------------------------------------------
    return MatchResult(
        match_type="NONE",
        po=None,
        ambiguous=False,
        confidence=0.0,
        notes=["no PO matched by reference or vendor+amount"],
    )
