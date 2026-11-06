"""Validation + decision engine.

``decide`` applies a fixed, ordered sequence of checks and returns a
``(Decision, reasoning_trail)`` pair. The reasoning trail is an ordered list of
human-readable check records (name, passed flag, observed values) so the UI can
show exactly why an invoice was approved, flagged, or rejected.

Rule order (first decisive rule wins):
    0. Extraction unavailable (reader could not read an image-only/scanned
       document) -> REVIEW. This precedes the missing-field check: fields are
       absent because nothing could be *read*, not because the invoice omits
       them. The two failure modes must produce different decisions.
    1. Missing critical field (invoice_number, date, total) -> REJECT
    2. Duplicate invoice (storage.is_duplicate)             -> REJECT
    3. Amount clearly outside tolerance                      -> REJECT
       (borderline within TOLERANCE_BORDERLINE_PCT          -> REVIEW)
    4. Low extraction confidence /
       ambiguous or split match needing confirmation         -> REVIEW
    5. Otherwise                                              -> APPROVE
"""

from typing import Any

from app import config
from app.models import Decision

_CRITICAL_FIELDS = ("invoice_number", "date", "total")


def _check(name: str, passed: bool, observed: dict | None = None) -> dict[str, Any]:
    return {"check": name, "passed": passed, "observed": observed or {}}


def _as_dict(obj: Any) -> dict:
    """Accept either a MatchResult (dataclass) or a plain dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    return dict(getattr(obj, "__dict__", {}))


def decide(fields: dict, extraction: Any, match_result: Any, storage) -> tuple[Decision, list]:
    """Return (Decision, reasoning_trail) for an extracted + matched invoice."""
    trail: list[dict] = []
    fields = fields or {}
    extract = _as_dict(extraction)
    match = _as_dict(match_result)

    # --- 0) Reader could not read the document -> REVIEW --------------------
    # "fields missing because the reader couldn't read the image" is a
    # different failure mode from "the invoice genuinely omits a field", and
    # must route to human REVIEW rather than an automatic missing-field REJECT.
    # extraction_unavailable is the reader-agnostic flag (OCR today, a vision
    # LLM later); ocr_unavailable is accepted as its alias.
    extraction_unavailable = bool(
        extract.get("extraction_unavailable") or extract.get("ocr_unavailable")
    )
    if extraction_unavailable:
        trail.append(
            _check(
                "extraction_available",
                False,
                {
                    "extraction_unavailable": True,
                    "path_used": extract.get("path_used"),
                    "reason": "extraction unavailable for image-only PDF -> REVIEW",
                },
            )
        )
        return Decision.REVIEW, trail

    # --- 1) Missing critical field -> REJECT --------------------------------
    missing = [f for f in _CRITICAL_FIELDS if not fields.get(f) and fields.get(f) != 0]
    passed = not missing
    observed = {"required": list(_CRITICAL_FIELDS), "missing": missing}
    if missing:
        observed["reason"] = f"invoice omits required field {missing[0]} -> REJECT"
    trail.append(_check("critical_fields_present", passed, observed))
    if not passed:
        return Decision.REJECT, trail

    # --- 2) Duplicate invoice -> REJECT -------------------------------------
    is_dupe = storage.is_duplicate(
        fields.get("vendor"),
        fields.get("invoice_number"),
        fields.get("total"),
        fields.get("date"),
    )
    trail.append(
        _check(
            "duplicate_invoice",
            not is_dupe,
            {
                "vendor": fields.get("vendor"),
                "invoice_number": fields.get("invoice_number"),
                "duplicate": is_dupe,
            },
        )
    )
    if is_dupe:
        return Decision.REJECT, trail

    # --- 3) Amount tolerance ------------------------------------------------
    po_amount = match.get("po_amount")
    match_type = match.get("match_type", "NONE")
    # For SPLIT the relevant figure is the cumulative total; else the invoice total.
    observed_amount = (
        match.get("cumulative_amount") if match_type == "SPLIT" else fields.get("total")
    )
    tolerance_check_applicable = po_amount is not None and observed_amount is not None

    if tolerance_check_applicable:
        variance = abs(float(observed_amount) - float(po_amount))
        variance_pct = variance / float(po_amount) if po_amount else 0.0
        tol = config.TOLERANCE_PCT
        borderline_edge = tol + config.TOLERANCE_BORDERLINE_PCT
        within = variance_pct <= tol
        borderline = tol < variance_pct <= borderline_edge

        observed = {
            "invoice_or_cumulative": round(float(observed_amount), 2),
            "po_amount": round(float(po_amount), 2),
            "variance": round(variance, 2),
            "variance_pct": round(variance_pct, 4),
            "tolerance_pct": tol,
        }
        trail.append(_check("amount_within_tolerance", within, observed))

        if not within and not borderline:
            return Decision.REJECT, trail
        if borderline:
            # Borderline over tolerance -> human review with the variance shown.
            trail.append(
                _check(
                    "amount_borderline",
                    False,
                    {"variance_pct": round(variance_pct, 4), "borderline_edge": borderline_edge},
                )
            )
            return Decision.REVIEW, trail
    else:
        trail.append(
            _check(
                "amount_within_tolerance",
                False,
                {"applicable": False, "reason": "no matched PO amount to compare"},
            )
        )

    # --- 4) Confidence / ambiguity -> REVIEW --------------------------------
    # (Reader-unavailable was already handled by rule 0.)
    confidence = float(extract.get("confidence", 0.0) or 0.0)
    low_confidence = confidence < config.LOW_CONFIDENCE_THRESHOLD
    ambiguous = bool(match.get("ambiguous")) or match_type in ("FUZZY", "SPLIT", "NONE")

    needs_review = low_confidence or ambiguous
    trail.append(
        _check(
            "no_review_flags",
            not needs_review,
            {
                "confidence": confidence,
                "low_confidence": low_confidence,
                "match_type": match_type,
                "ambiguous": ambiguous,
            },
        )
    )
    if needs_review:
        return Decision.REVIEW, trail

    # --- 5) APPROVE ---------------------------------------------------------
    trail.append(_check("approved", True, {"match_type": match_type}))
    return Decision.APPROVE, trail
