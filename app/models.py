"""Core data models shared across the pipeline and persistence layers."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Decision(str, Enum):
    """Final routing decision for a processed invoice."""

    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class StageStatus(str, Enum):
    """Outcome of an individual pipeline stage."""

    OK = "OK"
    DEGRADED = "DEGRADED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


@dataclass
class RunRecord:
    """A single invoice processing run.

    `stages` holds per-stage dicts with at least status/confidence/data keys.
    `id` and `timestamp` are assigned by storage on insert when omitted.
    """

    input_filename: str
    extracted_fields: dict = field(default_factory=dict)
    matched_po: Optional[dict] = None
    decision: str = Decision.REVIEW.value
    reasoning_trail: list = field(default_factory=list)
    stages: list = field(default_factory=list)
    id: Optional[int] = None
    timestamp: Optional[str] = None
