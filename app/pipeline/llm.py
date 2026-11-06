"""Deterministic text-field fallback.

The main pipeline is vision-only (see ``extract.py``); the vision reader returns
structured fields directly. This module is a tiny offline fallback used by the
accuracy-eval harness when there is only raw text (no vision fields) to work
with - it runs the deterministic RulesProvider, which never fails.
"""

from typing import Any

from app.pipeline.providers import RulesProvider


def extract_fields(raw_text: str) -> tuple[dict[str, Any], str]:
    """Structure invoice fields from raw text. Returns (fields, provider_used)."""
    rules = RulesProvider()
    return rules.extract(raw_text or ""), rules.name
