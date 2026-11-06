"""Tests for the vision-only extract() orchestration (mocked, offline).

extract() renders a PDF to page images and sends them to the configured vision
provider; a raw image is sent directly. When the provider is unavailable or
fails, extract() degrades to an ``extraction_unavailable`` result (never raises).
"""

import pytest
from PIL import Image

from app.pipeline import extract as extract_mod
from app.pipeline import providers as providers_mod
from app.pipeline.extract import extract

_FAKE_FIELDS = {
    "invoice_number": "INV-VISION",
    "date": "2024-06-01",
    "total": 1234.5,
    "vendor": "Vision Vendor",
    "po_reference": "PO-5555",
    "currency": "USD",
}


def _make_png(path: str) -> None:
    Image.new("RGB", (64, 64), color=(255, 255, 255)).save(path)


@pytest.fixture(autouse=True)
def _vision_ollama(monkeypatch):
    """Default to Ollama provider, available, returning fake fields."""
    monkeypatch.setattr(providers_mod.config, "VISION_PROVIDER", "ollama", raising=False)
    monkeypatch.setattr(providers_mod.config, "OLLAMA_MODEL", "llava-phi3", raising=False)
    monkeypatch.setattr(providers_mod.OllamaProvider, "available", lambda self: True)
    monkeypatch.setattr(providers_mod.OllamaProvider, "supports_vision", lambda self: True)
    monkeypatch.setattr(
        providers_mod.OllamaProvider,
        "extract_images",
        lambda self, images: dict(_FAKE_FIELDS),
    )
    yield


def test_png_input_reads_via_vision(tmp_path):
    png = str(tmp_path / "scan.png")
    _make_png(png)

    result = extract(png)

    assert result.path_used == "vision"
    assert result.reader == "ollama"
    assert result.meta.get("vision_fields") == _FAKE_FIELDS
    assert result.extraction_unavailable is False


def test_pdf_input_renders_then_reads_via_vision(tmp_path, monkeypatch):
    pdf = str(tmp_path / "invoice.pdf")
    # Stub the PyMuPDF rasteriser: we assert orchestration, not PyMuPDF itself.
    monkeypatch.setattr(
        extract_mod, "_render_all_pages_png", lambda p, max_pages=10: [b"\x89PNG p1"]
    )

    result = extract(pdf)

    assert result.path_used == "vision"
    assert result.reader == "ollama"
    assert result.meta.get("pages") == 1
    assert result.meta.get("vision_fields") == _FAKE_FIELDS


def test_multipage_pdf_sends_all_pages(tmp_path, monkeypatch):
    pdf = str(tmp_path / "multi.pdf")
    monkeypatch.setattr(
        extract_mod,
        "_render_all_pages_png",
        lambda p, max_pages=10: [b"p1", b"p2", b"p3"],
    )
    seen = {}
    monkeypatch.setattr(
        providers_mod.OllamaProvider,
        "extract_images",
        lambda self, images: seen.update(n=len(images)) or dict(_FAKE_FIELDS),
    )

    result = extract(pdf)

    assert result.meta.get("pages") == 3
    assert seen["n"] == 3  # all pages forwarded to the vision reader


def test_provider_unavailable_degrades_to_review(tmp_path, monkeypatch):
    png = str(tmp_path / "scan.png")
    _make_png(png)
    monkeypatch.setattr(providers_mod.OllamaProvider, "available", lambda self: False)

    result = extract(png)  # must not raise

    assert result.extraction_unavailable is True
    assert result.path_used == "extraction_unavailable"
    assert result.reader == "none"
    assert result.meta.get("error")


def test_provider_error_degrades_to_review(tmp_path, monkeypatch):
    png = str(tmp_path / "scan.png")
    _make_png(png)

    def _boom(self, images):
        raise providers_mod.ProviderError("simulated 503 overload")

    monkeypatch.setattr(providers_mod.OllamaProvider, "extract_images", _boom)

    result = extract(png)  # must not raise

    assert result.extraction_unavailable is True
    assert "503" in result.meta.get("error", "")


def test_non_vision_ollama_model_degrades(tmp_path, monkeypatch):
    png = str(tmp_path / "scan.png")
    _make_png(png)
    monkeypatch.setattr(providers_mod.OllamaProvider, "supports_vision", lambda self: False)

    result = extract(png)

    assert result.extraction_unavailable is True
    assert "not a vision model" in result.meta.get("error", "")
