"""Offline, mocked tests for the vision providers (Gemini + Ollama).

No API key and no network: ``requests`` is monkeypatched throughout.
"""

import pytest

from app.pipeline import providers
from app.pipeline.providers import (
    GeminiVisionProvider,
    OllamaProvider,
    ProviderError,
)


class _FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


# --- Gemini model autodetection -------------------------------------------
def test_detect_gemini_model_honours_present_override(monkeypatch):
    monkeypatch.setattr(providers.config, "GEMINI_MODEL", "gemini-3.8-flash")
    models = [
        {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
    ]
    assert providers.detect_gemini_vision_model(models) == "gemini-3.8-flash"


def test_detect_gemini_model_picks_newest_flash(monkeypatch):
    monkeypatch.setattr(providers.config, "GEMINI_MODEL", "")
    models = [
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/gemini-2.5-flash-image", "supportedGenerationMethods": ["generateContent"]},
    ]
    # Newest flash, excluding the image-gen variant.
    assert providers.detect_gemini_vision_model(models) == "gemini-3.8-flash"


def test_list_gemini_models_wraps_errors(monkeypatch):
    monkeypatch.setattr(providers.config, "GEMINI_API_KEY", "test-key")

    def _bad_get(*_a, **_k):
        raise providers.requests.RequestException("no network")

    monkeypatch.setattr(providers.requests, "get", _bad_get)
    with pytest.raises(ProviderError):
        providers.list_gemini_models()


# --- Gemini vision extraction (mocked) -------------------------------------
def test_gemini_extract_images_parses_response(monkeypatch):
    monkeypatch.setattr(providers.config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(providers.config, "GEMINI_MODEL", "gemini-3.8-flash")
    GeminiVisionProvider._resolved_model = "gemini-3.8-flash"

    content = (
        '{"invoice_number": "INV-9001", "date": "2024-05-01", '
        '"total": 2499.5, "vendor": "Globex LLC", '
        '"po_reference": "PO-7777", "currency": "USD"}'
    )
    payload = {"candidates": [{"content": {"parts": [{"text": content}]}}]}

    captured = {}

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse(payload)

    monkeypatch.setattr(providers.requests, "post", _fake_post)

    fields = GeminiVisionProvider().extract_images([(b"\x89PNG", "image/png")])
    assert fields["invoice_number"] == "INV-9001"
    assert fields["total"] == 2499.5
    assert fields["vendor"] == "Globex LLC"
    # One text part + one image part in the request.
    parts = captured["json"]["contents"][0]["parts"]
    assert any("text" in p for p in parts)
    assert any("inline_data" in p for p in parts)


def test_gemini_no_key_raises(monkeypatch):
    monkeypatch.setattr(providers.config, "GEMINI_API_KEY", "")
    GeminiVisionProvider._resolved_model = None
    with pytest.raises(ProviderError):
        GeminiVisionProvider().extract_images([(b"bytes", "image/png")])


# --- Ollama vision ---------------------------------------------------------
def test_ollama_supports_vision_by_model_name(monkeypatch):
    monkeypatch.setattr(providers.config, "OLLAMA_MODEL", "llava-phi3")
    assert OllamaProvider().supports_vision() is True
    monkeypatch.setattr(providers.config, "OLLAMA_MODEL", "llama3.1")
    assert OllamaProvider().supports_vision() is False


def test_ollama_extract_images_parses_response(monkeypatch):
    monkeypatch.setattr(providers.config, "OLLAMA_MODEL", "llava-phi3")
    content = '{"invoice_number": "A1", "date": "2024-01-02", "total": 10.0, "vendor": "ACME"}'
    payload = {"message": {"content": content}}

    def _fake_post(url, json=None, timeout=None):
        return _FakeResponse(payload)

    monkeypatch.setattr(providers.requests, "post", _fake_post)
    fields = OllamaProvider().extract_images([(b"img", "image/png")])
    assert fields["invoice_number"] == "A1"
    assert fields["total"] == 10.0
