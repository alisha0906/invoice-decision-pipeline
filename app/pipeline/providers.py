"""Vision-reader providers.

Two vision backends share a contract:

    available() -> bool                         # usable right now?
    extract_images(list[(bytes, mime)]) -> dict # structured invoice fields

    * OllamaProvider   - a LOCAL vision model (default). No API key, no limits.
    * GeminiVisionProvider - Google Gemini (free AI Studio tier). Hosted.

Both raise ``ProviderError`` on any config/network/parse failure. A small
deterministic ``RulesProvider`` (regex over text) is kept for the offline
accuracy-eval fallback only.

The structured field dict uses these keys (strings or None, except total=float):
    invoice_number, date, total, vendor, po_reference, currency
"""

import base64
import json
import re
from typing import Any, Optional

import requests

from app import config

# Canonical field set returned by every provider.
FIELD_KEYS = ("invoice_number", "date", "total", "vendor", "po_reference", "currency")

_REQUEST_TIMEOUT = 20
# Hosted vision calls (image upload + inference) need a longer budget.
_VISION_TIMEOUT = 90
_VISION_RETRIES = 2
# Local Ollama vision on CPU can be slow (esp. first load); generous timeout.
_OLLAMA_VISION_TIMEOUT = 300

_VISION_PROMPT = (
    "You are an invoice data extractor. Read the invoice IMAGE and "
    "return ONLY a strict JSON object (no prose, no markdown fences) with keys: "
    "invoice_number, date, total, vendor, po_reference, currency. "
    "Use null for any field you cannot find. 'total' must be a number."
)


class ProviderError(Exception):
    """Raised when a provider cannot produce fields (config/network/parse)."""


def _empty_fields() -> dict[str, Any]:
    return {k: None for k in FIELD_KEYS}


def _coerce_fields(data: dict) -> dict[str, Any]:
    """Normalize an arbitrary dict into the canonical field shape."""
    if not isinstance(data, dict):
        raise ProviderError("provider returned non-object JSON")
    out = _empty_fields()
    for key in FIELD_KEYS:
        if key in data:
            out[key] = data[key]
    if out["total"] is not None and not isinstance(out["total"], (int, float)):
        out["total"] = _parse_amount(str(out["total"]))
    return out


def _parse_json_strict(content: str) -> dict:
    """Parse strict JSON, tolerating a surrounding code fence if present."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProviderError(f"JSON parse failure: {exc}") from exc


def _shrink_image(image_bytes: bytes, mime: str, max_dim: int = 1600) -> tuple[bytes, str]:
    """Downscale large images and re-encode as JPEG to shrink the upload.

    Returns (bytes, mime); on any failure returns the original unchanged.
    """
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        scale = max(w, h) / float(max_dim)
        if scale > 1:
            img = img.resize((int(w / scale), int(h / scale)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return image_bytes, mime


# --------------------------------------------------------------------------- #
# Google Gemini vision (free AI Studio tier)
# --------------------------------------------------------------------------- #
def list_gemini_models() -> list[dict]:
    """Return the Gemini models catalog (the ``models`` array)."""
    host = getattr(config, "GEMINI_HOST", "https://generativelanguage.googleapis.com")
    url = f"{host}/v1beta/models"
    headers = {"x-goog-api-key": config.GEMINI_API_KEY}
    try:
        resp = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except (requests.RequestException, KeyError, ValueError) as exc:
        raise ProviderError(f"Gemini /models request failed: {exc}") from exc


def detect_gemini_vision_model(models: Optional[list[dict]] = None) -> Optional[str]:
    """Resolve the Gemini vision model id (newest usable flash model).

    Honors ``config.GEMINI_MODEL`` when present in the live list; otherwise picks
    the newest flash model that supports generateContent (Google gates some older
    models to existing users only, so newest-first avoids 404s).
    """
    if models is None:
        models = list_gemini_models()

    def _supports_generate(m: dict) -> bool:
        return "generateContent" in (m.get("supportedGenerationMethods") or [])

    names = []
    for m in models:
        full = m.get("name") or ""
        bare = full.split("/", 1)[1] if "/" in full else full
        if bare and _supports_generate(m):
            names.append(bare)

    override = getattr(config, "GEMINI_MODEL", "")
    if override and override in names:
        return override

    def _version_key(name: str) -> tuple:
        m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
        if not m:
            return (0, 0)
        return (int(m.group(1)), int(m.group(2)) if m.group(2) else 0)

    _bad = ("image", "tts", "embedding", "thinking", "transcribe", "audio",
            "robotics", "computer-use", "nano", "lyria", "aqa", "deep-research",
            "antigravity", "omni")
    flash = [n for n in names if "flash" in n and not any(b in n for b in _bad)]
    if flash:
        flash.sort(key=_version_key, reverse=True)
        return flash[0]
    general = [n for n in names if not any(b in n for b in _bad)]
    general.sort(key=_version_key, reverse=True)
    return general[0] if general else (names[0] if names else None)


def _gemini_extract_text(body: dict) -> str:
    """Pull generated text from a Gemini response (robust to thinking models)."""
    for cand in (body or {}).get("candidates") or []:
        parts = ((cand or {}).get("content") or {}).get("parts") or []
        texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
        if texts:
            return "\n".join(texts).strip()
    return ""


class GeminiVisionProvider:
    """Vision reader backed by Google Gemini (free AI Studio tier)."""

    name = "gemini-vision"
    _resolved_model: Optional[str] = None

    def available(self) -> bool:
        return bool(config.GEMINI_API_KEY)

    def _model(self) -> str:
        if GeminiVisionProvider._resolved_model:
            return GeminiVisionProvider._resolved_model
        try:
            resolved = detect_gemini_vision_model()
        except ProviderError:
            resolved = None
        model = resolved or config.GEMINI_MODEL
        GeminiVisionProvider._resolved_model = model
        return model

    def extract_image(self, image_bytes: bytes, mime: str = "image/png") -> dict[str, Any]:
        return self.extract_images([(image_bytes, mime)])

    def extract_images(self, images: list) -> dict[str, Any]:
        """Read a (possibly multi-page) invoice: all page images in one call."""
        if not self.available():
            raise ProviderError("GEMINI_API_KEY not configured")
        if not images:
            raise ProviderError("no images to read")
        host = getattr(config, "GEMINI_HOST", "https://generativelanguage.googleapis.com")
        model = self._model()
        url = f"{host}/v1beta/models/{model}:generateContent"

        n = len(images)
        prompt = _VISION_PROMPT
        if n > 1:
            prompt += (
                f"\n\nThis invoice has {n} pages, provided as {n} images in order. "
                "Treat them as ONE invoice: combine line items across pages and "
                "use the final/overall total. Return a single JSON object."
            )
        parts: list = [{"text": prompt}]
        for img_bytes, img_mime in images:
            shrunk, smime = _shrink_image(img_bytes, img_mime)
            parts.append(
                {"inline_data": {"mime_type": smime, "data": base64.b64encode(shrunk).decode()}}
            )
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        headers = {"x-goog-api-key": config.GEMINI_API_KEY}

        import time as _t

        body = None
        attempts = _VISION_RETRIES + 3  # extra tries for 503 overload
        for attempt in range(attempts):
            resp = None
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=_VISION_TIMEOUT)
                resp.raise_for_status()
                body = resp.json()
                break
            except requests.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                detail = ""
                if resp is not None:
                    try:
                        detail = f" | {resp.text[:400]}"
                    except Exception:
                        pass
                # 429 (quota) is not transient - fail fast.
                if status == 429:
                    raise ProviderError(f"Gemini vision request failed: {exc}{detail}") from exc
                # 500/503/timeout -> retry with backoff.
                if attempt < attempts - 1 and (status is None or status >= 500):
                    _t.sleep(min(2 ** attempt, 20))
                    continue
                raise ProviderError(f"Gemini vision request failed: {exc}{detail}") from exc

        content = _gemini_extract_text(body)
        if not content:
            raise ProviderError(f"Gemini returned no text part | {str(body)[:400]}")
        return _coerce_fields(_parse_json_strict(content))


# --------------------------------------------------------------------------- #
# Ollama (local vision + local chat)
# --------------------------------------------------------------------------- #
class OllamaProvider:
    name = "ollama"

    def available(self) -> bool:
        try:
            resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=2)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def supports_vision(self) -> bool:
        low = (config.OLLAMA_MODEL or "").lower()
        return any(h in low for h in ("llava", "vision", "bakllava", "minicpm", "moondream"))

    def extract_image(self, image_bytes: bytes, mime: str = "image/png") -> dict[str, Any]:
        return self.extract_images([(image_bytes, mime)])

    def extract_images(self, images: list) -> dict[str, Any]:
        if not images:
            raise ProviderError("no images to read")
        url = f"{config.OLLAMA_HOST}/api/chat"
        n = len(images)
        prompt = _VISION_PROMPT
        if n > 1:
            prompt += (
                f" This invoice has {n} pages; treat them as ONE invoice and "
                "return a single JSON object."
            )
        # Downscale each page before sending: smaller images are faster AND
        # avoid llava 500s on large/odd-sized inputs. 768px long-edge is a size
        # llava handles reliably while staying readable for invoice fields.
        encoded = []
        for b, m in images:
            shrunk, _ = _shrink_image(b, m, max_dim=768)
            encoded.append(base64.b64encode(shrunk).decode())
        payload = {
            "model": config.OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": encoded,
                }
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 512},
        }
        # Local models can intermittently 500 under load or on a tricky image;
        # retry a few times with a short backoff before giving up.
        import time as _t

        last_exc = None
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=_OLLAMA_VISION_TIMEOUT)
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                return _coerce_fields(_parse_json_strict(content))
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_exc = exc
                if attempt < 2:
                    _t.sleep(2 * (attempt + 1))
                    continue
        raise ProviderError(f"Ollama vision request failed: {last_exc}") from last_exc


# --------------------------------------------------------------------------- #
# Rules (deterministic regex) - offline fallback for the accuracy eval only
# --------------------------------------------------------------------------- #
_AMOUNT_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_CURRENCY_SYMBOLS = {"$": "USD", "\u20ac": "EUR", "\u00a3": "GBP", "\u00a5": "JPY"}


def _parse_amount(text: str) -> Optional[float]:
    m = _AMOUNT_RE.search(text.replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


class RulesProvider:
    name = "rules"

    def available(self) -> bool:
        return True

    def extract(self, raw_text: str) -> dict[str, Any]:
        text = raw_text or ""
        fields = _empty_fields()

        inv = re.search(
            r"invoice\s*(?:no\.?|number|#|num)?\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9\-/]*)",
            text, re.IGNORECASE,
        )
        if inv:
            fields["invoice_number"] = inv.group(1).strip()

        po = re.search(
            r"\b(?:P\.?O\.?|purchase\s*order)\s*(?:no\.?|number|#)?\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9\-/]*)",
            text, re.IGNORECASE,
        )
        if po:
            fields["po_reference"] = po.group(1).strip()

        date = re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", text)
        if date:
            fields["date"] = date.group(1).strip()

        total = re.search(
            r"(?:total|amount\s*due|grand\s*total|balance\s*due)\s*[:#]?\s*"
            r"([$\u20ac\u00a3\u00a5]?\s*[-+]?\d[\d,]*(?:\.\d+)?)",
            text, re.IGNORECASE,
        )
        if total:
            raw_total = total.group(1)
            fields["total"] = _parse_amount(raw_total)
            for sym, code in _CURRENCY_SYMBOLS.items():
                if sym in raw_total:
                    fields["currency"] = code
                    break

        if fields["currency"] is None:
            cur = re.search(r"\b(USD|EUR|GBP|JPY|INR|CAD|AUD)\b", text)
            if cur:
                fields["currency"] = cur.group(1).upper()
            else:
                for sym, code in _CURRENCY_SYMBOLS.items():
                    if sym in text:
                        fields["currency"] = code
                        break

        vendor = re.search(
            r"(?:vendor|supplier|from|bill\s*from|sold\s*by)\s*[:#]?\s*(.+)",
            text, re.IGNORECASE,
        )
        if vendor:
            fields["vendor"] = vendor.group(1).strip().splitlines()[0].strip()

        return fields
