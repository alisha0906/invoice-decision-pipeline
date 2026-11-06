"""Vision-LLM invoice extraction.

Every invoice is read by a vision model (the proven POD approach):

    Phase 1: PDF -> page images (PyMuPDF, 200 DPI, all pages)
    Phase 2: image(s) -> vision LLM -> structured invoice fields

A raw image input (.png/.jpg/...) skips Phase 1 and goes straight to the reader.
The vision reader is selected by ``config.VISION_PROVIDER`` (``ollama`` by
default, ``gemini`` optional). Multi-page PDFs send all pages to the model in
one call so line items / totals / PO refs that span pages are combined.

If the vision reader is unavailable or fails, extraction degrades gracefully to
``extraction_unavailable`` (confidence 0.0) so the decision engine routes the
run to REVIEW rather than crashing. Heavy imports are lazy so importing this
module never fails.
"""

from dataclasses import dataclass, field
from typing import Any

from app import config

# Image file extensions read directly (no PDF render needed).
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
# Safety cap on pages rendered per PDF.
_MAX_PAGES = 10


@dataclass
class ExtractionResult:
    """Outcome of the extraction stage.

    Attributes:
        raw_text: Reserved; vision extraction returns structured fields in meta.
        confidence: 0..1 confidence (0.9 on a successful vision read, 0.0 on fail).
        path_used: 'vision' on success, 'extraction_unavailable' on failure.
        reader: which reader produced the result (e.g. 'ollama', 'gemini-vision').
        meta: extra data incl. 'vision_fields', 'pages', 'error'.
    """

    raw_text: str = ""
    confidence: float = 0.0
    path_used: str = "vision"
    reader: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def extraction_unavailable(self) -> bool:
        """True when the vision reader could not read the document."""
        return self.path_used == "extraction_unavailable"

    # Back-compat alias consumed by the decision engine.
    @property
    def ocr_unavailable(self) -> bool:
        return self.extraction_unavailable

    @property
    def ocr_available(self) -> bool:
        return not self.extraction_unavailable

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "path_used": self.path_used,
            "reader": self.reader,
            "extraction_unavailable": self.extraction_unavailable,
            "ocr_unavailable": self.ocr_unavailable,
            "meta": self.meta,
        }


def _is_image_input(path: str) -> bool:
    """True when the input is a raw image file (not a PDF)."""
    dot = path.rfind(".")
    return dot != -1 and path[dot:].lower() in _IMAGE_SUFFIXES


def _mime_for(path: str) -> str:
    dot = path.rfind(".")
    suffix = path[dot:].lower() if dot != -1 else ""
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }.get(suffix, "image/png")


def _read_image_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


# Keep each rendered page under this many bytes. Oversized images are the main
# cause of vision-reader 500s (llava especially), so we cap size the way the POD
# pipeline does: render, measure, step the DPI down, then fall back to JPEG.
_MAX_IMAGE_BYTES = 1_200_000  # ~1.2 MB per page


def _render_all_pages_png(pdf_path: str, max_pages: int = _MAX_PAGES) -> list[bytes]:
    """Render every PDF page to image bytes, capping each page's byte size.

    POD Phase 1 pattern: render at a starting DPI, and if the image is larger
    than ``_MAX_IMAGE_BYTES`` step the DPI down (and finally re-encode as JPEG)
    until it fits. Oversized images are what make local vision models 500, so
    this keeps every page within a size the reader handles reliably.

    Returns one image blob per page (up to ``max_pages``). Raises only on a
    genuine render failure so the caller can report it.
    """
    import pymupdf  # PyMuPDF (fitz); lazy import

    images: list[bytes] = []
    doc = pymupdf.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            dpi = 150
            data = page.get_pixmap(dpi=dpi).tobytes("png")
            # Step the DPI down while the PNG is too large.
            while len(data) > _MAX_IMAGE_BYTES and dpi > 80:
                dpi -= 25
                data = page.get_pixmap(dpi=dpi).tobytes("png")
            # Last resort: JPEG is much smaller for dense/scanned pages.
            if len(data) > _MAX_IMAGE_BYTES:
                data = page.get_pixmap(dpi=dpi).tobytes("jpg")
            images.append(data)
    finally:
        doc.close()
    return images


def _unavailable(reason: str) -> ExtractionResult:
    """Graceful degradation: route the run to REVIEW instead of crashing."""
    return ExtractionResult(
        confidence=0.0,
        path_used="extraction_unavailable",
        reader="none",
        meta={"error": reason},
    )


def _select_provider():
    """Return (provider, reader_label, key_hint) for the configured reader."""
    from app.pipeline import providers

    choice = getattr(config, "VISION_PROVIDER", "ollama").lower()
    if choice == "gemini":
        return providers.GeminiVisionProvider(), "gemini-vision", "GEMINI_API_KEY"
    return providers.OllamaProvider(), "ollama", "OLLAMA_MODEL/ollama server"


def extract(path: str) -> ExtractionResult:
    """Read an invoice (PDF or image) with the configured vision LLM.

    Renders a PDF to page images (all pages) and sends them to the vision reader;
    an image input is sent directly. Never raises for a reader problem - returns
    an ``extraction_unavailable`` result so downstream logic forces a REVIEW.
    """
    from app.pipeline import providers

    # Phase 1: gather page images as (bytes, mime) tuples.
    if _is_image_input(path):
        try:
            pages = [(_read_image_bytes(path), _mime_for(path))]
        except OSError as exc:
            return _unavailable(f"could not read image: {exc}")
    else:
        try:
            page_blobs = _render_all_pages_png(path)
        except Exception as exc:
            return _unavailable(
                f"PDF->image render failed ({exc.__class__.__name__}): {exc}"
            )
        if not page_blobs:
            return _unavailable("PDF rendered to zero pages")
        pages = [(b, "image/png") for b in page_blobs]

    # Phase 2: read with the configured vision provider.
    provider, reader_label, key_hint = _select_provider()

    available = provider.available()
    # Ollama additionally needs the configured model to be a vision model.
    if reader_label == "ollama" and available and not provider.supports_vision():
        return _unavailable(
            f"OLLAMA_MODEL '{config.OLLAMA_MODEL}' is not a vision model "
            "(name must contain llava/vision/moondream/...)"
        )
    if not available:
        return _unavailable(f"{reader_label} unavailable (check {key_hint})")

    try:
        fields = provider.extract_images(pages)
    except providers.ProviderError as exc:
        return _unavailable(f"{reader_label} call failed: {exc}")

    return ExtractionResult(
        confidence=0.9,
        path_used="vision",
        reader=reader_label,
        meta={"reader": reader_label, "vision_fields": fields, "pages": len(pages)},
    )
