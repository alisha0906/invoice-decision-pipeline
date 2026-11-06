"""Verify the configured vision reader works on the real sample invoices.

Run from the PS-1 folder:  python scripts/verify_vision.py
Writes results to scripts/verify_vision.out.txt.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

out_lines = []


def log(msg=""):
    print(msg)
    out_lines.append(str(msg))


from app import config
from app.pipeline import providers
from app.pipeline.extract import extract

vp = getattr(config, "VISION_PROVIDER", "ollama").lower()
log(f"EXTRACTION_MODE = {config.EXTRACTION_MODE}")
log(f"VISION_PROVIDER = {vp}")
log(f"OLLAMA_MODEL    = {config.OLLAMA_MODEL}  host = {config.OLLAMA_HOST}")
log(f"GEMINI_API_KEY present = {bool(config.GEMINI_API_KEY)}  model = {config.GEMINI_MODEL or '(auto-detect)'}")

# 1) Confirm the configured vision provider is reachable.
log(f"\nConfigured vision reader = {vp}")
if vp == "gemini":
    p = providers.GeminiVisionProvider()
    log(f"GeminiVisionProvider.available() = {p.available()}")
    if not p.available():
        log("  -> set GEMINI_API_KEY in .env (free key from https://aistudio.google.com)")
else:  # ollama (default)
    o = providers.OllamaProvider()
    log(f"OllamaProvider.available() = {o.available()}  (is the ollama server running?)")
    log(f"OllamaProvider.supports_vision() = {o.supports_vision()}  (OLLAMA_MODEL must be a vision model)")
    if not o.available():
        log("  -> start Ollama, then: ollama pull llava-phi3  (and set OLLAMA_MODEL)")

# 2) Run the real HF invoices through the pipeline.
# Test on PDFs (what the UI actually feeds), so the full PDF->image->vision path
# is exercised; fall back to the PNG only if a PDF isn't present for that id.
real = ROOT / "sample_data" / "real_scans"
inputs = []
if real.exists():
    by_id = {}
    for pdf in real.glob("*.pdf"):
        by_id[pdf.stem] = pdf
    for img in list(real.glob("*.png")) + list(real.glob("*.jpg")):
        by_id.setdefault(img.stem, img)
    inputs = [by_id[k] for k in sorted(by_id)]

if not inputs:
    log("\nNo real_scans invoices found. Run: python scripts/download_hf_invoices.py")

for f in inputs:
    log(f"\n=== {f.name} ===")
    try:
        res = extract(str(f))
        log(f"  path_used = {res.path_used}")
        log(f"  reader    = {res.reader}")
        log(f"  pages     = {res.meta.get('pages')}")
        log(f"  confidence= {res.confidence}")
        log(f"  fields    = {res.meta.get('vision_fields')}")
        if res.meta.get("error"):
            log(f"  error     = {res.meta.get('error')}")
    except Exception as exc:
        log(f"  EXTRACT RAISED: {exc.__class__.__name__}: {exc}")

(Path(__file__).resolve().parent / "verify_vision.out.txt").write_text(
    "\n".join(out_lines), encoding="utf-8"
)
print("DONE - see scripts/verify_vision.out.txt")
