"""
EUR-Lex document registry and (optional) downloader.

EUR-Lex blocks automated HTTP downloads with AWS WAF.
Place PDFs manually in data/raw/{celex}/document.pdf

  PDF URL pattern:
  https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:{celex}
"""
import json
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"

SAMPLE_DOCUMENTS = {
    "32016R0679": "GDPR - General Data Protection Regulation",
    "32019R0881": "Cybersecurity Act",
    "32022L2555": "NIS2 Directive",
    "32022R0868": "Data Governance Act",
}


def pdf_url(celex: str) -> str:
    return f"https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:{celex}"


def check_downloads() -> dict[str, bool]:
    """Report which documents have been placed in data/raw/."""
    status = {}
    for celex, title in SAMPLE_DOCUMENTS.items():
        pdf_path = RAW_DIR / celex / "document.pdf"
        status[celex] = pdf_path.exists()
        mark = "✓" if status[celex] else "✗"
        print(f"  {mark} {celex}  {title}")
        if not status[celex]:
            print(f"      → {pdf_url(celex)}")
    return status


def ensure_meta() -> None:
    """Write meta.json sidecars for any PDF that exists but has no meta."""
    for celex, title in SAMPLE_DOCUMENTS.items():
        doc_dir = RAW_DIR / celex
        pdf_path = doc_dir / "document.pdf"
        meta_path = doc_dir / "meta.json"
        if pdf_path.exists() and not meta_path.exists():
            doc_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "celex": celex,
                "title": title,
                "pdf_url": pdf_url(celex),
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    print("EUR-Lex document download status:\n")
    status = check_downloads()
    found = sum(status.values())
    print(f"\n{found}/{len(SAMPLE_DOCUMENTS)} documents ready.")
    ensure_meta()
