"""Check PDFs are in place, then parse all of them."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ingestion.eurlex_fetcher import check_downloads, ensure_meta
from src.ingestion.parser import parse_all

if __name__ == "__main__":
    print("=== EUR-Lex document status ===\n")
    status = check_downloads()
    ready = sum(status.values())

    if ready == 0:
        print("\nNo PDFs found. Download them from the URLs above and place each at:")
        print("  data/raw/{celex}/document.pdf")
        sys.exit(1)

    ensure_meta()

    print(f"\n=== Parsing {ready} document(s) ===\n")
    docs = parse_all()
    print(f"\nDone. {len(docs)} documents saved to data/processed/")
