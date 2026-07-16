"""Build a small hand-labeling sample from already-extracted data, formatted
as CSV for easy review in Excel/Numbers/Sheets rather than editing raw JSON.

Picks ~5 chunks per document from data/extraction_all.jsonl (no LLM calls --
reuses extraction already run), preferring chunks with a reasonable number
of entities (2-6) so review is meaningful without being overwhelming, and
mixing recital/article chunk types.

Writes two review files:
  data/labeled/entities_review.csv   -- one row per candidate entity
  data/labeled/relations_review.csv  -- one row per candidate relation

Both leave blank columns for a human reviewer to fill in.
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

EXTRACTION_PATH = Path(__file__).parents[1] / "data" / "extraction_all.jsonl"
CHUNKS_DIR = Path(__file__).parents[1] / "data" / "chunks"
LABELED_DIR = Path(__file__).parents[1] / "data" / "labeled"

PER_DOCUMENT = 5
MIN_ENTITIES, MAX_ENTITIES = 2, 6


def celex_of(chunk_id: str) -> str:
    return chunk_id.split("_recital_")[0].split("_article_")[0]


def load_chunk_texts() -> dict:
    texts = {}
    for f in CHUNKS_DIR.glob("*_chunks.json"):
        for c in json.loads(f.read_text(encoding="utf-8")):
            texts[c["chunk_id"]] = c["text"]
    return texts


def pick_sample(records: list[dict]) -> list[dict]:
    by_doc = defaultdict(list)
    for r in records:
        if MIN_ENTITIES <= len(r["entities"]) <= MAX_ENTITIES:
            by_doc[celex_of(r["chunk_id"])].append(r)

    random.seed(42)
    sample = []
    for celex, recs in by_doc.items():
        recitals = [r for r in recs if "_recital_" in r["chunk_id"]]
        articles = [r for r in recs if "_article_" in r["chunk_id"]]
        half = PER_DOCUMENT // 2
        picked = (
            random.sample(recitals, min(half, len(recitals)))
            + random.sample(articles, min(PER_DOCUMENT - half, len(articles)))
        )
        sample.extend(picked[:PER_DOCUMENT])
    return sample


def main() -> None:
    records = [json.loads(l) for l in EXTRACTION_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunk_texts = load_chunk_texts()
    sample = pick_sample(records)

    LABELED_DIR.mkdir(parents=True, exist_ok=True)

    entities_path = LABELED_DIR / "entities_review.csv"
    with entities_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "chunk_text", "entity_text", "entity_type",
                          "is_correct(Y/N)", "corrected_type", "notes"])
        for r in sample:
            text = chunk_texts.get(r["chunk_id"], "")
            for e in r["entities"]:
                writer.writerow([r["chunk_id"], text, e["text"], e["type"], "", "", ""])

    relations_path = LABELED_DIR / "relations_review.csv"
    with relations_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "chunk_text", "subject", "relation", "object",
                          "is_correct(Y/N)", "notes"])
        for r in sample:
            text = chunk_texts.get(r["chunk_id"], "")
            for rel in r["relations"]:
                writer.writerow([r["chunk_id"], text, rel["subject"], rel["relation"], rel["object"], "", ""])

    print(f"{len(sample)} chunks sampled across {len({celex_of(r['chunk_id']) for r in sample})} documents")
    print(f"Wrote {entities_path} ({sum(len(r['entities']) for r in sample)} rows)")
    print(f"Wrote {relations_path} ({sum(len(r['relations']) for r in sample)} rows)")
    print("\nFill in 'is_correct(Y/N)' for each row (and 'corrected_type' where wrong), "
          "then let me know when you're done.")
    print("If you notice an entity/relation that's missing entirely (a false negative), "
          "add a new row by hand with notes=\"MISSED\".")


if __name__ == "__main__":
    main()
