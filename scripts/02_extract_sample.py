"""Run entity/relation extraction on a sample of chunks for quality review
before committing to the full multi-hour run across all chunks."""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.extraction.entity_relation_extractor import extract

CHUNKS_DIR = Path(__file__).parents[1] / "data" / "chunks"
OUT_PATH = Path(__file__).parents[1] / "data" / "extraction_sample.json"
SAMPLE_SIZE_PER_DOC = 10


def load_sample() -> list[dict]:
    sample = []
    for f in sorted(CHUNKS_DIR.glob("*_chunks.json")):
        chunks = json.loads(f.read_text(encoding="utf-8"))
        # skip trivially short chunks (e.g. bare cross-reference fragments)
        candidates = [c for c in chunks if len(c["text"]) > 150]
        random.seed(42)
        sample.extend(random.sample(candidates, min(SAMPLE_SIZE_PER_DOC, len(candidates))))
    return sample


def main() -> None:
    sample = load_sample()
    print(f"Extracting from {len(sample)} sampled chunks...\n")

    results = []
    entity_type_counts = Counter()
    relation_type_counts = Counter()
    start = time.time()

    for i, chunk in enumerate(sample, start=1):
        t0 = time.time()
        result = extract(chunk["text"])
        elapsed = time.time() - t0

        for e in result["entities"]:
            entity_type_counts[e["type"]] += 1
        for r in result["relations"]:
            relation_type_counts[r["relation"]] += 1

        results.append({"chunk_id": chunk["chunk_id"], **result})
        print(f"[{i}/{len(sample)}] {chunk['chunk_id']} "
              f"({len(result['entities'])} entities, {len(result['relations'])} relations, "
              f"{elapsed:.1f}s)")

    total_elapsed = time.time() - start
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone in {total_elapsed:.0f}s ({total_elapsed / len(sample):.1f}s/chunk avg).")
    print(f"Saved to {OUT_PATH}\n")
    print("Entity types:", dict(entity_type_counts))
    print("Relation types:", dict(relation_type_counts))


if __name__ == "__main__":
    main()
