"""
One-off script: retries the 18 chunks that timed out with Qwen2.5-7B during
the full extraction run (dense multi-citation content correlating with a
grammar-decoding slowdown -- see CLAUDE.md for the observed pattern) using
the 3B model instead, appending results to data/extraction_all.jsonl.

Not a general-purpose retry tool -- the chunk_id list is hardcoded to the
specific chunks that actually failed in that run. This means 18/1370 chunks
(1.3%) in the final dataset were processed by a weaker model than the rest
of the corpus; documented here and in CLAUDE.md rather than silently left
as an unexplained inconsistency.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

FAILED_CHUNK_IDS = [
    "32016R0679_recital_155_0",
    "32016R0679_article_85_0",
    "32019R0881_recital_14_1",
    "32019R0881_article_31_6",
    "32022L2555_recital_31_0",
    "32022L2555_recital_32_0",
    "32022L2555_recital_71_0",
    "32022L2555_recital_100_0",
    "32022L2555_recital_120_1",
    "32022L2555_article_1_0",
    "32022L2555_article_2_11",
    "32022L2555_article_13_3",
    "32022L2555_article_13_4",
    "32022L2555_article_21_1",
    "32022L2555_article_26_0",
    "32022L2555_article_32_3",
    "32022R0868_recital_4_0",
    "32022R0868_article_3_0",
]

CHUNKS_DIR = Path(__file__).parents[1] / "data" / "chunks"
OUT_PATH = Path(__file__).parents[1] / "data" / "extraction_all.jsonl"


def load_chunks_by_id(chunk_ids: set) -> dict:
    found = {}
    for f in CHUNKS_DIR.glob("*_chunks.json"):
        for c in json.loads(f.read_text(encoding="utf-8")):
            if c["chunk_id"] in chunk_ids:
                found[c["chunk_id"]] = c["text"]
    return found


def main() -> None:
    import src.extraction.entity_relation_extractor as ere

    # Point at the 3B model instead of 7B before the lazy singleton loads --
    # _get_llm() reads this module-level global fresh on first call, so
    # reassigning it here (before any extraction call in this process) is
    # enough to redirect it, no need to touch entity_relation_extractor.py.
    ere.MODEL_PATH = Path(__file__).parents[1] / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

    texts = load_chunks_by_id(set(FAILED_CHUNK_IDS))
    missing = set(FAILED_CHUNK_IDS) - texts.keys()
    if missing:
        print(f"WARNING: could not find chunk text for: {missing}")

    with OUT_PATH.open("a", encoding="utf-8") as out:
        for i, chunk_id in enumerate(FAILED_CHUNK_IDS, start=1):
            if chunk_id not in texts:
                continue
            result = ere.extract(texts[chunk_id])
            record = {"chunk_id": chunk_id, **result}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i}/{len(FAILED_CHUNK_IDS)}] {chunk_id} "
                  f"({len(result['entities'])} entities, {len(result['relations'])} relations)")

    print(f"\nDone. Appended to {OUT_PATH}")


if __name__ == "__main__":
    main()
