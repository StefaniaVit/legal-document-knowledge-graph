"""Run entity/relation extraction across every chunk in data/chunks/.

Writes one JSON line per chunk to data/extraction_all.jsonl as it goes,
rather than holding everything in memory and writing once at the end --
at multiple hours for the full corpus, a crash or interruption partway
through should lose minutes of work, not the whole run. Already-processed
chunk_ids are skipped on a subsequent run, so this script is safe to
re-invoke after a stop.

Extraction goes through TimeoutSafeExtractor, not entity_relation_extractor
.extract() directly -- with Qwen2.5-7B, an occasional call (~2.5% of chunks
in testing) took 5-10 minutes instead of the usual 5-20 seconds, for
reasons not fully root-caused (one trigger, a few-shot example the model
would sometimes regurgitate, was found and fixed, but a rarer residual
occurrence of the same symptom remained). Rather than let one pathological
call block the whole run, each is capped at a hard timeout and the chunk
is skipped (recorded as FAILED, same as any other error) if it doesn't
recover after one retry against a fresh worker process.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.extraction.timeout_worker import TimeoutSafeExtractor

CHUNKS_DIR = Path(__file__).parents[1] / "data" / "chunks"
OUT_PATH = Path(__file__).parents[1] / "data" / "extraction_all.jsonl"
CALL_TIMEOUT_SECONDS = 60


def load_all_chunks() -> list[dict]:
    chunks = []
    for f in sorted(CHUNKS_DIR.glob("*_chunks.json")):
        chunks.extend(json.loads(f.read_text(encoding="utf-8")))
    return chunks


def already_done() -> set:
    if not OUT_PATH.exists():
        return set()
    done = set()
    with OUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["chunk_id"])
    return done


def main() -> None:
    chunks = load_all_chunks()
    done = already_done()
    remaining = [c for c in chunks if c["chunk_id"] not in done]

    print(f"{len(chunks)} total chunks, {len(done)} already done, "
          f"{len(remaining)} remaining.\n")

    extractor = TimeoutSafeExtractor(timeout=CALL_TIMEOUT_SECONDS)
    start = time.time()
    with OUT_PATH.open("a", encoding="utf-8") as out:
        for i, chunk in enumerate(remaining, start=1):
            t0 = time.time()
            try:
                result = extractor.extract(chunk["text"])
            except Exception as e:
                print(f"[{i}/{len(remaining)}] {chunk['chunk_id']} FAILED: {e}")
                continue
            elapsed = time.time() - t0

            record = {"chunk_id": chunk["chunk_id"], **result}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            if i % 10 == 0 or i == len(remaining):
                avg = (time.time() - start) / i
                eta_min = avg * (len(remaining) - i) / 60
                print(f"[{i}/{len(remaining)}] {chunk['chunk_id']} "
                      f"({len(result['entities'])} entities, {len(result['relations'])} relations, "
                      f"{elapsed:.1f}s) -- avg {avg:.1f}s/chunk, ETA {eta_min:.0f}min")

    extractor.close()
    print(f"\nDone. Results in {OUT_PATH}")


if __name__ == "__main__":
    main()
