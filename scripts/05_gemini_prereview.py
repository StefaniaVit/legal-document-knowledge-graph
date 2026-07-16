"""
Uses Gemini as a fast first-pass judge on the hand-labeling sample (see
04_prepare_labeling_sample.py), pre-filling is_correct/corrected_type/notes
for every row in place.

This is a legitimate technique -- LLM-as-judge, Gemini judging output from
the separate, weaker local Qwen2.5-3B model that actually did the extraction,
not the model grading its own homework -- but it is NOT the same as human
ground truth: it measures agreement between two LLMs, which can share blind
spots. The CSVs this produces are a draft to review, not the final label.
Run this, then spot-check the shortlist it prints at the end (every row
Gemini flagged wrong, plus a random sample of the ones it marked correct)
before treating the numbers as real precision/recall.
"""
import csv
import os
import random
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.extraction.entity_relation_extractor import ENTITY_TYPES, RELATION_TYPES

load_dotenv()

LABELED_DIR = Path(__file__).parents[1] / "data" / "labeled"
GEMINI_MODEL = "gemini-2.5-flash-lite"
SPOT_CHECK_FRACTION = 0.2
# Free tier for gemini-2.5-flash is 5 requests/minute (not the ~1500/day figure
# that applies to other tiers/models) -- 13s keeps every call safely under that.
CALL_DELAY_SECONDS = 13.0
RATE_LIMIT_BACKOFF_SECONDS = 65.0

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


class EntityJudgment(BaseModel):
    is_correct: bool
    corrected_type: Optional[str] = None
    notes: str


class RelationJudgment(BaseModel):
    is_correct: bool
    notes: str


def _call_with_retry(prompt: str, schema):
    for attempt in range(6):
        try:
            response = _get_client().models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema),
            )
            return schema.model_validate_json(response.text)
        except Exception as e:
            if attempt == 5:
                raise
            transient = ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                         or "503" in str(e) or "UNAVAILABLE" in str(e))
            wait = RATE_LIMIT_BACKOFF_SECONDS if transient else 5 * (attempt + 1)
            print(f"  retrying in {wait:.0f}s after error: {e}")
            time.sleep(wait)


def judge_entity(chunk_text: str, entity_text: str, entity_type: str) -> EntityJudgment:
    prompt = (
        "You are reviewing an automated entity-extraction pipeline's output for a legal "
        "knowledge graph. Judge whether this extracted entity is correct: a genuine, "
        "reasonably concise named thing from the text, correctly typed.\n"
        f"Valid types: {', '.join(ENTITY_TYPES)}.\n\n"
        f"Source text:\n{chunk_text}\n\n"
        f'Extracted entity: "{entity_text}" (type: {entity_type})\n\n'
        "If the type is wrong but the entity itself is valid, set is_correct=false and give "
        "corrected_type. If the entity span itself is bad (too long, garbled, not a real "
        "entity, or not present in the text), set is_correct=false and explain in notes."
    )
    return _call_with_retry(prompt, EntityJudgment)


def judge_relation(chunk_text: str, subject: str, relation: str, obj: str) -> RelationJudgment:
    prompt = (
        "You are reviewing an automated relation-extraction pipeline's output for a legal "
        "knowledge graph. Judge whether this extracted relation is correct given the source text.\n"
        f"Valid relation types: {', '.join(RELATION_TYPES)}.\n\n"
        f"Source text:\n{chunk_text}\n\n"
        f'Extracted relation: "{subject}" --{relation}--> "{obj}"\n\n'
        "Set is_correct=false if the relation type is wrong, the direction is reversed, or "
        "the relationship isn't actually supported by the text. Explain in notes."
    )
    return _call_with_retry(prompt, RelationJudgment)


def _save(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def review_entities() -> list[dict]:
    path = LABELED_DIR / "entities_review.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for i, row in enumerate(rows, start=1):
        if row["is_correct(Y/N)"].strip():
            continue  # already judged in a previous (possibly interrupted) run
        j = judge_entity(row["chunk_text"], row["entity_text"], row["entity_type"])
        row["is_correct(Y/N)"] = "Y" if j.is_correct else "N"
        row["corrected_type"] = j.corrected_type or ""
        row["notes"] = j.notes
        print(f"  [{i}/{len(rows)}] {row['entity_text'][:40]!r} -> {row['is_correct(Y/N)']}")
        _save(path, rows)
        time.sleep(CALL_DELAY_SECONDS)
    return rows


def review_relations() -> list[dict]:
    path = LABELED_DIR / "relations_review.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for i, row in enumerate(rows, start=1):
        if row["is_correct(Y/N)"].strip():
            continue  # already judged in a previous (possibly interrupted) run
        j = judge_relation(row["chunk_text"], row["subject"], row["relation"], row["object"])
        row["is_correct(Y/N)"] = "Y" if j.is_correct else "N"
        row["notes"] = j.notes
        print(f"  [{i}/{len(rows)}] {row['subject'][:20]!r} --{row['relation']}--> "
              f"{row['object'][:20]!r} -> {row['is_correct(Y/N)']}")
        _save(path, rows)
        time.sleep(CALL_DELAY_SECONDS)
    return rows


def print_spot_check_shortlist(rows: list[dict], label: str) -> None:
    incorrect = [r for r in rows if r["is_correct(Y/N)"] == "N"]
    correct = [r for r in rows if r["is_correct(Y/N)"] == "Y"]
    random.seed(7)
    n_sample = max(1, int(len(correct) * SPOT_CHECK_FRACTION)) if correct else 0
    sample_correct = random.sample(correct, min(n_sample, len(correct)))
    print(f"\n{label}: {len(incorrect)} marked incorrect (spot-check ALL of these), "
          f"+ {len(sample_correct)} random 'correct' ones to spot-check (of {len(correct)} total)")


def main() -> None:
    print("Reviewing entities with Gemini...")
    entity_rows = review_entities()
    print("\nReviewing relations with Gemini...")
    relation_rows = review_relations()

    print_spot_check_shortlist(entity_rows, "Entities")
    print_spot_check_shortlist(relation_rows, "Relations")
    print("\nBoth CSVs updated in place. Open them and focus your review on rows flagged above.")


if __name__ == "__main__":
    main()
