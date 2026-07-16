"""
Structure-aware chunking of parsed EUR-Lex documents.

Rather than fixed-size character/token windows, chunks follow the
document's own legislative structure:
  1. Primary unit: one Article or one Recital -- each is already a
     self-contained legal unit produced by parser.py.
  2. If a unit exceeds MAX_CHARS, it's split further at its own internal
     structure: numbered sub-paragraphs, which EU drafting uses in two
     forms depending on context -- bare ("1. text") for a paragraph, or
     parenthesized ("(1) text") for a list inside one, e.g. a definitions
     article. Splitting requires a clean 1,2,3.. sequence; a messy or
     absent sequence falls back to packing sentences up to MAX_CHARS
     instead of forcing a bad structural split (this affects mainly long
     recitals, which are prose with no numbered sub-points at all).
"""
import json
import re
from pathlib import Path

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
CHUNKS_DIR = Path(__file__).parents[2] / "data" / "chunks"

MAX_CHARS = 2000

# (?<=\s)|^ allows a match right at the start of the string, not just after
# whitespace -- an article's paragraph 1 almost always opens the body text
# with no preceding character at all (e.g. "1. Personal data shall be..."),
# which a plain (?<=\s) lookbehind can never match. Confirmed via unit tests
# to have silently broken subparagraph splitting for 76 of 87 long articles
# across the whole corpus before this fix -- they fell back to sentence-
# packing instead, still producing readable but less structurally clean chunks.
_SUBPARA_RE = re.compile(r"(?:(?<=\s)|^)(\d{1,2})\.\s+|(?:(?<=\s)|^)\((\d{1,2})\)\s+")
# Split before a capital letter or "(" following sentence-ending punctuation,
# to avoid breaking on abbreviations like "No. 45" or "p. 20" (digits/lowercase
# after the period are common in legal citations; a real sentence boundary is
# almost always followed by a capital or an opening parenthesis).
_SENTENCE_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")


def _subpara_number(m: re.Match) -> int:
    return int(m.group(1) or m.group(2))


def _split_by_subparagraphs(text: str) -> list[str]:
    matches = list(_SUBPARA_RE.finditer(text))
    accepted = []
    expected = 1
    for m in matches:
        if _subpara_number(m) == expected:
            accepted.append(m)
            expected += 1

    if len(accepted) < 2:
        return [text]

    parts = []
    for i, m in enumerate(accepted):
        start = m.start()
        end = accepted[i + 1].start() if i + 1 < len(accepted) else len(text)
        parts.append(text[start:end].strip())
    return parts


def _split_by_sentences(text: str, max_chars: int) -> list[str]:
    sentences = _SENTENCE_RE.split(text)
    chunks = []
    current = ""
    for sent in sentences:
        if current and len(current) + len(sent) + 1 > max_chars:
            chunks.append(current.strip())
            current = sent
        else:
            current = f"{current} {sent}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def _chunk_unit(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    parts = _split_by_subparagraphs(text)
    if len(parts) == 1:
        return _split_by_sentences(text, max_chars)

    final = []
    for part in parts:
        if len(part) > max_chars:
            final.extend(_split_by_sentences(part, max_chars))
        else:
            final.append(part)
    return final


def chunk_document(doc: dict) -> list[dict]:
    celex = doc["celex"]
    title = doc["title"]
    chunks = []

    for i, recital in enumerate(doc["recitals"], start=1):
        for j, part in enumerate(_chunk_unit(recital)):
            chunks.append({
                "chunk_id": f"{celex}_recital_{i}_{j}",
                "celex": celex,
                "doc_title": title,
                "chunk_type": "recital",
                "unit_number": i,
                "part": j,
                "heading": None,
                "text": part,
            })

    for article in doc["articles"]:
        for j, part in enumerate(_chunk_unit(article["text"])):
            chunks.append({
                "chunk_id": f"{celex}_article_{article['number']}_{j}",
                "celex": celex,
                "doc_title": title,
                "chunk_type": "article",
                "unit_number": article["number"],
                "part": j,
                "heading": article["heading"],
                "text": part,
            })

    return chunks


def chunk_all() -> list[dict]:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    for f in sorted(PROCESSED_DIR.glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        chunks = chunk_document(doc)
        out_path = CHUNKS_DIR / f"{doc['celex']}_chunks.json"
        out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [{doc['celex']}] {len(chunks)} chunks "
              f"(from {len(doc['recitals'])} recitals + {len(doc['articles'])} articles)")
        all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    print(f"Chunking documents from {PROCESSED_DIR}\n")
    chunks = chunk_all()
    print(f"\nDone. {len(chunks)} total chunks saved to {CHUNKS_DIR}")
