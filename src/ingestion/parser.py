"""
Parses EUR-Lex PDFs into structured document objects.

Uses PyMuPDF (fitz) for text extraction — pypdf was tested first but
mis-handles this PDF's font kerning, injecting spurious spaces into words
(e.g. "Ar ticle", "ser ve") which broke every downstream regex.

EUR-Lex PDF layout:
  - Each page ends with a running footer containing four components
    (date, "L nnn/ppp" page ref, "Official Journal of the European Union",
    "EN") whose relative order varies unpredictably between pages/documents,
    so they're stripped as individual standalone lines rather than one
    fixed-order block.
  - Preamble: "Whereas:" then recitals numbered (1), (2), ... sequentially,
    ending at "HAVE ADOPTED THIS REGULATION/DIRECTIVE/DECISION:"
  - Footnote markers also look like "(1)", "(2)" but the numbering resets
    on every page, so recitals are identified by requiring the recital
    number sequence to be strictly increasing by 1. A footnote can still
    coincidentally carry the exact number expected next -- that collision
    is resolved by asking a local LLM (see recital_classifier.py) whether
    the candidate text reads as a citation or genuine recital.
  - Enacting terms: "Article N" on its own line, followed by a title line,
    then numbered paragraphs. The last article's body is truncated at the
    "Done at <city>," signature formula so trailing Annexes don't get
    absorbed into it.

Output JSON saved to data/processed/{celex}.json
"""
import json
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from .recital_classifier import is_recital

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

_FOOTER_LINE_RES = [
    re.compile(r"^EN$"),
    re.compile(r"^Official Journal of the European Union$", re.IGNORECASE),
    re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$"),
    re.compile(r"^L\s*\d+/\d+$"),
]

_TITLE_ANCHOR_RE = re.compile(r"^(REGULATION|DIRECTIVE|DECISION)\s*\(E[UC]\).*$", re.MULTILINE)
_ARTICLE_RE = re.compile(r"^Article\s+(\d+)\s*$", re.MULTILINE)
_RECITAL_MARKER_RE = re.compile(r"(?<=\s)\((\d+)\)\s+")
_ADOPTED_RE = re.compile(r"HA(S|VE) ADOPTED THIS (REGULATION|DIRECTIVE|DECISION)", re.IGNORECASE)
_SIGNATURE_RE = re.compile(r"Done at \w+,")

# A footnote block sitting at the bottom of a PDF page lands, after page
# text is concatenated, in between the two halves of whatever recital
# happens to span that page break. A single inline "(N)" reference is
# normal prose; a run of 2+ consecutive numbers close together is not
# (recitals don't have their own numbered sub-structure), so that pattern
# alone is enough to flag it without needing LLM classification. Some
# documents render the footnote number with internal spacing ("( 9 )")
# rather than compact ("(9)") -- both are matched.
_FOOTNOTE_MARKER_RE = re.compile(r"(?<=\s)\(\s*(\d{1,2})\s*\)\s+")
_FOOTNOTE_BLOCK_MAX_GAP = 600
# A footnote block can interrupt a sentence mid-way (e.g. "Personal or
# (1)...(2)... household activities...") rather than always landing at a
# sentence boundary, so the resumption point can start lowercase. The OJ
# citation ending is a far more reliable signal of where a footnote entry
# actually ends and is tried first; the capitalized-sentence-start check
# is only a fallback for entries that don't cite an OJ reference (e.g.
# "not yet published in the Official Journal ... 16 May 2022.").
_OJ_END_RE = re.compile(r"\(OJ [^)]*\)\.")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")


def _is_footer_line(line: str) -> bool:
    stripped = line.strip()
    return any(r.match(stripped) for r in _FOOTER_LINE_RES)


def _extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    text = "\n".join(pages)
    lines = [l for l in text.splitlines() if not _is_footer_line(l)]
    return "\n".join(lines)


def _extract_title_and_date(text: str) -> tuple[str, str]:
    """Title/date sit right after the act identifier line, e.g.:
    'REGULATION (EU) 2016/679 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL'
    'of 27 April 2016'
    'on the protection of natural persons ... (General Data Protection Regulation)'
    """
    m = _TITLE_ANCHOR_RE.search(text)
    if not m:
        return "", ""

    lines = text[m.start():m.start() + 2000].splitlines()
    header_line = lines[0].strip()
    date = ""
    subtitle_lines = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("of ") and not date:
            dm = re.search(r"\d{1,2}\s+\w+\s+\d{4}", line)
            if dm:
                date = dm.group()
                continue
        if line.startswith("(Text with") or (line.isupper() and len(line) > 15):
            break
        subtitle_lines.append(line)

    title = " ".join([header_line] + subtitle_lines)
    return title, date


def _strip_embedded_footnote_blocks(text: str) -> str:
    """Removes bursts of 2+ sequentially-numbered "(N)" footnote entries
    embedded mid-text. PyMuPDF extracts page by page, so a page's footnote
    block (rendered at the bottom of the page in the source PDF) ends up
    concatenated between that page's body text and the next page's --
    landing mid-sentence in whatever recital happens to span the break.
    A single inline "(N)" reference is normal prose; a run of 2+
    consecutive numbers close together is not (recitals don't have their
    own numbered sub-structure), so that pattern alone identifies a
    footnote block without needing LLM classification. Only the span from
    the first marker in a run through the end of the last entry is
    removed; if that end can't be found nearby, the run is left alone
    rather than guessing.
    """
    matches = list(_FOOTNOTE_MARKER_RE.finditer(text))

    runs = []
    i = 0
    while i < len(matches):
        run = [matches[i]]
        j = i + 1
        while (j < len(matches)
               and int(matches[j].group(1)) == int(run[-1].group(1)) + 1
               and matches[j].start() - run[-1].end() <= _FOOTNOTE_BLOCK_MAX_GAP):
            run.append(matches[j])
            j += 1
        if len(run) >= 2:
            runs.append(run)
        i = j

    if not runs:
        return text

    removals = []
    for run in runs:
        # A single citation entry can itself run long (a compound title
        # with cross-references to other acts) -- use the same scale as
        # the inter-marker gap threshold rather than a short fixed window.
        tail = text[run[-1].end():run[-1].end() + _FOOTNOTE_BLOCK_MAX_GAP]
        end_m = _OJ_END_RE.search(tail) or _SENTENCE_BOUNDARY_RE.search(tail)
        if end_m:
            removals.append((run[0].start(), run[-1].end() + end_m.end()))

    if not removals:
        return text

    cleaned = []
    prev_end = 0
    for start, end in removals:
        cleaned.append(text[prev_end:start])
        prev_end = end
    cleaned.append(text[prev_end:])
    return re.sub(r"\s+", " ", "".join(cleaned)).strip()


def _extract_recitals(text: str) -> list[str]:
    """Recitals sit between 'Whereas:' and 'HAVE ADOPTED THIS ...'.

    The preamble is whitespace-normalized first (all runs of whitespace,
    including newlines, collapsed to a single space) so marker detection
    doesn't depend on where PDF text extraction happened to wrap a line.

    Footnote markers use identical "(N) text" formatting and their
    numbering resets on every page, so they can coincidentally carry any
    number. A number-sequence filter alone is fragile to classifier error:
    requiring an exact match to advance means a single misclassified real
    recital permanently desyncs the count and silently drops everything
    after it. Instead, every candidate is classified independently and
    positional order (not the printed number) determines the final
    ordering -- a wrong call here only adds or drops one recital rather
    than truncating the rest of the document.
    """
    start_m = re.search(r"Whereas:", text)
    end_m = _ADOPTED_RE.search(text)
    if not start_m or not end_m or end_m.start() <= start_m.end():
        return []

    preamble = re.sub(r"\s+", " ", text[start_m.end():end_m.start()])
    raw_matches = list(_RECITAL_MARKER_RE.finditer(preamble))

    recital_matches = []
    for i, m in enumerate(raw_matches):
        window_end = raw_matches[i + 1].start() if i + 1 < len(raw_matches) else len(preamble)
        candidate_text = preamble[m.end():min(window_end, m.end() + 300)]
        if is_recital(candidate_text):
            recital_matches.append(m)

    recitals = []
    for i, m in enumerate(recital_matches):
        start = m.end()
        end = recital_matches[i + 1].start() if i + 1 < len(recital_matches) else len(preamble)
        body = preamble[start:end].strip()
        recitals.append(_strip_embedded_footnote_blocks(body))
    return recitals


def _extract_articles(text: str) -> list[dict]:
    """Extracts enacting-terms articles. Truncates at the 'Done at <city>,'
    signature formula so the last article doesn't absorb trailing Annexes.
    """
    sig_m = _SIGNATURE_RE.search(text)
    if sig_m:
        text = text[:sig_m.start()]

    matches = list(_ARTICLE_RE.finditer(text))
    articles = []
    for i, m in enumerate(matches):
        number = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        lines = [l.strip() for l in body.splitlines() if l.strip()]
        heading = lines[0] if lines else ""
        body_text = re.sub(r"\s+", " ", " ".join(lines[1:]))

        articles.append({"number": number, "heading": heading, "text": body_text})
    return articles


def parse_document(celex: str) -> Optional[dict]:
    pdf_path = RAW_DIR / celex / "document.pdf"
    if not pdf_path.exists():
        print(f"  [{celex}] No PDF found at {pdf_path}")
        return None

    print(f"  [{celex}] Parsing PDF ({pdf_path.stat().st_size // 1024} KB)...")
    text = _extract_pdf_text(pdf_path)

    title, date = _extract_title_and_date(text)
    recitals = _extract_recitals(text)
    articles = _extract_articles(text)

    meta_path = RAW_DIR / celex / "meta.json"
    if not title and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        title = meta.get("title", "")

    parsed = {
        "celex": celex,
        "title": title,
        "date": date,
        "recitals": recitals,
        "articles": articles,
        "full_text": text,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{celex}.json"
    out_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"    -> '{title[:60]}' | {len(articles)} articles | {len(recitals)} recitals")
    return parsed


def parse_all() -> list[dict]:
    celex_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    print(f"Parsing {len(celex_dirs)} document(s) from {RAW_DIR}\n")
    results = []
    for doc_dir in celex_dirs:
        result = parse_document(doc_dir.name)
        if result:
            results.append(result)
    return results


if __name__ == "__main__":
    parse_all()
