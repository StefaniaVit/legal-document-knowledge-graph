# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A portfolio project demonstrating document understanding, entity extraction, and knowledge
graph construction on EU legislation (EUR-Lex), targeting document-understanding / applied
science roles. Pipeline: PDF ingestion → structure-aware chunking → NER/relation extraction
→ Neo4j knowledge graph → GraphRAG query layer → MLflow-tracked evaluation.

Ingestion and entity/relation extraction run on a **local** LLM (llama.cpp + a quantized
GGUF model, no API key). The GraphRAG query layer uses **Gemini** (hosted, free tier) --
a deliberate exception, made because question-answering synthesis benefits from a stronger
model than the local one, and because query volume for interactive Q&A is low enough that
a free tier's rate limits are a non-issue (unlike the bulk extraction step, which needs
1000+ calls and stays local for exactly that reason).

**Current status (most recent milestone):** the full pipeline has now been through two
end-to-end rounds. Round 1 (Qwen2.5-3B): 44.1% entity / 20.0% relation precision on a
69-row hand-checked sample. This surfaced a specific, actionable problem
(`LEGAL_ACT`/`LEGAL_CONCEPT`/`ORGANIZATION` boundary confusion as the dominant failure mode),
which motivated adding a pytest suite (`tests/`, 58 tests -- which immediately found and fixed
a real chunker bug affecting 76/87 long articles, see Chunking below) and upgrading extraction
to **Qwen2.5-7B** (chosen over 14B specifically because this machine's 16GB unified memory is
a hard ceiling on Apple Silicon -- see Entity/relation extraction below for the full reasoning
and the two real bugs the upgrade surfaced along the way, one fixed and one mitigated with a
timeout safeguard rather than root-caused). The full corpus (1370 chunks after the chunker
fix) was re-extracted with 7B, Neo4j was reloaded from the new data, and the evaluation was
re-sampled and re-labeled from scratch (skipping Gemini pre-review entirely this round --
direct adjudication proved more reliable in round 1 anyway). **Round 2 result: 62.3% entity
precision (up from 44.1%), a real, meaningful improvement** consistent with 7B helping on
exactly the boundary-confusion problem round 1 identified. Relation precision on this small
round-2 sample (5 relations, entity 53) was 0%, but shouldn't be read as "relations got worse"
-- the sample is too small to be reliable, and all 5 failures were a qualitatively different,
more subtle pattern than round 1's (a real connection between two entities, described with a
relation type that's a mismatch from the fixed vocabulary, rather than round 1's outright
backwards-direction or fabricated relations). See Evaluation below for both rounds' full
methodology and findings. **Not yet done:** a README covering architecture and results.

## Commands

Virtual env is Python 3.9, at `.venv/`. Always invoke tools via the venv path rather than
assuming an activated shell:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/01_ingest.py
```

```bash
.venv/bin/python -m pytest tests/ -v
```

58 tests, no LLM calls or network access needed, run in well under a second. Covers pure
logic in `parser.py`, `chunker.py`, `neo4j_loader.py`, `entity_relation_extractor.py`, and
`graph_rag.py` -- deliberately excludes anything that calls a local or hosted LLM directly
(`_extract_recitals()`, `extract_entities()`/`extract_relations()`, Gemini calls). No linter
or build step configured.

### Ingestion pipeline

```bash
.venv/bin/python scripts/01_ingest.py
```

Checks `data/raw/{celex}/document.pdf` for each CELEX number in
`src/ingestion/eurlex_fetcher.py::SAMPLE_DOCUMENTS`, then parses every found PDF into
`data/processed/{celex}.json`. Exits with an error if no PDFs are present yet.

Running the parser alone (e.g. while iterating on `parser.py`) re-parses every PDF already
in `data/raw/`:

```bash
.venv/bin/python -m src.ingestion.parser
```

The recital-classification step in `parser.py` calls a local LLM once per candidate
recital marker (see Architecture below) — a full run across all documents takes several
minutes. Prefer running it in the background when iterating.

### Entity/relation extraction

```bash
.venv/bin/python -m src.ingestion.chunker            # if chunks aren't already built
.venv/bin/python scripts/02_extract_sample.py        # ~40-chunk sample, for checking quality
.venv/bin/python scripts/03_extract_all.py           # full corpus -- see timing note below
```

`03_extract_all.py` writes one JSON line per chunk to `data/extraction_all.jsonl` as it goes
(not one write at the end) and skips `chunk_id`s already present in that file on a re-run, so
it's safe to stop and resume rather than restarting a multi-hour job from scratch. It calls
through `TimeoutSafeExtractor` (`src/extraction/timeout_worker.py`), not
`entity_relation_extractor.extract()` directly -- see Entity/relation extraction below for why.

Full-corpus timing with Qwen2.5-7B (current model): ~28s/chunk average, so ~1370 chunks is a
~10-11 hour run, not the ~1.5-2 hours the earlier 3B model took -- budget accordingly, and
expect a small number of chunks (~1% in practice) to hit the timeout and get skipped; see
`scripts/07_retry_failed_with_3b.py` for how those were recovered with a fallback to 3B.

### Neo4j graph loading

```bash
.venv/bin/python -m src.graph.neo4j_loader
```

Requires `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` in `.env` (AuraDB
free tier). Reads `data/extraction_all.jsonl` plus `data/processed/` and `data/chunks/`, and
by default **wipes the database first** (`load_all(wipe_first=True)`) -- safe here since
everything is fully recreatable from files already on disk at zero cost, but worth knowing
before pointing this at a database you don't want cleared.

AuraDB free-tier instances auto-pause after a few days of inactivity, and the hostname can
briefly stop resolving (DNS `NXDOMAIN`) while paused -- if `verify_connectivity()` fails,
check the console at `console.neo4j.io` for a "Resume" option before assuming the instance
was deleted.

### GraphRAG queries

```bash
.venv/bin/python -m src.rag.graph_rag "What obligations does ENISA have?"
```

Requires `GEMINI_API_KEY` in `.env` (free tier, no credit card, from
`aistudio.google.com/apikey`).

### Frontend

```bash
.venv/bin/streamlit run app.py
```

Opens a search UI at `http://localhost:8501` wrapping `graph_rag.answer()` -- see
Architecture below.

### Evaluation

```bash
.venv/bin/python scripts/04_prepare_labeling_sample.py   # samples ~20 chunks -> data/labeled/*.csv
.venv/bin/python scripts/05_gemini_prereview.py           # optional: Gemini draft-fills judgments
.venv/bin/python -m src.evaluation.mlflow_eval            # computes precision, logs to MLflow
.venv/bin/mlflow ui                                       # view at http://127.0.0.1:5000
```

`scripts/06_finalize_labels.py` is a one-off script with the actual adjudicated verdicts
hardcoded in it (see Evaluation below for why) -- it's not a general-purpose tool to re-run,
it's a record of how the current `data/labeled/*.csv` files were finalized.

## Document acquisition (manual step)

EUR-Lex blocks automated HTTP downloads with AWS WAF (both the HTML and XML endpoints
return a JS challenge page, not the document). PDFs must be downloaded manually via a
browser and placed by hand:

```
https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:{celex}
→ data/raw/{celex}/document.pdf
```

`src/ingestion/eurlex_fetcher.py::check_downloads()` reports which of the documents listed
in `SAMPLE_DOCUMENTS` are present and prints the PDF URL for any missing ones.

## Architecture

### PDF parsing (`src/ingestion/parser.py`)

Extracts structured recitals/articles from EUR-Lex PDFs. Text extraction uses **PyMuPDF
(`fitz`)**, not `pypdf` — `pypdf` was tried first and mis-handles this PDF's font kerning,
silently injecting spurious spaces into words (`Ar ticle`, `ser ve`), which broke every
downstream regex. This is a known pypdf limitation, not a EUR-Lex-specific issue.

Document structure exploited (all standard EU legislative drafting conventions):
- Page footers (date / `L nnn/ppp` page ref / "Official Journal of the European Union" /
  "EN") appear in an unpredictable relative order across pages, so they're stripped as
  individual standalone lines rather than one fixed-order block regex.
- The preamble sits between the literal string `"Whereas:"` and
  `"HA(S|VE) ADOPTED THIS (REGULATION|DIRECTIVE|DECISION)"`.
- Enacting terms are headed by `Article N` alone on its own line. The last article's body
  is truncated at the `"Done at <city>,"` signature formula so trailing Annexes don't get
  absorbed into it.

**Recital vs. footnote-citation disambiguation is the trickiest part of this parser.**
Recitals are numbered `(1)`, `(2)`, ... sequentially, but EUR-Lex footnote markers use
identical `(N) text` formatting and their numbering resets on every page — a footnote can
coincidentally carry any number, including one that would otherwise look like a valid next
recital. This can't be resolved by position/formatting alone; it requires reading the text
(a citation is a bare bibliographic reference ending in `(OJ L ..., p. ...).`; a recital is
substantive legal reasoning — and a recital can still *open* by naming another act before
continuing with reasoning, which is the main source of remaining classifier error).

Resolution: every `(N)`-prefixed candidate is classified independently by a local LLM
(`src/ingestion/recital_classifier.py`) as RECITAL or CITATION, and **positional order —
not the printed number — determines final recital ordering**. This design choice matters:
an earlier version gated acceptance on the printed number matching a strict
monotonically-increasing sequence, which meant a single classifier false negative
permanently desynced the counter and silently truncated every recital after that point
(observed: GDPR dropped from 173 real recitals to 16 output). Decoupling classification
from sequencing means a wrong call only adds or drops one recital rather than truncating
the rest of the document. Measured false-negative rate is ~2.5% on GDPR's known-real
recitals; known residual failure mode is occasional recital splitting when an inline
footnote marker (e.g. `(2)` mid-sentence) gets misclassified as a fresh recital start.

**Embedded footnote blocks are a separate, distinct problem from the above** and are
handled by `_strip_embedded_footnote_blocks()`, applied to each recital's body text *after*
it's been isolated (never to the whole preamble -- that would see the genuine 1..173
recital sequence itself as one giant removable "run"). PyMuPDF extracts text page by page,
so a page's footnote block (rendered at the bottom of the page in the source PDF) ends up
concatenated between that page's content and the next page's, landing mid-paragraph or even
mid-sentence in whatever recital happens to span the break -- unlike the citation-collision
problem above, this recital is genuinely real and correctly identified; it just has garbage
spliced into the middle of it. Detection needs no LLM call: a single inline `(N)` reference
is normal prose, but a run of 2+ *consecutive* numbers close together isn't (recitals have
no numbered sub-structure of their own), so the pattern alone is a reliable signal. Once a
run is found, the removal boundary is the citation-ending pattern `(OJ ...).` if present
(tried first; a footnote entry can itself be very long -- a compound title with
cross-references to other acts -- so the search window matches the same ~600-char scale as
the inter-marker gap, not a short fixed window), falling back to a capitalized-sentence-start
heuristic for entries with no OJ reference (e.g. "not yet published in the Official
Journal)... 16 May 2022."); if neither is found, the run is left untouched rather than
guessing. Some documents render footnote numbers with internal spacing (`( 9 )`) instead of
compact (`(9)`) -- the marker regex matches either.

Known residual failure mode: a sentence that casually cites 2-3 consecutively-numbered
footnotes in a row (e.g. "...Regulation X (10) and Directive Y (11) and Directive Z (12)
also contribute...") produces the same "sequential markers close together" shape as a real
footnote-definition block, and gets wrongly swallowed along with the genuine block later in
the same recital -- truncating real prose rather than corrupting it silently. Confirmed
present in a handful of recitals in the Cybersecurity Act and NIS2 documents (not GDPR or
Data Governance Act). Not yet fixed: doing so would mean reusing `is_recital()`-style LLM
classification on individual run members to tell "these are citation definitions" from
"these are inline reference callouts," which was judged not worth the added complexity/LLM
calls for this stage relative to moving on to entity extraction -- revisit if downstream NER
quality turns out to be sensitive to it.

### Chunking (`src/ingestion/chunker.py`)

Structure-aware, not fixed-size: the primary chunk unit is one Article or one Recital
(already a self-contained legal unit from `parser.py`). Most are left as a single chunk
(avg 800-2500 chars). A unit is only split further when it exceeds `MAX_CHARS` (2000), and
even then the split follows the article's *own* internal numbering rather than an arbitrary
character count -- EU drafting uses two conventions for this depending on context: bare
(`"1. text"`) for a paragraph, or parenthesized (`"(1) text"`) for a list inside one, e.g. a
definitions article (confirmed by inspecting NIS2 Article 6). Both are matched. A clean
1,2,3.. sequence is required to accept a split; a messy or absent one falls back to packing
sentences up to `MAX_CHARS` (this mainly affects long recitals, which are prose with no
numbered sub-points at all). The sentence-packing fallback requires the next character after
a `.`/`;` to be capitalized or `(` before treating it as a boundary, to avoid breaking on
legal abbreviations like `No. 45` or `p. 20` (digit follows the period, not a capital).

**Fixed bug, found by writing `tests/test_chunker.py`:** `_SUBPARA_RE` used to require a
preceding whitespace character before a paragraph number (`(?<=\s)`), which can never match
at position 0 of a string -- and an article's paragraph 1 almost always opens the body text
with nothing before it at all (e.g. `"1. Personal data shall be..."`). This silently broke
subparagraph splitting for **76 of 87 long articles across the whole corpus**, falling back
to less-clean sentence-packing instead, with no visible symptom (the fallback still produces
readable chunks, so nothing looked obviously broken on manual inspection). Confirmed via a
direct scan of `data/processed/*.json` before fixing. Fix: `(?:(?<=\s)|^)` allows the match
at string start too. Re-running the chunker after the fix grew the corpus from 1051 to 1370
chunks, since the 76 affected articles now split into more, smaller, individually cleaner
chunks by their own paragraph structure instead of a few large sentence-packed ones.

Run with:

```bash
.venv/bin/python -m src.ingestion.chunker
```

Reads every file in `data/processed/`, writes `data/chunks/{celex}_chunks.json`.

### Local LLM inference (`src/ingestion/recital_classifier.py`)

Uses `llama-cpp-python` directly (not Ollama — evaluated and rejected: adds a server
process and a second package layer for no benefit here) against a GGUF model at
`models/qwen2.5-3b-instruct-q4_k_m.gguf` (not checked into git; ~2GB, downloaded from
`Qwen/Qwen2.5-3B-Instruct-GGUF` on Hugging Face). The `Llama` instance is a lazy-loaded
module-level singleton (`_get_llm()`) so the model is loaded once per process regardless of
how many classification calls are made.

**Still 3B, deliberately** -- only `entity_relation_extractor.py` was upgraded to 7B (see
below). Recital/citation classification is a much simpler binary judgment than entity/relation
extraction, 3B already tests reliably on it (see the false-negative-rate discussion below),
and this step runs during ingestion, before chunking -- re-verifying it against 7B wasn't
part of the extraction-quality investigation and hasn't been revisited.

Classification prompt is few-shot (see `_FEWSHOT` in that file) — zero-shot was tested
first and was unreliable (misclassified structurally-similar citations inconsistently).
The candidate-text snippet fed to the model is cut at the citation's natural end
(`(OJ ...).`) when present within the window, not at a fixed character count — a fixed-size
window was tested first and bled a short citation's tail into the next paragraph's opening
text, confusing the classifier.

**GPU note applying to every local-LLM module in this project** (`recital_classifier.py`,
`entity_relation_extractor.py`): `llama-cpp-python`'s `n_gpu_layers` defaults to `0` (CPU-only)
even when Metal is available. Both modules explicitly pass `n_gpu_layers=-1` to offload every
layer to the Mac's GPU -- roughly a 5x speedup over the silent CPU-only default. If adding a
new local-LLM module, carry this forward; it's easy to lose by copying an older, un-fixed
example instead of one of these two files.

### Entity/relation extraction (`src/extraction/entity_relation_extractor.py`)

Two separate LLM calls per chunk -- `extract_entities()` then `extract_relations()` -- not
one combined call. A single combined call was tried first and asked the local 3B model to
produce entities and relations together, which surfaced two failures at once: entities that
were entire run-on sentences instead of bounded spans, and relations naming subjects/objects
that were never listed as entities at all, directly violating the prompt's own instruction.
Splitting means relation extraction is grounded in an already-confirmed entity list (fed back
into the second call as plain text), and any relation still naming something outside that
list is dropped programmatically in `extract_relations()` as a hard backstop, regardless of
how well the model follows the instruction in a given case.

Output is grammar-constrained JSON (`LlamaGrammar.from_json_schema`), not free-text generation
parsed with `json.loads` afterward -- llama.cpp compiles the JSON schema into a GBNF grammar,
so the output is guaranteed schema-valid. This does *not* guarantee the model won't repeat
itself into a pathological loop at `temperature=0` (observed: the same 3-word phrase repeated
~40 times until hitting `max_tokens`, `finish_reason: "length"`) -- fixed with
`repeat_penalty=1.3`, not by changing the grammar.

`ENTITY_TYPES` = `ORGANIZATION, LEGAL_ACT, LEGAL_CONCEPT, DATE, PENALTY, OBLIGATION`.
`RELATION_TYPES` = `AMENDS, REPEALS, REFERENCES, DEFINES, IMPOSES_OBLIGATION_ON, ESTABLISHES,
APPLIES_TO, SUBJECT_TO, GRANTS_RIGHT_TO, RESPONSIBLE_FOR`. Both lists are also reused directly
by `neo4j_loader.py` as the whitelist for dynamic Cypher labels/relationship types (see below)
-- change them in one place only, here.

`_filter_valid_relations()` matches a relation's subject/object against the confirmed entity
list case/whitespace-insensitively, not by exact string equality -- entities and relations
come from two separate LLM calls, and the model isn't guaranteed to echo a name back with
identical formatting the second time; an exact-match filter would otherwise silently drop an
otherwise-valid relation over a trivial casing difference. (`graph_rag.py`'s
`_filter_known_names()` has the identical shape and reason, for candidate entity names
instead of relation subjects/objects -- see below.)

**Model: Qwen2.5-7B-Instruct-Q4_K_M, not 3B** (`models/qwen2.5-7b-instruct-q4_k_m-00001-of-
00002.gguf` -- a 2-shard GGUF; `llama-cpp-python` auto-detects and loads both shards from
just the first file's path, no manual merging needed). Chosen over 14B specifically because
this machine has **16GB unified memory** -- on Apple Silicon there's no separate VRAM, GPU
and CPU share one RAM pool, so that's a hard ceiling, not a soft preference. 14B's ~10GB
working set would leave too little headroom for a safe multi-hour unattended run; 7B's ~5-
5.5GB leaves real margin. Real GGUF sizes (not estimates): 3B ~2.0GB, 7B ~4.4GB (2 shards),
14B ~8.4GB (3 shards).

**The upgrade surfaced two real bugs, both found through deliberate testing against known
3B failure cases before committing to a full re-extraction -- worth reading in full before
touching this model or its prompt again:**

1. **Few-shot example leakage, 100% reproducible at `temperature=0`.** An earlier prompt
   version illustrated the "LEGAL_ACT isn't just numbered EU citations" rule with a worked
   example naming a real-sounding fabricated agency ("Senior Officials Group Mutual
   Recognition Agreement (MRA)"). Qwen2.5-7B was observed to regurgitate that *exact* phrase
   as a hallucinated entity when processing unrelated real text about treaties -- confirmed
   reproducible, not a fluke. First fix attempt: move the example out of prose embedded in
   the system prompt into a proper few-shot user/assistant exchange (matching
   `recital_classifier.py`'s established pattern), and rename the fabricated agency to
   something obviously fictional ("Zorvath Cooperation Framework Mutual Recognition Accord" /
   "Xylanti Standards Board") so a leak would be detectable rather than plausible. **This did
   not stop the leak** -- the model regurgitated the *fictional* name instead, on different
   thematically-similar real text, and the confused generation this produced also correlated
   with that single call taking ~600s instead of the usual 5-20s. The actual fix: remove the
   worked example entirely, keeping only the rule as plain prose with no illustration at all.
   Both symptoms (the hallucination and the ~600s slowdown) disappeared together when tested
   against the same two chunks that had triggered them -- strong evidence the worked example
   itself, regardless of content, was the trigger, not something fixable by further rewording
   it. Lesson: for this model, at least, a "for example" clause inside instructions is a real
   risk of being confused with actual input, not just a style question.

2. **Residual slow-call risk on dense multi-citation chunks, not fully root-caused.** Even
   after fixing (1), roughly 2.5% of chunks in testing (and ~1.3% of the full corpus, 18/1370)
   still triggered a similar multi-minute slowdown, this time with no obvious hallucination
   and no identified single trigger phrase. The clearest pattern across the 18 real failures:
   heavy overrepresentation of chunks containing several distinct, similarly-shaped entities
   in a row -- e.g. one recital listing five different regulation numbers in one sentence
   (`(EC) No 1060/2009, (EU) No 648/2012, ...`), or an article cross-referencing several
   different authority types and another Directive together. Plausible cause: grammar-
   constrained decoding search becoming harder to navigate with many similar, ambiguous
   candidates clustered together, but this is inference, not confirmed. **Mitigation, not a
   fix:** `src/extraction/timeout_worker.py`'s `TimeoutSafeExtractor` runs `extract()` in a
   persistent worker *subprocess* (not a thread -- a stuck llama.cpp call can't be cancelled
   from Python any other way, but a whole OS process can be killed regardless of what it's
   doing internally) and kills+restarts it if a call exceeds `DEFAULT_TIMEOUT_SECONDS` (60s),
   retrying once before giving up and recording the chunk as failed. `scripts/03_extract_all
   .py` uses this instead of calling `entity_relation_extractor.extract()` directly. The 18
   chunks this skipped in the full run were recovered with a one-off fallback to the 3B model
   (`scripts/07_retry_failed_with_3b.py`, hardcoded chunk_id list, not a general-purpose
   tool) -- meaning **1.3% of the final `extraction_all.jsonl` was processed by 3B, not 7B**,
   a small, documented inconsistency rather than a silent gap in the graph.

Known extraction-quality issues, confirmed by inspection (not yet fixed at the extraction
layer -- see `neo4j_loader.py`'s self-reference filter for the one that *is* handled
downstream): the model sometimes extracts a generic `"this Regulation"`/`"this Directive"`
self-reference instead of resolving which act is actually meant; sometimes crams a list into
one garbled entity string (`"'public authorities', 'CERTs', 'CSIRTs'..."`); sometimes tags the
same real entity with inconsistent types across mentions (see `neo4j_loader.py`); and can
mangle a relation's subject during a hard extraction (observed: NIS2's repeal of the old NIS
Directive was captured as `"repeal of Directive (EU)" --REPEALS--> "Directive (EU)
2016/1148"` -- directionally correct, but with an unclear, truncated subject name instead of
"NIS2 Directive"). This last one is the root cause of a GraphRAG limitation documented below.
These were characterized on the 3B model's output; not yet re-verified against 7B's.

### Neo4j graph loading (`src/graph/neo4j_loader.py`)

Schema: `(Document)-[:HAS_CHUNK]->(Chunk)-[:MENTIONS]->(Entity:<TYPE>)`, plus
`(Entity)-[:<RELATION_TYPE>]->(Entity)`. Entity/relation type names are interpolated directly
into Cypher as dynamic labels/relationship types (`f"MERGE (e:Entity:{etype} ...)"` --
Cypher has no parameter syntax for label/type names). This is safe specifically because both
come from the closed `ENTITY_TYPES`/`RELATION_TYPES` whitelist in `entity_relation_extractor.py`,
never from raw extraction output directly.

Entities are deduplicated by **normalized name alone**, not name+type. An earlier version
keyed on (name, type) together, which meant a real entity tagged with inconsistent types
across different chunks -- observed for "ENISA": `ORGANIZATION` in ~110 mentions, but
`LEGAL_ACT` and `LEGAL_CONCEPT` in a couple of others -- produced *separate* nodes per type,
fragmenting what should be one well-connected hub into several weaker ones (this was found by
the user manually double-clicking an "ENISA" node in the Neo4j Browser and discovering there
were three of them). Fix: `_compute_canonical_types()` does a majority vote across every
mention of a name in the whole corpus *before* loading, so the same key always gets the same
type label. This required a full wipe-and-reload (`load_all(wipe_first=True)`) since the old
fragmented nodes needed to disappear, not just sit alongside corrected ones -- cheap here since
everything reloads from `extraction_all.jsonl` with no need to re-run the LLM.

Self-referential entities (`"this Regulation"`, `"the Directive"`, etc., matched by
`_SELF_REFERENCE_RE`) are dropped entirely rather than loaded -- without resolving them to the
specific document they refer to, they'd merge across all 4 unrelated documents into one
meaningless shared node. **Known gap**: the regex only matches `(this|the) (regulation|
directive|act|decision)` -- a *bare* `"Regulation"` or `"Directive"` with no leading article
slips through and still collides across documents (visible in the graph as a `"Directive"`
LEGAL_ACT node with 44 connections, clearly an aggregate of many different directives' generic
self-references, not one real thing). Not yet fixed.

### GraphRAG query layer (`src/rag/graph_rag.py`)

Three-stage pipeline per question: (1) resolve which known graph entities the question
relates to, (2) retrieve those entities' neighborhoods (relations + source chunk text) via
**fixed** Cypher, (3) ask an LLM to synthesize an answer from that retrieved context. All
three stages use Gemini except the retrieval Cypher itself, which is deliberately not
LLM-generated ("text-to-Cypher") -- a model asked to write Cypher freeform risks syntactically
invalid or semantically wrong queries with no easy way to catch the mistake at run time; a
fixed template parameterized only by a resolved entity key can't be wrong in that way.

**Entity resolution is grounded selection, not blind-candidate-then-fuzzy-match.**
`_select_relevant_entities()` gives the model the full list of ~1200 known entity names
directly (fits comfortably in Gemini's context window) and asks it to pick which ones are
relevant, rather than asking it to invent candidate strings and then fuzzy-matching those
against the graph via Cypher `CONTAINS`. The blind-then-fuzzy version was implemented first
and failed two different ways on real test questions:
- *"What penalties can be imposed for violating GDPR?"* → candidate `"Penalties"` never
  matched the graph's `"administrative fines"` entity at all -- no lexical overlap between the
  words, despite being obviously the same concept to any reader.
- *"What obligations does ENISA have?"* → candidate `"obligation"` (generic, drawn from the
  question's own wording) matched dozens of unrelated long `OBLIGATION`-type entities purely
  because they happened to contain that substring.

Both failures share one root cause: `CONTAINS` matching can only find lexical overlap, not
semantic relationships -- exactly the gap closed by having the model choose from the real
options instead of generating candidates blind and pattern-matching against them afterward.
Every name the grounded version returns is guaranteed to exist (it's chosen from ground
truth), so no fuzzy-match noise-filtering logic is needed downstream -- `_filter_known_names()`
still matches case/whitespace-insensitively rather than by exact equality, though (same
reasoning as `entity_relation_extractor.py`'s `_filter_valid_relations()`: Gemini echoing a
name back isn't guaranteed to preserve exact casing), and returns the *known* name's own
casing, not the candidate's, since the downstream Cypher lookup needs to match what's
actually stored in the graph.

**Entity selection uses `temperature=0`.** Found via manual testing through the Streamlit
frontend: asking the identical question twice in a row returned 22 matched entities one time
and 1 the next, at the default sampling temperature. "Select every relevant name from this
list" is an exhaustive-selection task, not open-ended generation, so this call should behave
close to deterministically for a fixed question -- `_synthesize_answer()` is left at the
default temperature, since some variation in how the final answer is phrased is fine.

**Chunk text shown in retrieved context is truncated at the nearest sentence boundary, not a
blind character cut.** `_truncate_at_sentence()` backs up to the last `". "`/`"; "` before the
600-char limit (falls back to a hard cut + `"..."` only if no boundary exists in that span).
Found the same way -- a blind `text[:600]` slice was cutting mid-sentence
(`"...in accordance with Regulation (EU"`), visibly losing whatever clause came next, both in
what's shown in the UI and in what actually gets sent to the synthesis prompt.

**Known limitation, distinct from the above and not fixed by it: retrieval quality is capped
by upstream extraction quality.** Asking *"What does the NIS2 Directive repeal?"* correctly
resolves to the right entity (`"Directive (EU) 2016/1148"`, the actual old NIS Directive) and
Gemini declines to answer rather than hallucinate -- but the real fact is genuinely hard to
read off from what was retrieved: the `REPEALS` relation exists (`"repeal of Directive (EU)"
--REPEALS--> "Directive (EU) 2016/1148"`) but its subject is a mangled entity name instead of
"NIS2 Directive" (see the extraction-quality issue noted above), and the 5 sampled chunks
mentioning this entity all happened to come from the *Cybersecurity Act* (which also cites the
old NIS Directive) rather than from NIS2's own text, so no retrieved chunk states the repeal
in plain language either. Declining to guess here is arguably correct, cautious behavior given
what the model was actually shown -- but the underlying fix isn't anything in this file; it's
upstream, in `entity_relation_extractor.py`'s relation-subject extraction and/or increasing
`MAX_CHUNKS_PER_ENTITY` / prioritizing chunks from the entity's own source document rather than
wherever it's merely referenced. Not yet fixed -- documented here since it was found through
deliberate testing, not theoretical.

### Frontend (`app.py`)

Streamlit UI wrapping `graph_rag.answer()` -- a search box in, and the synthesized answer plus
the resolved entities and raw retrieved context out. Deliberately surfaces the entities/context,
not just the final answer: that transparency (which real graph entities and source chunks
grounded this answer) is the actual differentiator of graph-based retrieval over a plain
chatbot, so hiding it in the UI would throw away the point of the project. No caching/session
persistence beyond an in-memory `st.session_state` history for the current browser session --
each question still costs one call to Gemini's free tier (20 requests/day, shared with
`graph_rag.py`'s own quota; see Evaluation below for how that limit was discovered).

### Evaluation (`src/evaluation/mlflow_eval.py`)

Two rounds now exist in MLflow (`mlruns/`, experiment `legal-kg-extraction-eval`) -- both
precision-only, no recall measurement (that would require independently finding every
entity/relation actually present in the sampled chunks and comparing, which wasn't done).
`data/labeled/*.csv` currently holds round 2's sample; round 1's is preserved in
`data/labeled_3b_baseline/` for comparison.

**Labeling methodology (applies to both rounds) -- read this before trusting or re-deriving
either round's numbers.** The ground truth is AI-assisted review, not independent human
labeling. `scripts/04_prepare_labeling_sample.py` samples ~20 chunks from the already-extracted
`data/extraction_all.jsonl` (no LLM calls) into `data/labeled/*.csv`, leaving `is_correct`
columns blank; every row then gets a verdict via one of two paths described below.

**Round 1 (Qwen2.5-3B, pre-chunker-fix): 44.1% entity precision, 20.0% relation precision**,
on 59 entities + 10 relations from 20 chunks:
1. `scripts/05_gemini_prereview.py` was meant to draft-fill all 69 rows via Gemini as a fast
   first pass, but only got through 32 of 59 entity rows (0 of 10 relations) before repeatedly
   hitting a **20-requests/day** free-tier quota -- confirmed on two different model variants
   (`gemini-2.5-flash` and `gemini-2.5-flash-lite` each have their own separate 20/day cap on
   this project; switching models did not route around it, it just moved to a second, equally
   exhausted bucket). This is a much stricter limit than the ~1,500/day figure commonly quoted
   for Gemini free tiers elsewhere -- don't assume that number applies to a fresh project/key
   without checking.
2. Of the 32 rows Gemini did judge, several were unreliable on inspection: some were
   **self-contradictory** (marked an entity "incorrect" while recommending the *identical* type
   back as its own correction -- e.g. `'Commission'` ORGANIZATION -> "corrected" to ORGANIZATION),
   and others contradicted this project's *own* established conventions (Gemini downgraded
   `'supervisory authority'` from ORGANIZATION to LEGAL_CONCEPT, despite the graph's own
   hub-entity analysis, done earlier in this project, having already confirmed it denotes a
   real institutional body).
3. Given that, every one of the 69 rows was re-adjudicated directly (by Claude, with full
   project context) rather than trusting Gemini's judgments as-is. `scripts/06_finalize_labels
   .py` is the record of that adjudication -- every verdict hardcoded with a one-line
   justification, keyed by `(chunk_id, entity_text, entity_type)` / `(chunk_id, subject,
   relation, object)`.

Findings: entity precision by type highly uneven -- `ORGANIZATION` 87% (15 sampled) vs.
`LEGAL_ACT` 25% (20 sampled) vs. `PENALTY` 0% (3 sampled, small-sample caveat). Dominant
failure mode: `LEGAL_ACT`/`LEGAL_CONCEPT`/`ORGANIZATION` boundary confusion (mechanisms and
frameworks like "binding corporate rules" mistagged as specific enacted laws). Secondary:
span garbling, truncated citation fragments, generic self-references. Relations scored lower
mostly mechanically -- a relation needs both endpoints correct, and 0.44 x 0.44 ≈ 19% is close
to the measured 20% -- plus genuine direction/type confusion (e.g. a relation stated
backwards) and occasional fabrication.

**Round 2 (Qwen2.5-7B, post-chunker-fix): 62.3% entity precision (+18.2pp over round 1),
0.0% relation precision (n=5, too small to be reliable)**, on 53 entities + 5 relations from
a fresh 20-chunk sample (different `chunk_id`s than round 1 -- the chunker fix changed
boundaries). This round skipped Gemini pre-review entirely and went straight to direct
Claude adjudication (`scripts/08_finalize_labels_7b.py`), since round 1 showed Gemini's
judgments needed heavy correction anyway. A few ambiguous cases were checked against the
*full* chunk text (not just the CSV's truncated preview) before judging, to rule out
hallucination -- e.g. `'European Data Protection Board'` looked suspicious from the preview
alone but was confirmed genuinely present in the full text.

Findings: entity precision by type -- `ORGANIZATION` 76% (25 sampled), `LEGAL_CONCEPT` 53%
(15 sampled), `OBLIGATION` 50% (2 sampled, tiny), `LEGAL_ACT` 45% (11 sampled). `LEGAL_ACT`
improved substantially over round 1 (25% -> 45%) -- consistent with 7B helping on exactly the
boundary-confusion problem round 1 identified, though still the weakest well-sampled category.
**The relation failures this round are qualitatively different from round 1's**, not just a
lower number: all 5 involved a real, genuine connection between two correctly-identified
entities, just described with a relation type that's a mismatch from the fixed
`RELATION_TYPES` vocabulary rather than round 1's outright backwards-direction or fabricated
relations -- e.g. `'Commission' --IMPOSES_OBLIGATION_ON--> 'implementing acts'` when the text
actually says the Commission *adopts* implementing acts (closer to `ESTABLISHES`, still not a
great fit). This suggests the model is now correctly spotting *that* two things are related
more often, but the fixed relation vocabulary sometimes lacks a good option for *how*.

**Report either round as "AI-assisted review (Gemini + Claude, with project context)," not
"human-validated ground truth," if described elsewhere (e.g. a README)** -- that's a
materially weaker, if still useful, claim.

MLflow logs per run (local file store, `mlruns/`, experiment name `legal-kg-extraction-eval`):
overall and per-type precision metrics, `extraction_model`/`review_method` params, and the
labeled CSVs plus a generated markdown summary as artifacts -- `mlflow ui` shows both rounds
side by side for comparison.

### Data flow

```
data/raw/{celex}/document.pdf       -- manually downloaded EUR-Lex PDF
data/raw/{celex}/meta.json          -- title/URL sidecar, auto-generated
data/processed/{celex}.json         -- {celex, title, date, recitals[], articles[], full_text}
data/chunks/{celex}_chunks.json     -- flat list of chunk dicts, see chunker.py
data/extraction_all.jsonl           -- one line per chunk: {chunk_id, entities[], relations[]}.
                                        Current version: 7B, 1370 chunks, 18 of which (1.3%)
                                        fell back to 3B after timing out -- see Entity/relation
                                        extraction above.
data/extraction_all_3b_baseline.jsonl.bak -- the prior full run (3B, pre-chunker-fix, 1051
                                        chunks) -- kept as a historical comparison point, not
                                        read by any script.
data/extraction_sample.json         -- same shape, ~40-chunk quality-check sample (JSON array,
                                        not JSONL -- written once at the end, not incrementally,
                                        since it's small). Overwritten each time
                                        `scripts/02_extract_sample.py` runs -- current contents
                                        reflect whichever model was configured at the time, not
                                        necessarily 7B.
data/labeled/entities_review.csv    -- hand-checked sample: {chunk_id, chunk_text, entity_text,
                                        entity_type, is_correct(Y/N), corrected_type, notes}
data/labeled/relations_review.csv   -- same idea for relations: {chunk_id, chunk_text, subject,
                                        relation, object, is_correct(Y/N), notes}
Neo4j (AuraDB)                      -- the loaded graph; see neo4j_loader.py schema above
mlruns/                             -- local MLflow tracking store; see Evaluation above
```

`articles[]` entries are `{number, heading, text}`. `recitals[]` entries are plain strings
(no explicit number stored, since the printed number is unreliable — see above). Chunk dicts
are `{chunk_id, celex, doc_title, chunk_type, unit_number, part, heading, text}` --
`chunk_id` is `{celex}_{recital|article}_{unit_number}_{part}`; `part` is 0 for units that
weren't split further. Entity dicts are `{text, type}`; relation dicts are `{subject,
relation, object}` where `subject`/`object` must match an entity's `text` from the same chunk.

### Not yet implemented / not yet done

Every planned `src/` module has working code (ingestion, extraction, graph, rag, evaluation —
see Architecture above), a pytest suite (`tests/`) covers the pure-function logic in most of
them, Neo4j reflects the current 7B/1370-chunk data, and the evaluation has a real two-round
before/after comparison (see Evaluation above). What's left: a README covering architecture
decisions and results, not yet written. Possible future directions if this project continues:
tightening the `LEGAL_ACT` boundary-confusion prompt further now that 7B's specific remaining
error pattern is characterized, giving the fixed `RELATION_TYPES` vocabulary better options
for cases like "adopts"/"notifies" that round 2 showed don't fit any current type well, or
resolving the still-open `neo4j_loader.py` self-reference gap (bare `"Regulation"`/`"Directive"`
with no leading article).
