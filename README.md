# Legal Document Knowledge Graph

A knowledge graph built from real EU legislation (GDPR, the Cybersecurity Act, NIS2, the Data
Governance Act), with a natural-language search UI on top. Built as a portfolio project
demonstrating document understanding, information extraction, and knowledge graph
construction/retrieval on messy, real-world legal PDFs — end to end, from raw scanned-style
government PDFs to a working GraphRAG question-answering interface.

```
PDF ingestion → structure-aware chunking → entity/relation extraction (local LLM)
    → Neo4j knowledge graph → GraphRAG query layer → Streamlit search UI
                                                    ↳ MLflow-tracked evaluation
```

## Why this project

Most "RAG demo" projects retrieve by vector similarity over arbitrary chunks. This one instead:
- Builds an actual **knowledge graph** (entities + typed relations), not just embeddings.
- Retrieves by resolving a question to *real graph entities* and walking their neighborhood
  via fixed Cypher — not by asking an LLM to write Cypher freeform, and not by fuzzy string
  matching, both of which were tried and failed for reasons documented below.
- Runs the expensive part (bulk entity/relation extraction across ~1,370 chunks) entirely on a
  **local, open-weight LLM** (Qwen2.5-7B via `llama.cpp`) — no API key, no per-token cost, no
  rate limit. A hosted model (Gemini, free tier) is used only for the low-volume interactive
  query layer, a deliberate and explicit exception, not an inconsistency.
- Is evaluated honestly: precision is measured against an **AI-assisted labeled sample**, and
  the README and code both say so explicitly rather than implying "human-validated ground
  truth." Two full extraction rounds (3B → 7B) are compared before/after, tracked in MLflow.

## Demo

```bash
source .venv/bin/activate
streamlit run app.py
```

Ask a question (e.g. *"What obligations does ENISA have?"*) and the UI shows, in order:
1. The synthesized answer.
2. Every graph entity the question resolved to.
3. The raw relations and source-document chunks that were actually handed to the model to
   produce that answer — the retrieval is inspectable, not a black box.

## Results

Upgrading extraction from Qwen2.5-3B to Qwen2.5-7B raised entity precision from **44% to 62%**
on a hand-checked sample, concentrated exactly where the 3B failure analysis predicted it would:
mistagging mechanisms/frameworks as specific enacted laws (`LEGAL_ACT` type confusion). Relation
precision was also measured but the samples (10 and 5 relations) are too small to draw a
before/after conclusion from.

This is evaluated against an **AI-assisted labeled sample, not independent human-validated
ground truth** — see [Evaluation methodology](#evaluation-methodology). Full per-type
breakdowns and failure-mode analysis for both rounds are tracked in MLflow (`mlruns/`,
experiment `legal-kg-extraction-eval`) and in [`CLAUDE.md`](CLAUDE.md).

## Architecture

**Ingestion** ([`src/ingestion/parser.py`](src/ingestion/parser.py)) — EUR-Lex PDFs are parsed
with PyMuPDF (not `pypdf`, which corrupts kerning in this PDF's font) into structured
recitals/articles. The hardest part: EU legislation numbers recitals `(1), (2), ...` but
footnote citations reuse the exact same `(N)` format with per-page-reset numbering — resolved
by classifying every candidate with a local LLM rather than trusting position/formatting alone.

**Chunking** ([`src/ingestion/chunker.py`](src/ingestion/chunker.py)) — structure-aware, not
fixed-size: one Article or Recital per chunk, split further only when it exceeds a length
threshold, and even then along the article's own internal paragraph numbering rather than an
arbitrary character count.

**Entity/relation extraction**
([`src/extraction/entity_relation_extractor.py`](src/extraction/entity_relation_extractor.py))
— two separate local-LLM calls per chunk (entities, then relations grounded in the confirmed
entity list), output is grammar-constrained JSON (`LlamaGrammar.from_json_schema`), guaranteeing
schema-valid extraction. Runs on Qwen2.5-7B-Instruct (GGUF, 4-bit quantized), chosen over 14B
specifically because of a 16GB unified-memory ceiling on Apple Silicon. A subprocess-based
timeout safeguard (`src/extraction/timeout_worker.py`) handles the ~1% of chunks that trigger
pathological multi-minute slowdowns during decoding.

**Knowledge graph** ([`src/graph/neo4j_loader.py`](src/graph/neo4j_loader.py)) — Neo4j AuraDB
(free tier). Schema: `(Document)-[:HAS_CHUNK]->(Chunk)-[:MENTIONS]->(Entity:<TYPE>)`, plus
`(Entity)-[:<RELATION_TYPE>]->(Entity)`. Entities are deduplicated by normalized name with a
majority-vote canonical type across every mention, to avoid fragmenting one real-world entity
(e.g. "ENISA") into several graph nodes over inconsistent type-tagging.

**GraphRAG query layer** ([`src/rag/graph_rag.py`](src/rag/graph_rag.py)) — three stages per
question: (1) an LLM resolves which of the graph's ~1,200 *real* entity names the question
relates to (grounded selection from ground truth, not blind-candidate-then-fuzzy-match — see
code docstring for two concrete real-question failures that ruled out the fuzzy-match
approach), (2) fixed Cypher retrieves each entity's relations and source chunks — never
LLM-generated Cypher, so it can't be syntactically or semantically wrong, (3) an LLM
synthesizes an answer from only that retrieved context, explicitly declining rather than
guessing when the context doesn't support an answer.

**Frontend** ([`app.py`](app.py)) — a Streamlit search UI wrapping the query layer, showing the
resolved entities and retrieved context alongside the answer rather than hiding them.

**Evaluation** ([`src/evaluation/mlflow_eval.py`](src/evaluation/mlflow_eval.py)) — precision
computed against a hand-checked sample, logged to MLflow with per-type breakdowns and a
generated summary report.

## Evaluation methodology

The labeled ground truth in `data/labeled/` is **AI-assisted review, not independent
human-validated ground truth**. Each row was judged either by Gemini (free tier) as a first
pass, or — for every relation and every row Gemini's free-tier quota didn't reach — by direct
adjudication with full project context. Several of Gemini's own judgments were found to be
self-contradictory or inconsistent with this project's established conventions on manual
inspection, which is why direct adjudication was preferred over trusting Gemini's output as-is.
This is disclosed here, in the MLflow run artifacts, and in the evaluation module's own
docstring — treat the precision numbers above as directionally meaningful, not as a rigorously
independent benchmark.

## Setup

```bash
git clone https://github.com/StefaniaVit/legal-document-knowledge-graph.git
cd legal-document-knowledge-graph
python3.11 -m venv .venv          # or any Python 3.9+
source .venv/bin/activate
pip install -r requirements.txt
```

### Quick demo (just run the app — no PDF downloads, no local model, no extraction run)

The extracted data (`data/processed/`, `data/chunks/`, `data/extraction_all.jsonl`) is already
committed to this repo. To run the search UI against it, you only need:

**1. A free Neo4j AuraDB instance** — create one at `console.neo4j.io`, then put its
credentials in a `.env` file at the project root:

```
NEO4J_URI=...
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
NEO4J_DATABASE=...
```

**2. A free Gemini API key** — from `aistudio.google.com/apikey` (no credit card required),
added to the same `.env` file:

```
GEMINI_API_KEY=...
```

**3. Load the graph and run the app:**

```bash
python -m src.graph.neo4j_loader     # loads the committed extraction data into your Neo4j instance
streamlit run app.py                 # opens the search UI at localhost:8501
```

That's the whole "someone else can run this" path — no local LLM, no GPU, no multi-hour job.

### Full reproduction (redo extraction from the raw PDFs yourself)

Only needed if you want to re-run ingestion/extraction itself, e.g. to verify the pipeline or
try a different model:

**1. Download source documents manually.** EUR-Lex blocks automated downloads (AWS WAF), so
PDFs must be fetched by hand from
`https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:{celex}` and placed at
`data/raw/{celex}/document.pdf`. See `src/ingestion/eurlex_fetcher.py::SAMPLE_DOCUMENTS` for
the CELEX numbers used, and `check_downloads()` to verify what's present.

**2. Download the local LLM.** Qwen2.5-7B-Instruct-GGUF (Q4_K_M quantization) from Hugging
Face, placed at `models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` (+ its second shard).
Not checked into git (~4.4GB). A 16GB+ unified/VRAM memory machine is recommended — see
[`CLAUDE.md`](CLAUDE.md) for why.

**3. Run the pipeline:**

```bash
python scripts/01_ingest.py               # parse PDFs -> data/processed/
python -m src.ingestion.chunker           # -> data/chunks/
python scripts/03_extract_all.py          # entity/relation extraction, several hours
python -m src.graph.neo4j_loader          # load into Neo4j (needs .env, above)
streamlit run app.py                      # search UI
```

Full command reference, all architecture decisions with their reasoning, every bug found and
how it was fixed (or knowingly not fixed), and both evaluation rounds' full findings are in
[`CLAUDE.md`](CLAUDE.md) — the engineering log this README was distilled from.

## Known limitations

- **Entity deduplication is imperfect**: the same real-world entity extracted with slightly
  different wording across chunks (e.g. `"ENISA"` vs. `"ENISA (the European Union Agency for
  Cybersecurity)"`) currently becomes separate graph nodes rather than one merged node.
- **Generic self-references** (`"this Regulation"`) are filtered, but only the `"(this|the) X"`
  phrasing — a bare `"Regulation"`/`"Directive"` with no leading article still slips through and
  collides across unrelated documents.
- **The fixed relation-type vocabulary** sometimes lacks a good fit for a real relationship
  (e.g. "adopts", "notifies") — round 2's evaluation showed the model increasingly identifying
  *that* two things are related correctly, but not always *how*, within a closed 10-type list.
- **No recall measurement** — only precision is evaluated; nothing yet checks the graph for
  entities/relations that should exist but were missed entirely.

## Tech stack

Python 3.9 · `llama-cpp-python` (Qwen2.5-7B/3B, GGUF, Metal-accelerated) · Neo4j AuraDB ·
Google Gemini (`google-genai`, free tier) · PyMuPDF · Streamlit · MLflow · pytest
