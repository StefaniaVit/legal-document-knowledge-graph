"""
GraphRAG query layer: answers natural-language questions by resolving which
known graph entities the question refers to, retrieving that entity's
neighborhood (relations + source chunk text) from Neo4j, then asking an LLM
to synthesize an answer grounded in that retrieved context -- retrieval via
graph traversal instead of vector similarity search.

Entity resolution and answer synthesis both use Gemini (hosted, free tier).
Retrieval itself is fixed Cypher, not LLM-generated ("text-to-Cypher") --
a small or even large model asked to write Cypher freeform risks syntactically
invalid or semantically wrong queries with no easy way to catch the mistake;
a fixed template with a resolved entity key as its only parameter can't be
wrong in that way. The LLM's job is narrower: pick which of the graph's real
entities a question relates to, and turn retrieved graph context into a
readable answer.

Entity resolution gives the model the full list of ~1200 known entity names
directly and asks it to select from that list, rather than asking it to
invent candidate strings blind and then fuzzy-matching those against the
graph via substring containment. The blind-then-fuzzy approach was tried
first and failed in two different ways on real questions: a question about
"penalties" never matched the graph's "administrative fines" entity at all
(no lexical overlap between the words, even though they're obviously the
same concept), while a question containing the word "obligation" matched
dozens of unrelated long OBLIGATION-type entities purely because they
happened to contain that substring. Both failures come from the same root
cause -- CONTAINS matching can only find lexical overlap, not semantic
relationships, which is exactly the gap an LLM closes when it's shown the
real options instead of pattern-matching against them after the fact.
"""
import os
import re
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from neo4j import GraphDatabase

load_dotenv()

MAX_CHUNKS_PER_ENTITY = 5
GEMINI_MODEL = "gemini-2.5-flash-lite"

_client: Optional[genai.Client] = None
_driver = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
    return _driver


def _get_all_entity_names(session) -> list[str]:
    result = session.run("MATCH (e:Entity) RETURN DISTINCT e.name AS name ORDER BY name")
    return [r["name"] for r in result]


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _truncate_at_sentence(text: str, limit: int) -> str:
    """Truncates to at most `limit` characters, but backs up to the nearest
    preceding sentence boundary rather than cutting mid-sentence -- a blind
    slice (`text[:limit]`) tends to land mid-word/mid-clause on longer
    chunks, silently dropping whatever clause happened to be there.
    """
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    boundary = max(truncated.rfind(". "), truncated.rfind("; "))
    if boundary > 0:
        return truncated[:boundary + 1]
    return truncated.rstrip() + "..."


def _filter_known_names(candidate_names: list[str], known_names: list[str]) -> list[str]:
    """Keeps only candidates that match a known entity name, matched
    case/whitespace-insensitively rather than by exact string equality --
    the model isn't guaranteed to echo a name back with identical
    casing/whitespace, which would otherwise silently drop a valid match.
    Returns the *known* name (not the candidate's own casing), since
    downstream Cypher lookups need to match the name actually stored in
    the graph.
    """
    by_normalized = {}
    for name in known_names:
        by_normalized.setdefault(_normalize_for_match(name), name)

    result = []
    seen = set()
    for candidate in candidate_names:
        key = _normalize_for_match(candidate)
        if key in by_normalized and key not in seen:
            seen.add(key)
            result.append(by_normalized[key])
    return result


def _select_relevant_entities(question: str, entity_names: list[str]) -> list[str]:
    """Gives Gemini the full list of real entity names in the graph and asks
    it to pick which ones the question relates to -- closing the semantic
    gap a plain string-matching lookup can't (e.g. recognizing "penalties" in
    the question relates to a graph entity literally named "administrative
    fines"). Every name it returns is guaranteed to exist, since it's
    choosing from ground truth rather than generating candidates blind.
    """
    names_list = "\n".join(entity_names)
    prompt = (
        "Below is the complete list of entity names that exist in a knowledge graph "
        "built from EU legislation. Select every entity name relevant to answering the "
        "question. Reply with the exact names, one per line, copied verbatim from the "
        "list below -- do not alter them or invent names not in the list. "
        "If none are relevant, reply with NONE.\n\n"
        f"Entity names:\n{names_list}\n\n"
        f"Question: {question}"
    )
    # temperature=0: this is an exhaustive-selection task ("pick every relevant
    # name"), not open-ended generation -- the default sampling temperature
    # made repeat calls with the identical question return different numbers
    # of matched entities (e.g. 22 vs. 1), which isn't desirable here.
    response = _get_client().models.generate_content(
        model=GEMINI_MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    lines = [l.strip() for l in response.text.splitlines() if l.strip()]
    if lines == ["NONE"]:
        return []
    return _filter_known_names(lines, entity_names)


def _resolve_entities(session, names: list[str]) -> list[dict]:
    resolved = []
    for name in names:
        for r in session.run(
            "MATCH (e:Entity {name: $name}) RETURN e.key AS key, e.name AS name, e.type AS type",
            name=name,
        ):
            resolved.append({"key": r["key"], "name": r["name"], "type": r["type"]})
    return resolved


def _retrieve_context(session, entities: list[dict]) -> str:
    """For each resolved entity: its relations to other entities, and a
    sample of chunk text that mentions it, formatted as readable text with
    citations -- ready to drop straight into the synthesis prompt."""
    sections = []
    for entity in entities:
        key = entity["key"]

        relations = session.run(
            """
            MATCH (a:Entity {key: $key})-[r]-(b:Entity)
            RETURN a.name AS a_name, type(r) AS rel, b.name AS b_name,
                   startNode(r).key = $key AS a_is_subject
            LIMIT 15
            """,
            key=key,
        )
        rel_lines = [
            f"  {r['a_name']} --{r['rel']}--> {r['b_name']}" if r["a_is_subject"]
            else f"  {r['b_name']} --{r['rel']}--> {r['a_name']}"
            for r in relations
        ]

        chunks = session.run(
            """
            MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity {key: $key})
            RETURN d.title AS document, c.chunk_type AS chunk_type,
                   c.unit_number AS unit_number, c.text AS text
            LIMIT $limit
            """,
            key=key, limit=MAX_CHUNKS_PER_ENTITY,
        )
        chunk_lines = [
            f"  [{c['document'][:50]} -- {c['chunk_type']} {c['unit_number']}]\n"
            f"  {_truncate_at_sentence(c['text'], 600)}"
            for c in chunks
        ]

        sections.append(
            f"### {entity['name']} ({entity['type']})\n"
            "Relations:\n" + ("\n".join(rel_lines) if rel_lines else "  (none found)") + "\n"
            "Mentioned in:\n" + ("\n\n".join(chunk_lines) if chunk_lines else "  (no chunks found)")
        )
    return "\n\n".join(sections)


def _synthesize_answer(question: str, context: str) -> str:
    prompt = (
        "Answer the question using ONLY the context below, which was retrieved from a "
        "knowledge graph built from EU legislation. Cite the document and article/recital "
        "for each claim. If the context doesn't contain the answer, say so explicitly -- "
        "do not guess.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = _get_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


def answer(question: str) -> dict:
    driver = _get_driver()
    with driver.session(database=os.environ.get("NEO4J_DATABASE")) as session:
        entity_names = _get_all_entity_names(session)
        selected_names = _select_relevant_entities(question, entity_names)
        entities = _resolve_entities(session, selected_names)
        if not entities:
            return {
                "answer": "No matching entities found in the graph for this question.",
                "entities": [], "context": "",
            }
        context = _retrieve_context(session, entities)

    return {"answer": _synthesize_answer(question, context), "entities": entities, "context": context}


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What obligations does ENISA have?"
    result = answer(question)
    print(f"Q: {question}\n")
    print(f"A: {result['answer']}\n")
    print(f"Matched entities: {[e['name'] for e in result['entities']]}")
