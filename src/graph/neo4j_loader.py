"""
Loads parsed documents, chunks, and extracted entities/relations into Neo4j.

Schema:
  (Document {celex, title, date})
    -[:HAS_CHUNK]-> (Chunk {chunk_id, chunk_type, unit_number, part, heading, text})
    -[:MENTIONS]-> (Entity:<TYPE> {key, name, type})
  (Entity)-[:<RELATION_TYPE>]->(Entity)

Entity and relation type names come from a small, fixed whitelist controlled by
this codebase (ENTITY_TYPES / RELATION_TYPES), never from raw extraction output,
before being interpolated into Cypher as dynamic labels/relationship types --
Cypher has no parameter syntax for those, so this is the standard approach, and
it's safe specifically because the whitelist is closed and not user input.

Entities are deduplicated across the whole corpus by normalized name alone, not
name+type. An earlier version keyed on (name, type) together, which meant that
if the model tagged the same real entity with different types in different
chunks -- observed for "ENISA": ORGANIZATION in ~110 mentions, but LEGAL_ACT and
LEGAL_CONCEPT in a couple of others -- each type produced its own separate node,
fragmenting what should be one well-connected hub into several weaker ones.
Deduplicating by name and resolving a single canonical type via majority vote
across all of that name's mentions (see _compute_canonical_types) fixes this
without needing to re-run extraction.

Self-referential entities ("this Regulation", "this Directive", "this Act")
are dropped -- without resolving them to the specific document they refer to,
they'd incorrectly merge across unrelated documents into one meaningless shared
node.
"""
import json
import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from src.extraction.entity_relation_extractor import ENTITY_TYPES, RELATION_TYPES

load_dotenv()

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
CHUNKS_DIR = Path(__file__).parents[2] / "data" / "chunks"
EXTRACTION_PATH = Path(__file__).parents[2] / "data" / "extraction_all.jsonl"

_SELF_REFERENCE_RE = re.compile(r"^(this|the)\s+(regulation|directive|act|decision)$", re.IGNORECASE)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _entity_key(text: str) -> str:
    return _normalize(text)


def _is_self_reference(text: str) -> bool:
    return bool(_SELF_REFERENCE_RE.match(text.strip()))


def _compute_canonical_types(extraction_results: list[dict]) -> dict[str, str]:
    """One canonical type per normalized entity name, chosen by majority vote
    across every mention of that name in the whole corpus."""
    votes: dict[str, Counter] = {}
    for result in extraction_results:
        for e in result["entities"]:
            if _is_self_reference(e["text"]) or e["type"] not in ENTITY_TYPES:
                continue
            name = _normalize(e["text"])
            votes.setdefault(name, Counter())[e["type"]] += 1
    return {name: counter.most_common(1)[0][0] for name, counter in votes.items()}


def get_driver() -> GraphDatabase.driver:
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )


def setup_constraints(session) -> None:
    session.run("CREATE CONSTRAINT document_celex IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.celex IS UNIQUE")
    session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
    session.run("CREATE CONSTRAINT entity_key IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.key IS UNIQUE")


def clear_database(session) -> None:
    session.run("MATCH (n) DETACH DELETE n")


def load_documents_and_chunks(session, chunk_ids_with_extraction: set) -> None:
    for doc_path in sorted(PROCESSED_DIR.glob("*.json")):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        celex = doc["celex"]

        session.run(
            "MERGE (d:Document {celex: $celex}) SET d.title = $title, d.date = $date",
            celex=celex, title=doc["title"], date=doc["date"],
        )

        chunks_path = CHUNKS_DIR / f"{celex}_chunks.json"
        if not chunks_path.exists():
            continue
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        for c in chunks:
            if c["chunk_id"] not in chunk_ids_with_extraction:
                continue  # only load chunks we've actually run extraction on, for now
            session.run(
                """
                MATCH (d:Document {celex: $celex})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.chunk_type = $chunk_type, c.unit_number = $unit_number,
                    c.part = $part, c.heading = $heading, c.text = $text
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                celex=celex, chunk_id=c["chunk_id"], chunk_type=c["chunk_type"],
                unit_number=c["unit_number"], part=c["part"], heading=c["heading"],
                text=c["text"],
            )


def load_entities_and_relations(
    session, extraction_results: list[dict], canonical_types: dict[str, str],
) -> None:
    for result in extraction_results:
        chunk_id = result["chunk_id"]
        entity_keys = {}  # entity text -> key, to resolve this chunk's relation subject/object

        for e in result["entities"]:
            text, etype = e["text"], e["type"]
            if _is_self_reference(text) or etype not in ENTITY_TYPES:
                continue
            key = _entity_key(text)
            canonical_type = canonical_types[key]
            entity_keys[text] = key
            session.run(
                f"""
                MERGE (e:Entity:{canonical_type} {{key: $key}})
                ON CREATE SET e.name = $name, e.type = $type
                """,
                key=key, name=text, type=canonical_type,
            )
            session.run(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})
                MATCH (e:Entity {key: $key})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                chunk_id=chunk_id, key=key,
            )

        for r in result["relations"]:
            rel_type = r["relation"]
            subj_key = entity_keys.get(r["subject"])
            obj_key = entity_keys.get(r["object"])
            if not subj_key or not obj_key or rel_type not in RELATION_TYPES:
                continue  # subject/object was dropped as a self-reference, or unresolved
            session.run(
                f"""
                MATCH (a:Entity {{key: $subj_key}})
                MATCH (b:Entity {{key: $obj_key}})
                MERGE (a)-[:{rel_type}]->(b)
                """,
                subj_key=subj_key, obj_key=obj_key,
            )


def load_all(wipe_first: bool = True) -> None:
    extraction_results = [
        json.loads(line)
        for line in EXTRACTION_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunk_ids_with_extraction = {r["chunk_id"] for r in extraction_results}
    canonical_types = _compute_canonical_types(extraction_results)

    driver = get_driver()
    with driver.session(database=os.environ.get("NEO4J_DATABASE")) as session:
        if wipe_first:
            clear_database(session)
        setup_constraints(session)
        load_documents_and_chunks(session, chunk_ids_with_extraction)
        load_entities_and_relations(session, extraction_results, canonical_types)
    driver.close()

    print(f"Loaded {len(extraction_results)} chunks with extraction data into Neo4j.")


if __name__ == "__main__":
    load_all()
