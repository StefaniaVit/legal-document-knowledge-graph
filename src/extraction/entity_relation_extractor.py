"""
Extracts entities and relations from document chunks with a local LLM,
using grammar-constrained JSON decoding (llama.cpp compiles the JSON
schema into a GBNF grammar, so output is guaranteed schema-valid --
no free-text generation + hope-it-parses fallback needed).

Entities and relations are extracted in two separate calls rather than
one. A single combined call was tried first and asked a 3B model to
juggle both tasks at once, which produced entities that were entire
run-on sentences instead of bounded spans, and relations that referenced
text never listed as an entity at all -- directly violating the prompt's
own instruction to only reference listed entities. Splitting means
relation extraction is grounded in an already-confirmed entity list (fed
back into the second call), and any relation that still names something
outside that list is dropped programmatically as a hard backstop,
regardless of how well the model follows the instruction.
"""
import json
import re
from pathlib import Path
from typing import Optional

from llama_cpp import Llama, LlamaGrammar

MODEL_PATH = Path(__file__).parents[2] / "models" / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

ENTITY_TYPES = ["ORGANIZATION", "LEGAL_ACT", "LEGAL_CONCEPT", "DATE", "PENALTY", "OBLIGATION"]
RELATION_TYPES = [
    "AMENDS", "REPEALS", "REFERENCES", "DEFINES", "IMPOSES_OBLIGATION_ON",
    "ESTABLISHES", "APPLIES_TO", "SUBJECT_TO", "GRANTS_RIGHT_TO", "RESPONSIBLE_FOR",
]

_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": ENTITY_TYPES},
                },
                "required": ["text", "type"],
            },
        },
    },
    "required": ["entities"],
}

_RELATION_SCHEMA = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {"type": "string", "enum": RELATION_TYPES},
                    "object": {"type": "string"},
                },
                "required": ["subject", "relation", "object"],
            },
        },
    },
    "required": ["relations"],
}

_ENTITY_SYSTEM_PROMPT = (
    "You extract entities from EU legislation text for a knowledge graph.\n"
    f"Entity types: {', '.join(ENTITY_TYPES)}.\n"
    "Entities must be short noun phrases (2-6 words), never a full sentence or clause.\n"
    "LEGAL_ACT is not limited to numbered EU citations like \"Regulation (EU) 2016/679\" -- "
    "it also covers named treaties, conventions, and agreements, not just the organization "
    "that negotiated one.\n"
    "Return an empty list if nothing applies. Never invent an entity that isn't present in "
    "the given text, including anything from these instructions or examples."
)

# Real, worked few-shot exchange (not a prose example embedded in the system
# prompt) -- an earlier version described the LEGAL_ACT-scope rule with an
# inline worked example in the system prompt text itself, first citing a
# real-sounding fabricated agency ("Senior Officials Group Mutual Recognition
# Agreement (MRA)"), then a deliberately fictional one ("Zorvath Cooperation
# Framework Mutual Recognition Accord" / "Xylanti Standards Board") after the
# first leaked. Both were observed to leak near-verbatim into unrelated real
# text about treaties/frameworks, 100% reproducible at temperature=0 -- the
# fictional naming made a leak detectable rather than dangerously plausible,
# but did not stop Qwen2.5-7B from regurgitating it on thematically similar
# input, and the confused generation this produced also correlated with a
# single call taking ~600s instead of the usual 5-20s. The LEGAL_ACT-scope
# rule is stated in the system prompt above with no worked example at all
# now -- the example itself, not its content, appears to be what the model
# was pattern-matching against and echoing back.
_ENTITY_FEWSHOT = [
    ("user", "Each supervisory authority shall ensure that the imposition of administrative "
             "fines pursuant to this Article in respect of infringements of this Regulation "
             "shall in each individual case be effective, proportionate and dissuasive."),
    ("assistant", '{"entities": [{"text": "supervisory authority", "type": "ORGANIZATION"}, '
                  '{"text": "administrative fines", "type": "PENALTY"}]}'),
]

_RELATION_SYSTEM_PROMPT = (
    "You extract relations between entities in EU legislation text for a knowledge graph.\n"
    f"Relation types: {', '.join(RELATION_TYPES)}.\n"
    'You will be given the text and a list of entities already confirmed present in it. '
    'subject and object must each exactly match the "text" of one of those entities -- '
    "never introduce a subject or object that isn't in the given list.\n"
    "Return an empty list if no relation holds between any of the given entities."
)

_llm: Optional[Llama] = None
_entity_grammar: Optional[LlamaGrammar] = None
_relation_grammar: Optional[LlamaGrammar] = None


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _filter_valid_relations(relations: list[dict], entities: list[dict]) -> list[dict]:
    """Keeps only relations whose subject and object both match a confirmed
    entity's text. Matched case/whitespace-insensitively rather than by exact
    string equality -- entities and relations come from separate LLM calls,
    and the model isn't guaranteed to echo a name back with identical
    casing/whitespace the second time, which would otherwise silently drop
    an otherwise-valid relation.
    """
    valid = {_normalize_for_match(e["text"]) for e in entities}
    return [
        r for r in relations
        if _normalize_for_match(r["subject"]) in valid and _normalize_for_match(r["object"]) in valid
    ]


def _get_llm() -> Llama:
    global _llm
    if _llm is None:
        # n_gpu_layers defaults to 0 (CPU-only) in llama-cpp-python; -1 offloads
        # all layers to Metal on this Mac, ~5x faster than the CPU-only default.
        _llm = Llama(model_path=str(MODEL_PATH), n_ctx=4096, verbose=False, n_gpu_layers=-1)
    return _llm


def _get_entity_grammar() -> LlamaGrammar:
    global _entity_grammar
    if _entity_grammar is None:
        _entity_grammar = LlamaGrammar.from_json_schema(json.dumps(_ENTITY_SCHEMA))
    return _entity_grammar


def _get_relation_grammar() -> LlamaGrammar:
    global _relation_grammar
    if _relation_grammar is None:
        _relation_grammar = LlamaGrammar.from_json_schema(json.dumps(_RELATION_SCHEMA))
    return _relation_grammar


def extract_entities(chunk_text: str) -> list[dict]:
    messages = [{"role": "system", "content": _ENTITY_SYSTEM_PROMPT}]
    for role, content in _ENTITY_FEWSHOT:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": chunk_text})

    out = _get_llm().create_chat_completion(
        messages=messages,
        grammar=_get_entity_grammar(),
        temperature=0,
        repeat_penalty=1.3,
        max_tokens=1000,
    )
    raw = out["choices"][0]["message"]["content"]
    return json.loads(raw)["entities"]


def extract_relations(chunk_text: str, entities: list[dict]) -> list[dict]:
    if len(entities) < 2:
        return []

    entity_list = "\n".join(f'- "{e["text"]}" ({e["type"]})' for e in entities)
    user_content = f"Text:\n{chunk_text}\n\nConfirmed entities:\n{entity_list}"

    messages = [
        {"role": "system", "content": _RELATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    out = _get_llm().create_chat_completion(
        messages=messages,
        grammar=_get_relation_grammar(),
        temperature=0,
        repeat_penalty=1.3,
        max_tokens=1000,
    )
    raw = out["choices"][0]["message"]["content"]
    relations = json.loads(raw)["relations"]
    return _filter_valid_relations(relations, entities)


def extract(chunk_text: str) -> dict:
    entities = extract_entities(chunk_text)
    relations = extract_relations(chunk_text, entities)
    return {"entities": entities, "relations": relations}
