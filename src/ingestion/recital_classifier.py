"""
Distinguishes genuine EU legislation recitals from footnote citations.

Both use identical "(N) <text>" formatting, and footnote numbering resets
on every page, so a footnote can coincidentally carry the exact number the
monotonic recital sequence expects next (see parser.py). That collision
can't be resolved by position or formatting alone -- it requires reading
the text: a citation is a bare bibliographic reference ending in
"(OJ L ..., p. ...)."; a recital is substantive legal reasoning.
"""
import re
from pathlib import Path
from typing import Optional

from llama_cpp import Llama

MODEL_PATH = Path(__file__).parents[2] / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

_SYSTEM_PROMPT = (
    "You classify short text spans from EU legislation preambles as RECITAL or CITATION.\n"
    "A CITATION is always a bare bibliographic reference to another legal instrument, "
    'always ending in a page/date reference like "(OJ L ..., p. ...)." It contains no '
    "policy reasoning.\n"
    "A RECITAL is substantive legal reasoning about rights, obligations, or policy "
    'justification. It never ends in "(OJ L ..., p. ...)."\n'
    "Reply with exactly one word: RECITAL or CITATION."
)

_FEWSHOT = [
    ("user", "Council Directive 93/13/EEC of 5 April 1993 on unfair terms in consumer "
             "contracts (OJ L 95, 21.4.1993, p. 29)."),
    ("assistant", "CITATION"),
    ("user", "Member States should ensure that their competent authorities have adequate "
             "powers and resources to carry out their tasks under this Regulation in an "
             "effective and efficient manner."),
    ("assistant", "RECITAL"),
    # A recital can open by *naming* another act without citing it -- the
    # giveaway is that it keeps going with substantive reasoning instead
    # of stopping at a page reference.
    ("user", "Directive 95/46/EC of the European Parliament and of the Council sought to "
             "harmonise the protection of natural persons' fundamental rights and freedoms "
             "in respect of processing activities and to ensure the free flow of personal "
             "data between Member States."),
    ("assistant", "RECITAL"),
]

_OJ_END_RE = re.compile(r"\(OJ [^)]*\)\.")

_llm: Optional[Llama] = None


def _get_llm() -> Llama:
    global _llm
    if _llm is None:
        # n_gpu_layers defaults to 0 (CPU-only) in llama-cpp-python; -1 offloads
        # all layers to Metal on this Mac, ~5x faster than the CPU-only default.
        _llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, verbose=False, n_gpu_layers=-1)
    return _llm


def _snippet(candidate_text: str, window: int = 300) -> str:
    """Cuts at the citation's natural end ('(OJ ...).') if present within
    the window, so a short citation doesn't bleed into the next paragraph
    and confuse the classifier."""
    w = candidate_text[:window]
    m = _OJ_END_RE.search(w)
    return w[:m.end()] if m else candidate_text[:200]


def is_recital(candidate_text: str) -> bool:
    """True if candidate_text reads as a genuine recital, False if it
    reads as a footnote citation."""
    snippet = _snippet(candidate_text)
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for role, content in _FEWSHOT:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": snippet})

    out = _get_llm().create_chat_completion(messages=messages, max_tokens=10, temperature=0)
    answer = out["choices"][0]["message"]["content"].strip().upper()
    return answer.startswith("RECITAL")
