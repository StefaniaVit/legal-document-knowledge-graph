"""
Streamlit front end for the GraphRAG query layer (src/rag/graph_rag.py).

Wraps graph_rag.answer() with a search box, showing not just the synthesized
answer but the resolved entities and retrieved graph context that grounded it
-- that transparency is the whole point of graph-based retrieval over a plain
chatbot, so the UI surfaces it rather than hiding it.

Run with: streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.rag.graph_rag import answer

st.set_page_config(page_title="Legal KG Search", page_icon="⚖️", layout="wide")

st.title("Legal Document Knowledge Graph — GraphRAG Search")
st.caption(
    "Ask a question about GDPR, the Cybersecurity Act, NIS2, or the Data Governance Act. "
    "Answers are grounded in a knowledge graph built from the source PDFs: entities are "
    "resolved first, then only the graph's own relations and source text are used to "
    "synthesize the answer below."
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("question_form"):
    question = st.text_input(
        "Your question",
        placeholder="e.g. What obligations does ENISA have?",
    )
    submitted = st.form_submit_button("Search")

if submitted and question.strip():
    with st.spinner("Resolving entities and retrieving graph context..."):
        try:
            result = answer(question)
        except Exception as e:
            result = None
            st.error(
                f"Query failed: {e}\n\n"
                "If this is a Neo4j connection error, the AuraDB instance may be paused "
                "due to inactivity -- resume it at console.neo4j.io and try again."
            )
    if result is not None:
        st.session_state.history.insert(0, {"question": question, **result})

for item in st.session_state.history:
    st.markdown(f"### Q: {item['question']}")
    st.markdown(item["answer"])

    if item["entities"]:
        with st.expander(f"Matched entities ({len(item['entities'])})"):
            st.table([{"Name": e["name"], "Type": e["type"]} for e in item["entities"]])

    if item["context"]:
        with st.expander("Retrieved graph context (raw)"):
            st.text(item["context"])

    st.divider()
