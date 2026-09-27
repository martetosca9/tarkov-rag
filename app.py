"""Streamlit Web UI for Tarkov Quest RAG Assistant."""

import os
from pathlib import Path
import sys
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag import TarkovRAG
from src.vectorstore import QuestVectorStore

# Page configuration
st.set_page_config(
    page_title="Tarkov Quest RAG Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stAlert {
        border-radius: 8px;
    }
    .quest-card {
        background-color: #1a1c24;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .prereq-badge {
        background-color: #2b3a4a;
        color: #58a6ff;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_rag_engine():
    """Initialize and cache the Quest Vector Store and RAG pipeline."""
    store = QuestVectorStore()
    return TarkovRAG(vector_store=store)


rag = get_rag_engine()

# Sidebar
with st.sidebar:
    st.title("🎯 Tarkov RAG")
    st.caption("AI-Powered Quest Intelligence for Escape from Tarkov")

    st.markdown("---")
    st.subheader("⚙️ Query Settings")

    top_k = st.slider("Context Quests (Top K)", min_value=1, max_value=5, value=3)

    traders = [
        "All Traders",
        "Prapor",
        "Therapist",
        "Skier",
        "Peacekeeper",
        "Mechanic",
        "Ragman",
        "Jaeger",
        "Fence",
        "Lightkeeper",
        "Ref",
        "BTR Driver",
    ]
    selected_trader = st.selectbox("Filter by Trader", options=traders)
    trader_filter = None if selected_trader == "All Traders" else selected_trader

    force_offline = st.checkbox(
        "Direct Vector Mode (No LLM API)",
        value=not rag.has_llm_client,
        help="Display raw extracted quest data and prerequisite graph without invoking an LLM API.",
    )

    st.markdown("---")
    st.subheader("🔑 API Key Status")
    if rag.has_llm_client:
        st.success("OpenAI API Key detected in `.env`")
    else:
        st.info("Running in offline vector mode. Add `OPENAI_API_KEY` to `.env` for generative answers.")

    st.markdown("---")
    st.caption(f"Indexed Quests: **{len(rag.vector_store._quest_graph)}**")
    st.caption("Data Source: Escape from Tarkov Wiki")

# Header
st.title("🎯 Escape from Tarkov Quest Assistant")
st.markdown(
    "Ask natural language questions about EFT quests, prerequisites, item requirements, and unlocks. "
    "Every answer includes citations back to the source wiki page."
)

# Example query pills
example_queries = [
    "What do I need to unlock Textile?",
    "What are the objectives for Debut?",
    "What are the rewards for Chemical - Part 4?",
    "Who gives Background Check and where is it located?",
]

selected_example = st.pills("Quick Examples:", options=example_queries, selection_mode="single")

# Query input
query = st.text_input(
    "Ask a quest question:",
    value=selected_example or "",
    placeholder="e.g. What items do I need to find in raid for Collector?",
)

if st.button("Search & Answer", type="primary", use_container_width=False) or query:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching quest database and resolving dependency chains..."):
            result = rag.answer_question(
                question=query,
                n_results=top_k,
                trader_filter=trader_filter,
                force_offline=force_offline,
            )

        # Main Answer Panel
        st.markdown("### 💡 Answer")
        st.markdown(result["answer"])

        # Source Citations & Prerequisite Chains
        if result.get("retrieved"):
            st.markdown("---")
            st.markdown("### 📚 Source Quests & Dependency Chains")

            for item in result["retrieved"]:
                with st.expander(f"📍 {item['name']} (Trader: {item['trader']}) - Similarity: {item['similarity']:.3f}", expanded=True):
                    cols = st.columns([1, 1])
                    with cols[0]:
                        st.markdown(f"**Trader:** {item['trader']}")
                        st.markdown(f"**Location:** {item.get('location') or 'Any / Various'}")
                        st.markdown(f"**Wiki Link:** [Open Wiki Page]({item['wiki_url']})")

                    with cols[1]:
                        chain = item.get("prerequisite_chain", [])
                        if chain:
                            st.markdown(f"**Prerequisite Chain ({len(chain)} steps):**")
                            st.code(" → ".join(chain))
                        else:
                            st.markdown("**Prerequisites:** None (Starter Quest or Level Only)")

                    st.markdown("**Full Structured Content:**")
                    st.code(item["document"], language="markdown")
