"""Streamlit Web UI for Tarkov Quest RAG Assistant with Tactical Military Aesthetic."""

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
    page_title="TARKOV // INTEL RAG",
    page_icon="🎖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tarkov War / Bunker / Typewriter Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:ital,wght@0,400;0,700;1,400&family=Share+Tech+Mono&family=Oswald:wght@500;700&display=swap');

    /* Global Typography & Palette */
    html, body, [class*="css"], .stMarkdown, p, span, label, div {
        font-family: 'Courier Prime', 'Share Tech Mono', 'Courier New', monospace !important;
        color: #d1d5db;
    }

    h1, h2, h3, h4, .stTitle {
        font-family: 'Oswald', 'Share Tech Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #e59b2c !important; /* Tarkov Signature Amber */
    }

    /* Backgrounds */
    .stApp {
        background-color: #0b0d10 !important;
        background-image: 
            radial-gradient(#151921 1px, transparent 1px),
            linear-gradient(to bottom, #090a0d 0%, #0d1015 100%);
        background-size: 24px 24px, 100% 100%;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0e1116 !important;
        border-right: 1px solid #232730 !important;
    }

    /* Input Field */
    div[data-baseweb="input"] {
        background-color: #12151c !important;
        border: 1px solid #3a414d !important;
        border-radius: 2px !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #e59b2c !important;
        box-shadow: 0 0 8px rgba(229, 155, 44, 0.4) !important;
    }
    input[data-testid="stTextInputRootElement"] {
        color: #f3f4f6 !important;
        font-family: 'Courier Prime', monospace !important;
        font-size: 1.05rem !important;
    }

    /* Primary Action Buttons */
    button[kind="primary"], .stButton > button {
        font-family: 'Oswald', 'Share Tech Mono', monospace !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        background-color: #181d26 !important;
        color: #e59b2c !important;
        border: 1px solid #e59b2c !important;
        border-radius: 2px !important;
        padding: 8px 24px !important;
        transition: all 0.2s ease-in-out !important;
    }
    button[kind="primary"]:hover, .stButton > button:hover {
        background-color: #e59b2c !important;
        color: #090b0e !important;
        box-shadow: 0 0 12px rgba(229, 155, 44, 0.6) !important;
        border-color: #e59b2c !important;
    }

    /* Pill Selection */
    button[data-testid="stPill"] {
        font-family: 'Courier Prime', monospace !important;
        background-color: #13171f !important;
        color: #9ca3af !important;
        border: 1px dashed #2f3542 !important;
        border-radius: 2px !important;
        font-size: 0.85rem !important;
    }
    button[data-testid="stPill"][aria-selected="true"] {
        background-color: #242c1f !important;
        color: #7fa865 !important;
        border: 1px solid #7fa865 !important;
    }

    /* Tactical Cards & Panels */
    .tactical-panel {
        background: #10131a;
        border: 1px solid #282f3c;
        border-left: 4px solid #e59b2c;
        border-radius: 2px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }

    .tactical-badge {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        padding: 3px 8px;
        border-radius: 2px;
        text-transform: uppercase;
        background: #1b2217;
        color: #7fa865;
        border: 1px solid #3c4d32;
        margin-right: 8px;
    }

    .tactical-badge-amber {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        padding: 3px 8px;
        border-radius: 2px;
        text-transform: uppercase;
        background: #2a1f10;
        color: #e59b2c;
        border: 1px solid #573f1d;
        margin-right: 8px;
    }

    /* Expander / Intel Details */
    .streamlit-expanderHeader {
        background-color: #12161f !important;
        border: 1px solid #252b37 !important;
        border-radius: 2px !important;
        font-family: 'Share Tech Mono', monospace !important;
        color: #d1d5db !important;
    }
    .streamlit-expanderContent {
        background-color: #0d1017 !important;
        border: 1px solid #252b37 !important;
        border-top: none !important;
    }

    /* Code Blocks (Military Terminal Look) */
    pre, code {
        font-family: 'Courier Prime', 'Share Tech Mono', monospace !important;
        background-color: #07080b !important;
        color: #6bb05d !important; /* Night vision terminal phosphor green */
        border: 1px solid #1a2217 !important;
        border-radius: 2px !important;
    }

    /* Links */
    a {
        color: #e59b2c !important;
        text-decoration: underline !important;
    }
    a:hover {
        color: #ffc069 !important;
    }

    hr {
        border-color: #232730 !important;
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

# Sidebar - Tactical Controls
with st.sidebar:
    st.markdown("### 📡 COMMS & INTEL // CONFIG")
    st.caption("TACTICAL DATABASE // ESCAPE FROM TARKOV")

    st.markdown("---")
    st.markdown("#### 🎯 OPERATIONAL FILTER")

    traders = [
        "ALL TRADERS",
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
    selected_trader = st.selectbox("OPERATIVE / TRADER", options=traders)
    trader_filter = None if selected_trader == "ALL TRADERS" else selected_trader

    top_k = st.slider("INTEL DEPTH (CANDIDATES)", min_value=1, max_value=5, value=3)

    force_offline = st.checkbox(
        "DIRECT TELEMETRY (NO LLM API)",
        value=not rag.has_llm_client,
        help="Query vector store and dependency tree directly without external AI calls.",
    )

    st.markdown("---")
    st.markdown("#### 🔒 ENCRYPTION / API STATUS")
    if rag.has_llm_client:
        st.markdown("<span class='tactical-badge'>LINKED: OPENAI GPT-4o-mini</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='tactical-badge-amber'>STANDALONE: OFFLINE VECTOR DB</span>", unsafe_allow_html=True)
        st.caption("Add OPENAI_API_KEY to .env for generative tactical briefings.")

    st.markdown("---")
    st.caption(f"ARCHIVED MISSIONS: **{len(rag.vector_store._quest_graph)}**")
    st.caption("SOURCE: EFT MEDIAWIKI CLASSIFIED REPO")

# Tactical Header
st.markdown("## 🎖️ ESCAPE FROM TARKOV // INTEL TERMINAL")
st.markdown(
    "<div style='font-family: \"Share Tech Mono\", monospace; color: #8c97a8; letter-spacing: 1px; margin-bottom: 24px;'>"
    "SEARCH MISSIONS, ITEMS, CRAFTS, PREREQUISITES & UNLOCKS // ASK ANYTHING"
    "</div>",
    unsafe_allow_html=True,
)

# Quick Tactical Query Pills
quick_intel = [
    "What do I need to unlock Textile?",
    "What are the objectives for Debut?",
    "What are the rewards for Chemical - Part 4?",
    "Who gives Background Check and where is it located?",
    "Which quests give a grenade case?",
]

selected_pill = st.pills("QUICK INTEL SCAN:", options=quick_intel, selection_mode="single")

# Search Field
query = st.text_input(
    "INPUT QUERY //:",
    value=selected_pill or "",
    placeholder="Ask anything (e.g. objectives for Debut, what unlocks Textile, items for Collector)...",
    label_visibility="collapsed",
)

col_btn, _ = st.columns([1, 5])
with col_btn:
    run_search = st.button("EXECUTE SCAN ↵", type="primary", use_container_width=True)

if run_search or query:
    if not query.strip():
        st.warning("[WARNING] ENTER A VALID QUERY PARAMETER.")
    else:
        with st.spinner("TRANSMITTING ENCRYPTED REQUEST // QUERYING DATABASE..."):
            result = rag.answer_question(
                question=query,
                n_results=top_k,
                trader_filter=trader_filter,
                force_offline=force_offline,
            )

        # Answer Section
        st.markdown("---")
        st.markdown("### 📄 DECRYPTED INTEL REPORT")
        st.markdown(
            f"<div class='tactical-panel'>{result['answer']}</div>",
            unsafe_allow_html=True,
        )

        # Source Quests & Dependency Chains
        if result.get("retrieved"):
            st.markdown("### 🗂️ MISSION RECORDS & PREREQUISITE CHAINS")

            for item in result["retrieved"]:
                chain = item.get("prerequisite_chain", [])
                chain_summary = f" // PREREQUISITES: {len(chain)}" if chain else " // STARTER MISSION"

                with st.expander(
                    f"📁 {item['name'].upper()} [TRADER: {item['trader'].upper()}{chain_summary}] - SCORE: {item['similarity']:.3f}",
                    expanded=True,
                ):
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.markdown(f"**GIVER:** {item['trader']}")
                        st.markdown(f"**MAP DEPLOYMENT:** {item.get('location') or 'ANY / CLASSIFIED'}")
                        st.markdown(f"**WIKI DOSSIER:** [OPEN SOURCE INTEL]({item['wiki_url']})")

                    with c2:
                        if chain:
                            st.markdown(f"**PREREQUISITE LINEAGE ({len(chain)} STEPS):**")
                            st.code(" → ".join(chain))
                        else:
                            st.markdown("**PREREQUISITES:** NONE (LEVEL OR IMMEDIATE ACCESS)")

                    st.markdown("**RAW TELEMETRY / OBJECTIVES:**")
                    st.code(item["document"], language="markdown")
