import sys
import os
from pathlib import Path
from collections import Counter
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.embedding import get_embedding_model
from src.vector_store import get_or_create_store, get_all_documents
from src.explorer import full_pipeline_retrieve
from src.prompt import format_context

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(
    page_title="NEXUS · AI Governance Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

TOP_K = 8  # hidden default, no UI setting needed

# ── NEXUS × HERITAGE PALETTE CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
  /* Primary palette */
  --accent-red:      #CC0001;
  --accent-red-glow: rgba(204,0,1,0.14);
  --accent-blue:     #003893;
  --accent-blue-mid: #1A5CC8;
  --accent-blue-glow:rgba(0,56,147,0.13);
  --accent-gold:     #FFCC00;
  --accent-gold-dim: rgba(255,204,0,0.13);

  /* Light theme background */
  --bg-base:     #F4F1EC;
  --bg-surface:  #FFFFFF;
  --bg-elevated: #FBF8F3;
  --bg-hover:    #F0EDE6;
  --border:      rgba(0,56,147,0.09);
  --border-red:  rgba(204,0,1,0.2);
  --border-gold: rgba(255,204,0,0.35);

  /* Text */
  --text-primary:   #0F1624;
  --text-secondary: #475270;
  --text-muted:     #8990AA;

  /* Border radius */
  --radius-xl: 22px;
  --radius-lg: 16px;
  --radius-md: 12px;
  --radius-sm: 8px;
}



/* Base */
html, body, .stApp {
  background: linear-gradient(135deg, var(--bg-base), var(--bg-surface));
  background-size: 200% 200%;
  animation: gradientShift 12s ease infinite;
  font-family: 'Inter', -apple-system, sans-serif !important;
  color: var(--text-primary) !important;
}
.block-container { padding: 1.5rem 2.5rem 3rem !important; max-width: 1440px !important; }

/* ── Hide chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stSidebarCollapsedControl"],
button[kind="header"] { visibility: hidden !important; display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--accent-red); border-radius: 10px; opacity: 0.5; }

/* ── Animations ── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGreen {
  0%,100% { box-shadow: 0 0 0 0 rgba(0,185,75,0.5); }
  50%      { box-shadow: 0 0 0 5px rgba(0,185,75,0); }
}
@keyframes glowPulse {
  0%,100% { opacity: 0.6; }
  50%      { opacity: 1; }
}

/* ── Override containers ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: 0 2px 10px rgba(0,56,147,0.04) !important;
  transition: box-shadow 0.25s, border-color 0.25s, transform 0.2s;
  animation: fadeUp 0.4s ease-out both;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  box-shadow: 0 8px 28px rgba(0,56,147,0.08) !important;
  border-color: rgba(0,56,147,0.16) !important;
  transform: translateY(-2px) !important;
}

/* ── Headings ── */
h1,h2,h3,h4 { color: var(--accent-blue) !important; font-weight: 800 !important; letter-spacing: -0.3px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-surface);
  border-radius: var(--radius-sm);
  padding: 4px; gap: 4px;
  border: 1px solid var(--border);
  width: fit-content;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: 6px !important;
  color: var(--text-muted) !important;
  font-weight: 600 !important;
  font-size: 0.84rem !important;
  padding: 8px 20px !important;
  border: none !important;
  transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
  background: var(--accent-blue) !important;
  color: white !important;
}
.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── Buttons ── */
.stButton > button {
  background: var(--bg-elevated) !important;
  color: var(--text-secondary) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  background: var(--accent-blue) !important;
  color: white !important;
  border-color: var(--accent-blue) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px var(--accent-blue-glow) !important;
}

/* Chip buttons */
.chip-btn .stButton > button {
  background: white !important;
  border: 1.5px solid var(--border-red) !important;
  color: var(--accent-red) !important;
  border-radius: 100px !important;
  font-size: 0.78rem !important;
  padding: 5px 14px !important;
}
.chip-btn .stButton > button:hover {
  background: var(--accent-red) !important;
  color: white !important;
  border-color: var(--accent-red) !important;
  box-shadow: 0 4px 14px var(--accent-red-glow) !important;
}

/* Danger/clear button */
.danger-btn .stButton > button {
  background: white !important;
  border: 1.5px solid var(--border-red) !important;
  color: var(--accent-red) !important;
  border-radius: var(--radius-sm) !important;
}
.danger-btn .stButton > button:hover {
  background: var(--accent-red) !important;
  color: white !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  padding: 1rem 1.25rem !important;
  margin-bottom: 10px !important;
  box-shadow: 0 1px 5px rgba(0,56,147,0.03) !important;
  animation: fadeUp 0.3s ease-out both;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
  border-left: 4px solid var(--accent-red) !important;
  background: linear-gradient(135deg,rgba(204,0,1,0.025),white) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  border-left: 4px solid var(--accent-blue) !important;
  background: linear-gradient(135deg,rgba(0,56,147,0.025),white) !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
  color: var(--text-primary) !important;
  font-size: 0.92rem !important;
  line-height: 1.75 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
  background: var(--bg-surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: 0 2px 12px rgba(0,56,147,0.05) !important;
  transition: border-color 0.3s, box-shadow 0.3s !important;
  margin-top: 1rem;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent-red) !important;
  box-shadow: 0 0 0 3px var(--accent-red-glow), 0 4px 20px rgba(204,0,1,0.07) !important;
}
[data-testid="stChatInput"] textarea {
  color: var(--text-primary) !important;
  background: transparent !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.92rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }
[data-testid="stChatInput"] button { background: var(--accent-red) !important; border-radius: 10px !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}
[data-testid="stExpander"] summary { color: var(--text-secondary) !important; font-size: 0.82rem !important; }

/* ── Markdown ── */
.stMarkdown p { color: var(--text-secondary) !important; }
.stMarkdown strong { color: var(--text-primary) !important; }
.stMarkdown code {
  background: var(--accent-gold-dim) !important;
  color: #7A5C00 !important;
  border-radius: 4px; padding: 1px 6px; font-size: 0.83em;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ════════ CUSTOM COMPONENTS ════════ */

/* Top stripe: 3 flag colours */
.heritage-stripe {
  height: 5px;
  border-radius: 100px;
  background: linear-gradient(90deg, var(--accent-red) 0%, var(--accent-red) 33%, var(--accent-gold) 33%, var(--accent-gold) 66%, var(--accent-blue) 66%, var(--accent-blue) 100%);
  margin-bottom: 1.5rem;
  opacity: 0.75;
  animation: fadeUp 0.2s ease-out both;
}

/* NEXUS Header */
.nexus-header {
  background: linear-gradient(135deg, var(--accent-blue) 0%, #001F5C 100%);
  border-radius: var(--radius-xl);
  padding: 1.75rem 2rem;
  display: flex;
  align-items: center;
  gap: 1.4rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 10px 40px rgba(0,56,147,0.22);
  animation: fadeUp 0.4s ease-out both;
}
/* Gold shimmer line top */
.nexus-header::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent 0%, var(--accent-gold) 40%, var(--accent-gold) 60%, transparent 100%);
}
/* Decorative radial glow */
.nexus-header::after {
  content: '';
  position: absolute; top: -80px; right: -80px;
  width: 280px; height: 280px;
  background: radial-gradient(circle, rgba(255,204,0,0.07) 0%, transparent 65%);
  pointer-events: none;
}

.nexus-logo-wrap {
  position: relative; flex-shrink: 0;
}
.nexus-logo {
  width: 58px; height: 58px;
  background: linear-gradient(135deg, var(--accent-red), #8B0010);
  border-radius: 16px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.75rem;
  box-shadow: 0 6px 24px rgba(204,0,1,0.4);
}
/* gold ring around logo */
.nexus-logo-wrap::after {
  content: '';
  position: absolute; inset: -3px;
  border-radius: 19px;
  border: 1.5px solid rgba(255,204,0,0.35);
}

.nexus-title { font-size: 1.75rem; font-weight: 900; color: white !important; margin: 0; letter-spacing: -0.6px; }
.nexus-sub   { font-size: 0.8rem; color: rgba(255,255,255,0.85); margin: 4px 0 0 0; font-weight: 500; }

.nexus-badges { margin-left: auto; display: flex; flex-direction: column; align-items: flex-end; gap: 8px; flex-shrink: 0; }

.badge-online {
  display: flex; align-items: center; gap: 7px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 100px; padding: 6px 14px;
}
.badge-dot { width: 8px; height: 8px; background: #00C850; border-radius: 50%; animation: pulseGreen 2s infinite; }
.badge-text { font-size: 0.74rem; font-weight: 700; color: white; letter-spacing: 0.05em; }

.badge-model {
  background: var(--accent-gold);
  border-radius: 100px; padding: 4px 12px;
}
.badge-model-text { font-size: 0.7rem; font-weight: 800; color: #3D2B00; letter-spacing: 0.04em; }

/* Stat card */
.stat-card {
  background: var(--bg-surface);
  border: 1.5px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.2rem 1rem;
  text-align: center;
  position: relative; overflow: hidden;
  transition: all 0.25s;
  animation: fadeUp 0.5s ease-out both;
  box-shadow: 0 2px 8px rgba(0,56,147,0.04);
}
.stat-card:hover { transform: translateY(-4px); box-shadow: 0 10px 28px rgba(0,56,147,0.1); border-color: var(--accent-blue); }
.stat-top-bar { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 0; opacity: 0; transition: opacity 0.25s; }
.stat-card:hover .stat-top-bar { opacity: 1; }
.red-bar  { background: var(--accent-red); }
.blue-bar { background: var(--accent-blue); }
.gold-bar { background: var(--accent-gold); }
.stat-icon  { font-size: 1.75rem; margin-bottom: 0.35rem; display: block; }
.stat-value { font-size: 2.1rem; font-weight: 900; color: var(--accent-blue) !important; line-height: 1; }
.stat-label { font-size: 0.68rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 5px; }

/* Pipeline card */
.pipeline-card {
  background: var(--bg-surface);
  border: 1.5px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.15rem 1.4rem;
  box-shadow: 0 2px 8px rgba(0,56,147,0.04);
  animation: fadeUp 0.55s ease-out both;
}
.pl-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
.pl-label { font-size: 0.68rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
.pl-tag {
  font-size: 0.68rem; font-weight: 700;
  background: rgba(0,56,147,0.07);
  color: var(--accent-blue-mid);
  border: 1px solid rgba(0,56,147,0.18);
  border-radius: 100px; padding: 2px 10px;
}
.pl-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.pl-step { border-radius: 6px; padding: 4px 10px; font-size: 0.72rem; font-weight: 700; }
.pl-step.red  { background: rgba(204,0,1,0.06);   border: 1px solid rgba(204,0,1,0.18);   color: var(--accent-red); }
.pl-step.blue { background: rgba(0,56,147,0.06);  border: 1px solid rgba(0,56,147,0.18);  color: var(--accent-blue); }
.pl-step.gold { background: rgba(255,204,0,0.14); border: 1px solid rgba(255,204,0,0.4);  color: #7A5C00; }
.pl-arrow { color: var(--text-muted); font-size: 0.8rem; }

/* Section label */
.section-label { font-size: 0.68rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }

/* Tri-colour divider */
.tri-divider {
  height: 3px; border-radius: 100px; margin: 1.25rem 0;
  background: linear-gradient(90deg, var(--accent-red), var(--accent-gold), var(--accent-blue));
  opacity: 0.35;
}

/* Empty state */
.empty-state {
  text-align: center; padding: 4.5rem 2rem;
}
.empty-icon  { font-size: 3.5rem; display: block; margin-bottom: 1rem; opacity: 0.3; }
.empty-title { font-size: 1.05rem; font-weight: 700; color: var(--text-secondary); margin-bottom: 5px; }
.empty-sub   { font-size: 0.83rem; color: var(--text-muted); }

/* Insights stats grid */
.ig-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.ig-card { background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 12px; }
.ig-val  { font-size: 1.5rem; font-weight: 900; color: var(--accent-blue); }
.ig-val.red { color: var(--accent-red); }
.ig-lbl  { font-size: 0.65rem; color: var(--text-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚡ Loading embedding engine...")
def load_embedding_model():
    return get_embedding_model()

@st.cache_resource(show_spinner="🗄️ Connecting to vector store...")
def load_store(_embedding_model):
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    return get_or_create_store(chroma_dir, _embedding_model)

@st.cache_resource(show_spinner="🤖 Initialising NEXUS agent...")
def load_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-flash",
        temperature=0,
        streaming=True,
        model_kwargs={"extra_body": {"reasoning_effort": "high"}}
    )

# ── Load ──────────────────────────────────────────────────────────────────────
embedding_model = load_embedding_model()
store = load_store(embedding_model)
llm = load_llm()

if llm is None:
    st.error("⚠️ DEEPSEEK_API_KEY not set in .env — agent offline.")
    st.stop()

all_docs = get_all_documents(store)
if not all_docs:
    st.warning("⚠️ ChromaDB is empty. Run `python src/main.py --ingest` first.")
    st.stop()

# ── Agent ─────────────────────────────────────────────────────────────────────
@tool
def search_governance_documents(query: str) -> str:
    """Search the AI Governance knowledge base for policies, frameworks, and guidance.
    Input: a clear, specific search query string."""
    docs = full_pipeline_retrieve(query, store, all_docs, embedding_model, TOP_K)
    st.session_state.latest_docs = docs
    if not docs:
        return "No relevant documents found for this query."
    return format_context(docs)

tools = [search_governance_documents]
system_prompt = (
    "You are NEXUS, an elite autonomous AI agent specialising in AI Governance. "
    "You have access to a rich knowledge base of AI governance documents. "
    "Use the search_governance_documents tool to retrieve information before answering substantive questions. "
    "Always cite source documents when using retrieved facts. "
    "Be concise, precise, and insightful. For greetings or trivial questions, respond directly without searching."
)
agent_executor = create_react_agent(llm, tools)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prompt_action" not in st.session_state:
    st.session_state.prompt_action = None
if "latest_docs" not in st.session_state:
    st.session_state.latest_docs = None

# ─────────────────────────────────────────────────────────────────────────────
#  HERITAGE STRIPE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="heritage-stripe"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nexus-header">
  <div class="nexus-logo-wrap">
    <div class="nexus-logo">⚡</div>
  </div>
  <div>
    <p class="nexus-title">NEXUS</p>
    <p class="nexus-sub">Autonomous AI Governance Agent · Hybrid RAG · LangGraph ReAct</p>
  </div>
  <div class="nexus-badges">
    <div class="badge-online">
      <div class="badge-dot"></div>
      <span class="badge-text">AGENT ONLINE</span>
    </div>
    <div class="badge-model">
      <span class="badge-model-text">⚡ DEEPSEEK V4 FLASH</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  BENTO STATS
# ─────────────────────────────────────────────────────────────────────────────
num_queries = len([m for m in st.session_state.messages if m["role"] == "user"])
c1, c2, c3, c4 = st.columns([1, 1, 1, 3])

with c1:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-top-bar red-bar"></div>
      <span class="stat-icon">📄</span>
      <div class="stat-value">646</div>
      <div class="stat-label">Documents</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-top-bar blue-bar"></div>
      <span class="stat-icon">🧩</span>
      <div class="stat-value">{len(all_docs):,}</div>
      <div class="stat-label">Chunks</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-top-bar gold-bar"></div>
      <span class="stat-icon">💬</span>
      <div class="stat-value">{num_queries}</div>
      <div class="stat-label">Queries</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="pipeline-card">
      <div class="pl-header">
        <span class="pl-label">🔧 Active Pipeline</span>
        <span class="pl-tag">LangGraph ReAct</span>
      </div>
      <div class="pl-row">
        <span class="pl-step blue">🔍 Dense Embed</span>
        <span class="pl-arrow">+</span>
        <span class="pl-step blue">📊 BM25 Sparse</span>
        <span class="pl-arrow">→</span>
        <span class="pl-step gold">🔀 MMR Fusion</span>
        <span class="pl-arrow">→</span>
        <span class="pl-step red">🤖 DeepSeek V4</span>
        <span class="pl-arrow">→</span>
        <span class="pl-step red">✅ Answer</span>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_insights = st.tabs(["⚡  Chat with NEXUS", "📊  Knowledge Insights"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CHAT TAB
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_chat:
    # Top bar: quick prompts + clear button
    left_col, right_col = st.columns([5, 1])
    with left_col:
        st.markdown('<div class="section-label">Quick Prompts</div>', unsafe_allow_html=True)
        qc1, qc2, qc3, qc4, qc5 = st.columns(5)
        quick_prompts = [
            ("👋 Say Hi",        "Hi, who are you and what can you help me with?"),
            ("⚖️ AI Risks",      "What are the policies on AI risks and who enforces them?"),
            ("🏛️ Agencies",     "What organisations or agencies are referenced in the governance documents?"),
            ("📋 Requirements",  "What are the key AI governance requirements for organisations?"),
            ("🌐 Strategy",      "What is the national AI strategy and its framework?"),
        ]
        for col, (label, prompt_text) in zip([qc1, qc2, qc3, qc4, qc5], quick_prompts):
            with col:
                st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                if st.button(label, use_container_width=True, key=f"qp_{label}"):
                    st.session_state.prompt_action = prompt_text
                st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-label">&nbsp;</div>', unsafe_allow_html=True)
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="tri-divider"></div>', unsafe_allow_html=True)

    # Empty state
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
          <span class="empty-icon">⚡</span>
          <div class="empty-title">NEXUS is ready and online.</div>
          <div class="empty-sub">Ask anything about AI Governance, or pick a quick prompt above.</div>
        </div>
        """, unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.messages:
        avatar = "⚡" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("docs"):
                with st.expander("📚 View Retrieved Documents"):
                    for i, d in enumerate(msg["docs"], 1):
                        source = os.path.basename(d.metadata.get("source", "Unknown"))
                        page = d.metadata.get("page", "N/A")
                        st.markdown(f"**[{i}] {source} (Page {page})**")
                        content = d.page_content
                        st.caption(content[:400] + "..." if len(content) > 400 else content)
                        if i < len(msg["docs"]):
                            st.divider()

    # Chat input
    user_input = st.chat_input("Ask NEXUS anything about AI Governance...")
    prompt = user_input if user_input else st.session_state.prompt_action

    if prompt:
        st.session_state.prompt_action = None
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="⚡"):
            cb_container = st.container()
            st_callback = StreamlitCallbackHandler(
                cb_container,
                expand_new_thoughts=True,
                collapse_completed_thoughts=True,
                max_thought_containers=5,
            )
            chat_history = []
            for m in st.session_state.messages[:-1]:
                if m["role"] == "user":
                    chat_history.append(("human", m["content"]))
                elif m["role"] == "assistant":
                    chat_history.append(("ai", m["content"]))

            try:
                st.session_state.latest_docs = None
                with st.spinner("NEXUS is thinking..."):
                    response = agent_executor.invoke(
                        {"messages": [("system", system_prompt)] + chat_history + [("user", prompt)]},
                        config={"callbacks": [st_callback]}
                    )
                full_answer = response["messages"][-1].content
                st.markdown(full_answer)
                
                docs_to_save = st.session_state.latest_docs
                if docs_to_save:
                    with st.expander("📚 View Retrieved Documents"):
                        for i, d in enumerate(docs_to_save, 1):
                            source = os.path.basename(d.metadata.get("source", "Unknown"))
                            page = d.metadata.get("page", "N/A")
                            st.markdown(f"**[{i}] {source} (Page {page})**")
                            content = d.page_content
                            st.caption(content[:400] + "..." if len(content) > 400 else content)
                            if i < len(docs_to_save):
                                st.divider()

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_answer,
                    "docs": docs_to_save
                })
                st.rerun()

            except Exception as e:
                st.error(f"Agent error: {e}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INSIGHTS TAB
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_insights:
    st.markdown("#### 📚 Knowledge Base Distribution")
    st.markdown('<p style="color:var(--text-muted); font-size:0.84rem; margin-top:-8px;">Chunk count by indexed document</p>', unsafe_allow_html=True)

    sources_list = [os.path.basename(d.metadata.get("source", "Unknown")) for d in all_docs]
    counts = Counter(sources_list)
    df_counts = (
        pd.DataFrame(counts.items(), columns=["Document", "Chunks"])
        .sort_values("Chunks", ascending=False)
        .reset_index(drop=True)
    )

    st.bar_chart(df_counts.head(15).set_index("Document"), color="#CC0001", height=360)
    st.markdown('<div class="tri-divider"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1.5, 1])
    with col_a:
        with st.expander("📋 Full document list"):
            st.dataframe(df_counts, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown(f"""
        <div class="ig-grid">
          <div class="ig-card">
            <div class="ig-val">{len(df_counts)}</div>
            <div class="ig-lbl">Unique Docs</div>
          </div>
          <div class="ig-card">
            <div class="ig-val">{df_counts['Chunks'].mean():.0f}</div>
            <div class="ig-lbl">Avg Chunks</div>
          </div>
          <div class="ig-card">
            <div class="ig-val red">{df_counts['Chunks'].max()}</div>
            <div class="ig-lbl">Largest Doc</div>
          </div>
          <div class="ig-card">
            <div class="ig-val">{len(all_docs):,}</div>
            <div class="ig-lbl">Total Chunks</div>
          </div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="tri-divider" style="margin-top:2rem; opacity:0.2;"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:var(--text-muted); font-size:0.73rem; padding-bottom:1rem;">
  ⚡ NEXUS &nbsp;·&nbsp; Powered by LangGraph + DeepSeek + ChromaDB &nbsp;·&nbsp; Hybrid RAG Pipeline
</div>
""", unsafe_allow_html=True)
