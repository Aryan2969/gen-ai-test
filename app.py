"""
app.py — Interactive Web UI for GE Gas Turbine RAG Maintenance Assistant.
Run:
    streamlit run app.py
"""

import json
from pathlib import Path
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="GE Gas Turbine RAG Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(8, 12, 24) 90.2%);
        color: #f8fafc;
    }
    
    .hero-header {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .sub-hero {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 1.2rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    
    .q-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
    }
    
    .badge-found {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    
    .badge-notfound {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    
    .page-tag {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# Load results.json if exists
@st.cache_data
def load_results():
    p = Path("results.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except:
            return []
    return []


# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/ff/General_Electric_logo.svg", width=60)
    st.markdown("### ⚙️ System Specifications")
    st.markdown("""
    - **Document**: `GER-3620-P.pdf`
    - **Total Pages**: `61 Pages`
    - **Embedding**: `all-MiniLM-L6-v2`
    - **Vector Store**: `ChromaDB (Local)`
    - **Base LLM**: `google/flan-t5-base`
    - **Zero Hallucination**: Guardrail active
    """)
    st.divider()
    st.markdown("👨‍💻 **Developer**: Aryan Singh Rana")
    st.markdown("⚡ *RAG System Evaluation Suite*")

# Main Header
st.markdown('<div class="hero-header">GE Gas Turbine Maintenance RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-hero">Intelligent Document Retrieval & Question-Answering for Heavy-Duty Gas Turbines</div>', unsafe_allow_html=True)

# Metric Row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <div class="stat-card">
        <span style="color: #94a3b8; font-size: 0.85rem;">Document</span>
        <h3 style="margin: 4px 0 0; color: #38bdf8;">GER-3620 Rev P</h3>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="stat-card">
        <span style="color: #94a3b8; font-size: 0.85rem;">Vector DB</span>
        <h3 style="margin: 4px 0 0; color: #a855f7;">ChromaDB (242)</h3>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="stat-card">
        <span style="color: #94a3b8; font-size: 0.85rem;">Evaluation Tests</span>
        <h3 style="margin: 4px 0 0; color: #34d399;">5 Official Questions</h3>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="stat-card">
        <span style="color: #94a3b8; font-size: 0.85rem;">Accuracy & Hallucination</span>
        <h3 style="margin: 4px 0 0; color: #f59e0b;">100% Guarded</h3>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

tabs = st.tabs(["📋 Official 5 Questions", "💬 Ask Custom Question", "📊 Raw Results JSON"])

results = load_results()

with tabs[0]:
    st.subheader("Benchmark Evaluation Questions")
    st.write("Answers extracted directly with verifiable page citations:")

    if results:
        for item in results:
            q_num = item.get("q_num", "")
            q = item.get("question", "")
            ans = item.get("answer", "")
            pages = item.get("source_pages", [])
            is_found = "Not found" not in ans

            badge_html = (
                '<span class="badge-found">✅ Document Answer</span>'
                if is_found
                else '<span class="badge-notfound">❌ Out of Scope</span>'
            )

            pages_html = "".join([f'<span class="page-tag">Page {p}</span>' for p in pages])

            st.markdown(f"""
            <div class="q-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h4 style="margin: 0; color: #e2e8f0;">Q{q_num}: {q}</h4>
                    {badge_html}
                </div>
                <p style="color: #cbd5e1; font-size: 1.05rem; line-height: 1.6; margin: 12px 0;">{ans}</p>
                <div style="margin-top: 10px; display: flex; align-items: center;">
                    <span style="color: #64748b; font-size: 0.9rem; margin-right: 10px;">Sources:</span>
                    {pages_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Run `python3 evaluate.py` to populate results.")

with tabs[1]:
    st.subheader("Interactive Query Interface")
    user_q = st.text_input("Enter your technical gas turbine question:", placeholder="e.g. What are the key elements of combustion inspection?")
    if st.button("Submit Question", type="primary"):
        if user_q.strip():
            with st.spinner("Retrieving from ChromaDB and reasoning..."):
                try:
                    from main import load_llm
                    from rag import build_rag_chain, answer_question
                    llm = load_llm()
                    chain, retriever = build_rag_chain(llm)
                    res = answer_question(chain, retriever, user_q)
                    st.success("Answer Generated:")
                    st.write(f"**Answer**: {res['answer']}")
                    st.write(f"**Source Pages**: {res['pages']}")
                except Exception as e:
                    st.error(f"Error executing pipeline: {e}")
        else:
            st.warning("Please type a question.")

with tabs[2]:
    st.subheader("Raw results.json Preview")
    if results:
        st.json(results)
    else:
        st.warning("No results.json found.")
