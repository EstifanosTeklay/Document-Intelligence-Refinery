"""
Document Intelligence Refinery — Streamlit UI
=============================================
API key loaded from .env file only — not exposed in UI.
Clean chat interface for Q&A with full provenance.
"""
import streamlit as st
import sys
import os
import json
from pathlib import Path
from collections import Counter

# ── Page config must be first ─────────────────────────────────
st.set_page_config(
    page_title="Document Intelligence Refinery",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project root + env ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0f1117; color: #e8eaf0; }
[data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #2a2f3e; }

.refinery-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem; font-weight: 600;
    color: #5b9bd5; letter-spacing: -0.5px;
}
.refinery-sub {
    font-size: 0.78rem; color: #6b7280;
    font-family: 'IBM Plex Mono', monospace;
}
.stage-card {
    background: #1a1f2e; border: 1px solid #2a2f3e;
    border-left: 3px solid #5b9bd5; border-radius: 6px;
    padding: 14px 18px; margin: 8px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
}
.stage-card.success { border-left-color: #22c55e; }
.metric-box {
    background: #1a1f2e; border: 1px solid #2a2f3e;
    border-radius: 6px; padding: 12px 16px; text-align: center;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem; font-weight: 600; color: #5b9bd5;
}
.metric-label {
    font-size: 0.72rem; color: #6b7280;
    text-transform: uppercase; letter-spacing: 1px;
}
.answer-box {
    background: #1a1f2e; border: 1px solid #2a2f3e;
    border-radius: 8px; padding: 20px 24px; margin: 10px 0;
    line-height: 1.75; font-size: 0.95rem;
}
.citation-card {
    background: #111827; border: 1px solid #1f2937;
    border-left: 3px solid #22c55e; border-radius: 6px;
    padding: 10px 14px; margin: 5px 0;
    font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace;
}
.badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.badge-a { background:#14532d; color:#86efac; }
.badge-b { background:#1e3a5f; color:#93c5fd; }
.badge-c { background:#451a03; color:#fcd34d; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# Session state
# ════════════════════════════════════════════════════════════
for key, default in {
    "profile":       None,
    "extracted":     None,
    "chunks":        None,
    "index":         None,
    "doc_id":        None,
    "pipeline_done": False,
    "chat_history":  [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="refinery-title">🔬 Refinery</div>', unsafe_allow_html=True)
    st.markdown('<div class="refinery-sub">Document Intelligence Pipeline</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Upload Document**")
    uploaded = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        data_dir = ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        pdf_path = data_dir / uploaded.name
        if not pdf_path.exists():
            with open(pdf_path, "wb") as f:
                f.write(uploaded.getbuffer())
        st.success(f"Ready: {uploaded.name}")

        if st.button("▶  Run Full Pipeline", use_container_width=True, type="primary"):
            st.session_state.pipeline_done = False
            st.session_state.chat_history  = []

            with st.spinner("Running pipeline..."):
                try:
                    from src.agents.triage import TriageAgent
                    from src.agents.extractor import ExtractionRouter
                    from src.agents.chunker import ChunkingEngine
                    from src.agents.indexer import PageIndexBuilder

                    triage   = TriageAgent()
                    profile  = triage.run(pdf_path)
                    st.session_state.profile = profile

                    router   = ExtractionRouter()
                    extracted = router.run(pdf_path, profile)
                    st.session_state.extracted = extracted

                    chunker  = ChunkingEngine()
                    chunks   = chunker.run(extracted)
                    st.session_state.chunks = chunks

                    builder  = PageIndexBuilder()
                    index    = builder.run(
                        doc_id=profile.doc_id,
                        filename=profile.filename,
                        total_pages=profile.total_pages,
                        chunks=chunks,
                    )
                    st.session_state.index        = index
                    st.session_state.doc_id       = profile.doc_id
                    st.session_state.pipeline_done = True
                    st.rerun()

                except Exception as e:
                    st.error(f"Pipeline error: {e}")

    st.markdown("---")

    # Navigation — lands on Query once pipeline is done
    st.markdown("**Navigate**")
    default_idx = 3 if st.session_state.pipeline_done else 0
    page = st.radio(
        "page",
        ["Pipeline", "Chunks", "Index", "Query"],
        index=default_idx,
        label_visibility="collapsed",
    )

    # API key status indicator — read only, no input field
    st.markdown("---")
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY", "")
    if api_key and api_key not in ("your_key_here", "your_actual_key_here"):
        st.markdown(
            "<span style='color:#22c55e;font-size:0.8rem'>● Gemini API ready</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<span style='color:#f59e0b;font-size:0.8rem'>"
            "⚠ Add GOOGLE_GEMINI_API_KEY to .env</span>",
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════════════
# PAGE 1 — Pipeline
# ════════════════════════════════════════════════════════════
if page == "Pipeline":
    st.markdown("## Pipeline Overview")

    if not st.session_state.pipeline_done:
        st.info("Upload a PDF and click **Run Full Pipeline** to get started.")
        for sid, name, desc in [
            ("Stage 1", "Triage Agent",       "Classifies origin type, layout complexity, domain and selects extraction strategy."),
            ("Stage 2", "Extraction Router",  "Runs Strategy A/B/C with confidence-gated escalation."),
            ("Stage 3", "Chunking Engine",    "Splits content into semantic LDUs following the 5 chunking rules."),
            ("Stage 4", "PageIndex Builder",  "Builds a hierarchical navigation tree with LLM-generated section summaries."),
            ("Stage 5", "Query Agent",        "Answers questions with full provenance — citations, page numbers, bounding boxes."),
        ]:
            st.markdown(f"""
            <div class="stage-card">
                <strong>{sid} — {name}</strong><br>
                <span style="color:#9ca3af">{desc}</span>
            </div>""", unsafe_allow_html=True)
    else:
        profile   = st.session_state.profile
        extracted = st.session_state.extracted
        chunks    = st.session_state.chunks
        index     = st.session_state.index
        routing   = extracted.routing_decision or {}

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, val, label in [
            (c1, str(profile.total_pages),                 "Pages"),
            (c2, str(len(chunks)),                         "Chunks"),
            (c3, str(index.total_sections),                "Sections"),
            (c4, str(len(extracted.tables)),               "Tables"),
            (c5, f"{extracted.extraction_confidence:.0%}", "Confidence"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("### Stage Results")

        origin  = getattr(profile.origin_type,               'value', str(profile.origin_type))
        layout  = getattr(profile.layout_complexity,         'value', str(profile.layout_complexity))
        domain  = getattr(profile.domain_hint,               'value', str(profile.domain_hint))
        cost    = getattr(profile.estimated_extraction_cost, 'value', str(profile.estimated_extraction_cost))

        st.markdown(f"""
        <div class="stage-card success">
            <strong>✓ Stage 1 — Triage Agent</strong><br>
            origin: <strong>{origin}</strong> &nbsp;|&nbsp;
            layout: <strong>{layout}</strong> &nbsp;|&nbsp;
            domain: <strong>{domain}</strong> &nbsp;|&nbsp;
            cost: <strong>{cost}</strong>
        </div>""", unsafe_allow_html=True)

        strategy  = extracted.strategy_used
        badge_cls = f"badge-{strategy.lower()}" if strategy in ("A","B","C") else "badge-a"
        escalated = routing.get("escalation_occurred", False)
        esc_path  = routing.get("escalation_path", strategy)
        st.markdown(f"""
        <div class="stage-card success">
            <strong>✓ Stage 2 — Extraction Router</strong><br>
            strategy: <span class="badge {badge_cls}">Strategy {strategy}</span> &nbsp;|&nbsp;
            confidence: <strong>{extracted.extraction_confidence:.2f}</strong> &nbsp;|&nbsp;
            escalation: <strong>{'Yes → ' + str(esc_path) if escalated else 'No'}</strong> &nbsp;|&nbsp;
            time: <strong>{extracted.processing_time_seconds:.1f}s</strong>
        </div>""", unsafe_allow_html=True)

        type_counts = Counter(c.chunk_type for c in chunks)
        type_str = "  |  ".join(f"{k}: {v}" for k, v in type_counts.items())
        st.markdown(f"""
        <div class="stage-card success">
            <strong>✓ Stage 3 — Chunking Engine</strong><br>
            total: <strong>{len(chunks)} chunks</strong> &nbsp;|&nbsp; {type_str}
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stage-card success">
            <strong>✓ Stage 4 — PageIndex Builder</strong><br>
            sections: <strong>{index.total_sections}</strong> &nbsp;|&nbsp;
            chunks indexed: <strong>{index.total_chunks_indexed}</strong><br>
            <em style="color:#9ca3af">{index.document_summary}</em>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stage-card" style="border-left-color:#5b9bd5">
            <strong>● Stage 5 — Query Agent</strong>
            <span style="color:#9ca3af"> → go to the Query tab to ask questions</span>
        </div>""", unsafe_allow_html=True)

        if profile.triage_notes:
            st.warning(f"Triage note: {profile.triage_notes}")


# ════════════════════════════════════════════════════════════
# PAGE 2 — Chunks
# ════════════════════════════════════════════════════════════
elif page == "Chunks":
    st.markdown("## Chunks (LDUs)")

    if not st.session_state.chunks:
        st.info("Run the pipeline first.")
    else:
        chunks = st.session_state.chunks
        col1, col2 = st.columns([2, 1])
        with col1:
            search = st.text_input("Search", placeholder="keyword...")
        with col2:
            all_types   = sorted(set(c.chunk_type for c in chunks))
            filter_type = st.selectbox("Type", ["all"] + list(all_types))

        filtered = chunks
        if search:
            filtered = [c for c in filtered if search.lower() in c.content.lower()]
        if filter_type != "all":
            filtered = [c for c in filtered if c.chunk_type == filter_type]

        st.markdown(f"Showing **{len(filtered)}** of **{len(chunks)}** chunks")
        for chunk in filtered:
            with st.expander(
                f"[{chunk.chunk_id}]  {chunk.chunk_type.upper()}  —  "
                f"p{chunk.page_refs}  —  {chunk.token_count} tokens"
            ):
                st.markdown(f"**Section:** {chunk.parent_section or '—'}")
                st.text(chunk.content)
                st.markdown(f"**Hash:** `{chunk.content_hash[:16]}...`")


# ════════════════════════════════════════════════════════════
# PAGE 3 — PageIndex
# ════════════════════════════════════════════════════════════
elif page == "Index":
    st.markdown("## PageIndex — Section Tree")

    if not st.session_state.index:
        st.info("Run the pipeline first.")
    else:
        index = st.session_state.index
        st.markdown(
            f"**{index.filename}** — {index.total_pages} pages — "
            f"{index.total_sections} sections — {index.total_chunks_indexed} chunks"
        )
        st.markdown(f"*{index.document_summary}*")
        st.markdown("---")

        for node in index.nodes:
            icon = "📂" if node.child_node_ids else "📄"
            with st.expander(
                f"{icon}  {'  ' * (node.level - 1)}{node.title}  "
                f"(p{node.page_start}–{node.page_end}, {len(node.chunk_ids)} chunks)"
            ):
                if node.summary:
                    st.markdown(f"**Summary:** {node.summary}")
                st.markdown(
                    f"**Level:** {node.level}  |  "
                    f"**Chunks:** {len(node.chunk_ids)}  |  "
                    f"**Node ID:** `{node.node_id}`"
                )
                if node.data_types_present:
                    st.markdown(f"**Data types:** {', '.join(node.data_types_present)}")
                if node.key_entities:
                    st.markdown(
                        "**Key entities:** " +
                        ", ".join(f"{e.text} ({e.entity_type})" for e in node.key_entities[:5])
                    )


# ════════════════════════════════════════════════════════════
# PAGE 4 — Query (chat)
# ════════════════════════════════════════════════════════════
elif page == "Query":
    st.markdown("## Query Agent")

    if not st.session_state.pipeline_done:
        st.info("Run the pipeline first.")
    else:
        doc_id  = st.session_state.doc_id
        api_key = os.getenv("GOOGLE_GEMINI_API_KEY", "")

        if not api_key or api_key in ("your_key_here", "your_actual_key_here"):
            st.warning(
                "No Gemini API key found. Add `GOOGLE_GEMINI_API_KEY=your_key` "
                "to your `.env` file and restart Streamlit. "
                "Showing raw excerpts as fallback for now."
            )

        # Render existing chat history
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.markdown(
                    f'<div class="answer-box">{entry["answer"]}</div>',
                    unsafe_allow_html=True
                )
                if entry["citations"]:
                    st.markdown(f"**{len(entry['citations'])} source(s):**")
                    for i, cite in enumerate(entry["citations"], 1):
                        st.markdown(f"""
                        <div class="citation-card">
                            [{i}] Page {cite['page']} &nbsp;|&nbsp;
                            Section: {cite['section'] or '—'} &nbsp;|&nbsp;
                            Chunk: <code>{cite['chunk_id']}</code><br>
                            <em style="color:#9ca3af">"{cite['excerpt'][:120]}..."</em>
                        </div>""", unsafe_allow_html=True)
                if entry.get("nodes_traversed"):
                    st.markdown(
                        f"<span style='color:#6b7280;font-size:0.75rem'>"
                        f"PageIndex nodes: {', '.join(entry['nodes_traversed'])}"
                        f"</span>",
                        unsafe_allow_html=True
                    )

        # Chat input box
        query = st.chat_input("Ask a question about the document...")

        if query:
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    try:
                        from src.agents.query_agent import QueryAgent
                        agent  = QueryAgent(top_k=5)
                        result = agent.run(query, doc_id)

                        answer    = result.answer
                        citations = [
                            {
                                "page":     c.page_number,
                                "section":  c.section_title,
                                "chunk_id": c.chunk_id,
                                "excerpt":  c.excerpt,
                            }
                            for c in result.citations
                        ]

                        st.markdown(
                            f'<div class="answer-box">{answer}</div>',
                            unsafe_allow_html=True
                        )

                        if citations:
                            st.markdown(f"**{len(citations)} source(s):**")
                            for i, cite in enumerate(citations, 1):
                                st.markdown(f"""
                                <div class="citation-card">
                                    [{i}] Page {cite['page']} &nbsp;|&nbsp;
                                    Section: {cite['section'] or '—'} &nbsp;|&nbsp;
                                    Chunk: <code>{cite['chunk_id']}</code><br>
                                    <em style="color:#9ca3af">"{cite['excerpt'][:120]}..."</em>
                                </div>""", unsafe_allow_html=True)

                        if result.pageindex_nodes_traversed:
                            st.markdown(
                                f"<span style='color:#6b7280;font-size:0.75rem'>"
                                f"PageIndex nodes: "
                                f"{', '.join(result.pageindex_nodes_traversed)}"
                                f"</span>",
                                unsafe_allow_html=True
                            )

                        # Save to history
                        st.session_state.chat_history.append({
                            "question":        query,
                            "answer":          answer,
                            "citations":       citations,
                            "nodes_traversed": result.pageindex_nodes_traversed,
                        })

                    except Exception as e:
                        st.error(f"Query error: {e}")
