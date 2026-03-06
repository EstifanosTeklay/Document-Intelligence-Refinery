"""
Query Agent — Stage 5 of the Document Intelligence Refinery

Implements three explicit retrieval tools and a router that selects
the right tool based on query type:

  Tool 1 — pageindex_navigate
      Keyword search over PageIndex section titles and summaries.
      Best for: "what does section X say", "summarise Y", navigation queries.

  Tool 2 — semantic_search
      TF-IDF cosine similarity search over the VectorStore.
      Best for: open-ended factual questions, concept retrieval.

  Tool 3 — structured_query
      SQL query over the FactTable for numerical/financial facts.
      Best for: "how much", "what percentage", "total revenue", numbers.

Router logic:
  - Query contains number words / financial terms → structured_query
  - Query contains section/chapter navigation terms → pageindex_navigate
  - Default → semantic_search
  - Results are merged and deduplicated before passing to Gemini.

Every answer returns a ProvenanceChain with tool-specific provenance.
"""
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Optional

from src.models.ldu import LDU
from src.models.pageindex import PageIndex, PageIndexNode
from src.models.provenance import ProvenanceChain, SourceCitation
from src.models.extracted_document import BoundingBox
from src.utils.config import config


# ════════════════════════════════════════════════════════════
# Tool result dataclass
# ════════════════════════════════════════════════════════════

class ToolResult:
    """Holds the output of a single tool invocation."""
    def __init__(self, tool_name: str, chunks: list[LDU], facts: list[dict] = None):
        self.tool_name = tool_name
        self.chunks    = chunks or []
        self.facts     = facts  or []


# ════════════════════════════════════════════════════════════
# Query classifier
# ════════════════════════════════════════════════════════════

# Keywords that indicate a numerical/financial query → structured_query
NUMERICAL_SIGNALS = re.compile(
    r"\b(how much|how many|total|sum|revenue|profit|loss|cost|budget|"
    r"percentage|percent|ratio|expenditure|income|salary|payment|amount|"
    r"figure|number|count|average|rate|growth|decline|increase|decrease|"
    r"billion|million|thousand|etb|usd|\$|%)\b",
    re.IGNORECASE,
)

# Keywords that indicate section navigation → pageindex_navigate
NAVIGATION_SIGNALS = re.compile(
    r"\b(section|chapter|part|appendix|introduction|conclusion|summary|"
    r"overview|background|methodology|findings|recommendation|what does|"
    r"what is covered|describe|explain)\b",
    re.IGNORECASE,
)


def classify_query(query: str) -> str:
    """
    Route query to the most appropriate tool.

    Returns one of: 'structured_query', 'pageindex_navigate', 'semantic_search'
    """
    if NUMERICAL_SIGNALS.search(query):
        return "structured_query"
    if NAVIGATION_SIGNALS.search(query):
        return "pageindex_navigate"
    return "semantic_search"


# ════════════════════════════════════════════════════════════
# Query Agent
# ════════════════════════════════════════════════════════════

class QueryAgent:
    """
    Answers natural language questions using three retrieval tools
    with automatic routing based on query classification.

    Tools:
        pageindex_navigate  — section tree traversal
        semantic_search     — TF-IDF vector similarity
        structured_query    — SQL over FactTable

    Usage:
        agent = QueryAgent()
        result = agent.run("What was the total revenue in 2023?", "annual_report")
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    # ----------------------------------------------------------
    # Public entry point
    # ----------------------------------------------------------

    def run(self, query: str, doc_id: str) -> ProvenanceChain:
        """
        Route query to the best tool(s), retrieve context,
        generate answer with Gemini, return ProvenanceChain.
        """
        index  = self._load_index(doc_id)
        chunks = self._load_chunks(doc_id)

        if not chunks:
            return self._empty_response(query, "No chunks found for this document.")

        # Route and invoke tools
        primary_tool = classify_query(query)
        tool_results = self._invoke_tools(query, doc_id, primary_tool, index, chunks)

        # Merge chunks from all tools (deduplicated)
        top_chunks      = self._merge_chunks(tool_results, chunks)
        all_facts        = [f for tr in tool_results for f in tr.facts]
        nodes_traversed  = self._get_traversed_nodes(tool_results, index)
        tools_used       = [tr.tool_name for tr in tool_results]

        # Build context and call Gemini
        context = self._build_context(top_chunks, all_facts)
        answer  = self._call_gemini(query, context)

        # Build ProvenanceChain with tool-specific metadata
        citations = self._build_citations(top_chunks, index)
        chain = ProvenanceChain(
            query=query,
            answer=answer,
            citations=citations,
            is_verified=len(citations) > 0,
            verified_chunk_ids=[c.chunk_id for c in top_chunks],
            pageindex_nodes_traversed=nodes_traversed,
        )
        chain.set_primary_from_citations()

        # Attach tool routing info as extra metadata on the chain
        chain.__dict__["tools_used"]    = tools_used
        chain.__dict__["primary_tool"]  = primary_tool
        chain.__dict__["facts_used"]    = len(all_facts)

        return chain

    # ----------------------------------------------------------
    # Tool invocation and routing
    # ----------------------------------------------------------

    def _invoke_tools(
        self,
        query:        str,
        doc_id:       str,
        primary_tool: str,
        index:        PageIndex,
        chunks:       list[LDU],
    ) -> list[ToolResult]:
        """
        Invoke primary tool and one fallback tool.
        Always returns at least one ToolResult with chunks.
        """
        results = []

        # ── Tool 1: pageindex_navigate ──────────────────────
        if primary_tool == "pageindex_navigate":
            r = self.tool_pageindex_navigate(query, index, chunks)
            results.append(r)
            # Supplement with semantic search if few results
            if len(r.chunks) < 3:
                results.append(self.tool_semantic_search(query, doc_id, chunks))

        # ── Tool 2: semantic_search ─────────────────────────
        elif primary_tool == "semantic_search":
            results.append(self.tool_semantic_search(query, doc_id, chunks))
            # Supplement with pageindex navigation
            nav = self.tool_pageindex_navigate(query, index, chunks)
            if nav.chunks:
                results.append(nav)

        # ── Tool 3: structured_query ────────────────────────
        elif primary_tool == "structured_query":
            results.append(self.tool_structured_query(query, doc_id))
            # Always supplement with semantic search for context
            results.append(self.tool_semantic_search(query, doc_id, chunks))

        # Fallback — if all tools returned nothing, use first k chunks
        if not any(r.chunks for r in results):
            results.append(ToolResult("fallback", chunks[:self.top_k]))

        return results

    # ----------------------------------------------------------
    # Tool 1 — pageindex_navigate
    # ----------------------------------------------------------

    def tool_pageindex_navigate(
        self, query: str, index: PageIndex, all_chunks: list[LDU]
    ) -> ToolResult:
        """
        Navigate the PageIndex by keyword overlap to find relevant sections,
        then return chunks that belong to those sections.

        Provenance: pageindex_nodes_traversed records every node visited.
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        scored = []

        for node in index.nodes:
            node_text  = (node.title + " " + node.summary).lower()
            node_words = set(re.findall(r"\w+", node_text))
            overlap    = len(query_words & node_words)
            if overlap > 0:
                scored.append((overlap, node))

        scored.sort(key=lambda x: -x[0])
        top_nodes = [n for _, n in scored[:3]]

        if not top_nodes:
            return ToolResult("pageindex_navigate", [])

        # Collect chunk IDs from candidate nodes
        candidate_ids = {cid for node in top_nodes for cid in node.chunk_ids}
        candidate_chunks = [c for c in all_chunks if c.chunk_id in candidate_ids]

        # Score candidate chunks by keyword overlap
        scored_chunks = []
        for chunk in candidate_chunks:
            chunk_words = set(re.findall(r"\w+", chunk.content.lower()))
            score = len(query_words & chunk_words)
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda x: -x[0])
        top = [c for _, c in scored_chunks[:self.top_k]]

        return ToolResult("pageindex_navigate", top)

    # ----------------------------------------------------------
    # Tool 2 — semantic_search
    # ----------------------------------------------------------

    def tool_semantic_search(
        self, query: str, doc_id: str, fallback_chunks: list[LDU]
    ) -> ToolResult:
        """
        TF-IDF cosine similarity search over the VectorStore.
        Falls back to keyword scoring over in-memory chunks if store unavailable.

        Provenance: chunk_id, content_hash, page_refs, parent_section all returned.
        """
        try:
            from src.storage.vector_store import VectorStore
            vs      = VectorStore()
            results = vs.query(query, doc_id=doc_id, top_k=self.top_k)

            if not results:
                raise ValueError("Empty results from vector store")

            # Map result chunk_ids back to LDU objects
            chunk_map   = {c.chunk_id: c for c in fallback_chunks}
            top_chunks  = []
            for r in results:
                chunk = chunk_map.get(r["chunk_id"])
                if chunk:
                    top_chunks.append(chunk)

            return ToolResult("semantic_search", top_chunks)

        except Exception:
            # Fallback: keyword scoring over in-memory chunks
            query_words = set(re.findall(r"\w+", query.lower()))
            scored = []
            for chunk in fallback_chunks:
                words = set(re.findall(r"\w+", chunk.content.lower()))
                score = len(query_words & words)
                if score > 0:
                    scored.append((score, chunk))
            scored.sort(key=lambda x: -x[0])
            return ToolResult("semantic_search", [c for _, c in scored[:self.top_k]])

    # ----------------------------------------------------------
    # Tool 3 — structured_query
    # ----------------------------------------------------------

    def tool_structured_query(self, query: str, doc_id: str) -> ToolResult:
        """
        SQL query over the FactTable for numerical/financial facts.
        Detects fact type from query terms and runs targeted SQL retrieval.

        Provenance: each fact carries chunk_id and content_hash back to source.
        Returns facts in ToolResult.facts; no chunks (facts are context).
        """
        try:
            from src.storage.fact_table import FactTable
            ft = FactTable()

            # Detect what type of fact to query
            query_lower = query.lower()
            if any(w in query_lower for w in ["revenue", "profit", "cost", "etb", "usd", "$", "payment", "salary", "budget"]):
                fact_type = "CURRENCY"
            elif any(w in query_lower for w in ["percent", "%", "rate", "ratio", "growth"]):
                fact_type = "PERCENTAGE"
            elif any(w in query_lower for w in ["employees", "staff", "branches", "members", "count", "how many"]):
                fact_type = "COUNT"
            else:
                fact_type = None  # search all types

            # Extract year from query if present
            year_match = re.search(r"\b(20\d{2})\b", query)
            year = year_match.group(1) if year_match else None

            facts = ft.query_facts(
                doc_id=doc_id,
                fact_type=fact_type,
                year=year,
                limit=self.top_k * 2,
            )

            return ToolResult("structured_query", [], facts=facts)

        except Exception:
            return ToolResult("structured_query", [], facts=[])

    # ----------------------------------------------------------
    # Merge and deduplicate chunks from multiple tools
    # ----------------------------------------------------------

    def _merge_chunks(
        self, tool_results: list[ToolResult], all_chunks: list[LDU]
    ) -> list[LDU]:
        """Deduplicate chunks from multiple tools, preserving order."""
        seen   = set()
        merged = []
        for tr in tool_results:
            for chunk in tr.chunks:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    merged.append(chunk)
        return merged[:self.top_k]

    def _get_traversed_nodes(
        self, tool_results: list[ToolResult], index: PageIndex
    ) -> list[str]:
        """Collect PageIndex node IDs visited by pageindex_navigate tool."""
        all_chunk_ids = {
            c.chunk_id
            for tr in tool_results
            for c in tr.chunks
        }
        return [
            node.node_id
            for node in index.nodes
            if any(cid in node.chunk_ids for cid in all_chunk_ids)
        ]

    # ----------------------------------------------------------
    # Context builder
    # ----------------------------------------------------------

    def _build_context(self, chunks: list[LDU], facts: list[dict]) -> str:
        """Build context string combining chunks and structured facts."""
        parts = []

        for i, chunk in enumerate(chunks, 1):
            section = chunk.parent_section or "Unknown section"
            pages   = ", ".join(str(p) for p in chunk.page_refs)
            parts.append(
                f"[Source {i} | Section: {section} | Page: {pages}]\n"
                f"{chunk.content}"
            )

        if facts:
            parts.append("\n[Structured Facts from FactTable]")
            for f in facts[:10]:
                parts.append(
                    f"  {f['fact_label']}: {f['fact_value']} {f['fact_unit']} "
                    f"({f['fact_type']}, {f['fact_year'] or 'n/a'}) "
                    f"— Section: {f['section_title'] or '—'} | Page: {f['page_number']}"
                )

        return "\n\n".join(parts)

    # ----------------------------------------------------------
    # Gemini call
    # ----------------------------------------------------------

    def _call_gemini(self, query: str, context: str) -> str:
        """Call Google Gemini to generate a grounded answer."""
        api_key = os.getenv("GOOGLE_GEMINI_API_KEY", "")

        if not api_key or api_key in ("your_key_here", "your_actual_key_here"):
            return f"[No API key — showing most relevant excerpt]\n\n{context[:800]}"

        try:
            model  = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            prompt = (
                "You are a document analyst. Answer the question using ONLY "
                "the provided document excerpts and structured facts. "
                "If the answer is not present, say 'Not found in the document'.\n\n"
                f"Document context:\n{context}\n\n"
                f"Question: {query}\n\nAnswer:"
            )
            payload = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}]
            }).encode()
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={api_key}"
            )
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        except Exception:
            return f"[Gemini call failed — showing most relevant excerpt]\n\n{context[:800]}"

    # ----------------------------------------------------------
    # Citations
    # ----------------------------------------------------------

    def _build_citations(
        self, chunks: list[LDU], index: PageIndex
    ) -> list[SourceCitation]:
        citations = []
        for chunk in chunks:
            node = next(
                (n for n in index.nodes if chunk.chunk_id in n.chunk_ids), None
            )
            citations.append(SourceCitation(
                doc_id=chunk.doc_id,
                document_name=index.filename,
                page_number=chunk.page_refs[0] if chunk.page_refs else 1,
                bbox=chunk.bounding_box,
                chunk_id=chunk.chunk_id,
                content_hash=chunk.content_hash,
                excerpt=chunk.content[:200],
                strategy_used=chunk.strategy_used,
                section_title=chunk.parent_section,
                section_node_id=node.node_id if node else None,
            ))
        return citations

    # ----------------------------------------------------------
    # Load helpers
    # ----------------------------------------------------------

    def _load_index(self, doc_id: str) -> PageIndex:
        path = config.refinery_dir / "pageindex" / f"{doc_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"PageIndex not found for '{doc_id}'.")
        with open(path) as f:
            data = json.load(f)
        nodes = []
        for n in data.get("nodes", []):
            nodes.append(PageIndexNode(
                node_id=n["node_id"], doc_id=n["doc_id"],
                title=n["title"], level=n.get("level", 1),
                page_start=n["page_start"], page_end=n["page_end"],
                summary=n.get("summary", ""),
                chunk_ids=n.get("chunk_ids", []),
                parent_node_id=n.get("parent_node_id"),
                child_node_ids=n.get("child_node_ids", []),
                data_types_present=n.get("data_types_present", []),
            ))
        return PageIndex(
            doc_id=data["doc_id"], filename=data["filename"],
            total_pages=data["total_pages"], nodes=nodes,
            root_node_ids=data.get("root_node_ids", []),
            document_summary=data.get("document_summary", ""),
            total_sections=data.get("total_sections", 0),
            total_chunks_indexed=data.get("total_chunks_indexed", 0),
        )

    def _load_chunks(self, doc_id: str) -> list[LDU]:
        path = config.refinery_dir / "chunks" / f"{doc_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Chunks not found for '{doc_id}'.")
        chunks = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d    = json.loads(line)
                bbox = None
                if d.get("bounding_box"):
                    b = d["bounding_box"]
                    bbox = BoundingBox(
                        x0=b.get("x0", 0), y0=b.get("y0", 0),
                        x1=b.get("x1", 0), y1=b.get("y1", 0),
                        page=b.get("page", 1),
                    )
                chunks.append(LDU(
                    chunk_id=d["chunk_id"], doc_id=d["doc_id"],
                    chunk_index=d["chunk_index"], content=d["content"],
                    chunk_type=d["chunk_type"],
                    page_refs=d.get("page_refs", [1]),
                    bounding_box=bbox,
                    content_hash=d.get("content_hash", ""),
                    parent_section=d.get("parent_section"),
                    parent_section_page=d.get("parent_section_page"),
                    token_count=d.get("token_count", 0),
                    strategy_used=d.get("strategy_used", ""),
                ))
        return chunks

    def _empty_response(self, query: str, reason: str) -> ProvenanceChain:
        return ProvenanceChain(
            query=query, answer=reason, citations=[], is_verified=False,
        )


# ════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter
    from src.agents.chunker import ChunkingEngine
    from src.agents.indexer import PageIndexBuilder

    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.query_agent <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    pageindex_path = (
        config.refinery_dir / "pageindex" /
        f"{pdf_path.stem.lower().replace(' ', '_')}.json"
    )

    if not pageindex_path.exists():
        print("Running full pipeline first...\n")
        triage    = TriageAgent()
        profile   = triage.run(pdf_path)
        router    = ExtractionRouter()
        extracted = router.run(pdf_path, profile)
        chunker   = ChunkingEngine()
        chunks    = chunker.run(extracted)
        builder   = PageIndexBuilder()
        index     = builder.run(
            doc_id=profile.doc_id, filename=profile.filename,
            total_pages=profile.total_pages, chunks=chunks,
        )
        doc_id = profile.doc_id
    else:
        doc_id = pdf_path.stem.lower().replace(" ", "_")
        print(f"Using existing index for '{doc_id}'")

    agent = QueryAgent(top_k=5)

    print(f"\n{'='*55}")
    print(f"  QUERY AGENT READY — document: {doc_id}")
    print(f"  Tools: pageindex_navigate | semantic_search | structured_query")
    print(f"  Type your question. Type 'exit' to quit.")
    print(f"{'='*55}\n")

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query or query.lower() in ("exit", "quit", "q"):
            break

        tool_used = classify_query(query)
        print(f"  → routing to: {tool_used}")

        result = agent.run(query, doc_id)

        print(f"\n{'='*55}")
        print(f"ANSWER:\n{result.answer}")
        print(f"\nSOURCES ({len(result.citations)} citations):")
        for i, cite in enumerate(result.citations, 1):
            print(f"  [{i}] Page {cite.page_number} | "
                  f"Section: {cite.section_title} | "
                  f"Chunk: {cite.chunk_id}")
            print(f"      \"{cite.excerpt[:100]}...\"")
        tools = result.__dict__.get("tools_used", [])
        facts = result.__dict__.get("facts_used", 0)
        print(f"\nTools used    : {tools}")
        print(f"Facts used    : {facts}")
        print(f"Nodes visited : {result.pageindex_nodes_traversed}")
        print(f"Verified      : {result.is_verified}")
        print(f"{'='*55}\n")
