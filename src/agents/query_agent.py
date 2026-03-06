"""
Query Agent — Stage 5 of the Document Intelligence Refinery

Takes a natural language question, retrieves the most relevant chunks
using the PageIndex + semantic search, then calls Gemini to generate
a grounded answer with full provenance.

Retrieval flow:
  1. Search PageIndex by title keywords → narrow candidate sections
  2. Score all chunks in candidate sections by keyword overlap
  3. Pass top-k chunks as context to Gemini
  4. Return answer + ProvenanceChain with full citations
"""
import json
import os
import re
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

from src.models.ldu import LDU
from src.models.pageindex import PageIndex, PageIndexNode
from src.models.provenance import ProvenanceChain, SourceCitation
from src.models.extracted_document import BoundingBox
from src.utils.config import config

# Load .env file to ensure environment variables are available
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")


class QueryAgent:
    """
    Answers natural language questions over an indexed document.
    Uses PageIndex for navigation, keyword scoring for retrieval,
    and Google Gemini for answer generation.
    """

    def __init__(self, top_k: int = 5):
        self.cfg   = config
        self.top_k = top_k

    # ----------------------------------------------------------
    # Public entry point
    # ----------------------------------------------------------

    def run(
        self,
        query: str,
        doc_id: str,
    ) -> ProvenanceChain:
        """
        Answer a query over a previously indexed document.

        Args:
            query:  Natural language question
            doc_id: Document ID (stem of PDF filename, lowercase)

        Returns:
            ProvenanceChain with answer + full citations
        """
        # Step 1 — Load PageIndex and chunks
        index  = self._load_index(doc_id)
        chunks = self._load_chunks(doc_id)

        if not chunks:
            return self._empty_response(query, "No chunks found for this document.")

        # Step 2 — PageIndex navigation: find relevant sections
        candidate_nodes = self._navigate_index(query, index)
        nodes_traversed = [n.node_id for n in candidate_nodes]

        # Step 3 — Retrieve top-k chunks from candidate sections
        top_chunks = self._retrieve_chunks(query, chunks, candidate_nodes)

        if not top_chunks:
            top_chunks = chunks[:self.top_k]  # fallback to first k chunks

        # Step 4 — Call Gemini with context
        context   = self._build_context(top_chunks)
        answer    = self._call_gemini(query, context)

        # Step 5 — Build ProvenanceChain
        citations = self._build_citations(top_chunks, index)
        chain     = ProvenanceChain(
            query=query,
            answer=answer,
            citations=citations,
            is_verified=len(citations) > 0,
            verified_chunk_ids=[c.chunk_id for c in top_chunks],
            pageindex_nodes_traversed=nodes_traversed,
        )
        chain.set_primary_from_citations()
        return chain

    # ----------------------------------------------------------
    # Step 1 — Load index and chunks from disk
    # ----------------------------------------------------------

    def _load_index(self, doc_id: str) -> PageIndex:
        """Load PageIndex from .refinery/pageindex/{doc_id}.json"""
        path = config.refinery_dir / "pageindex" / f"{doc_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"PageIndex not found for '{doc_id}'. "
                f"Run the indexer first: python -m src.agents.indexer <pdf>"
            )
        with open(path) as f:
            data = json.load(f)

        # Reconstruct PageIndex from dict
        nodes = []
        for n in data.get("nodes", []):
            nodes.append(PageIndexNode(
                node_id=n["node_id"],
                doc_id=n["doc_id"],
                title=n["title"],
                level=n.get("level", 1),
                page_start=n["page_start"],
                page_end=n["page_end"],
                summary=n.get("summary", ""),
                chunk_ids=n.get("chunk_ids", []),
                parent_node_id=n.get("parent_node_id"),
                child_node_ids=n.get("child_node_ids", []),
                data_types_present=n.get("data_types_present", []),
            ))

        return PageIndex(
            doc_id=data["doc_id"],
            filename=data["filename"],
            total_pages=data["total_pages"],
            nodes=nodes,
            root_node_ids=data.get("root_node_ids", []),
            document_summary=data.get("document_summary", ""),
            total_sections=data.get("total_sections", 0),
            total_chunks_indexed=data.get("total_chunks_indexed", 0),
        )

    def _load_chunks(self, doc_id: str) -> list[LDU]:
        """Load all LDUs from .refinery/chunks/{doc_id}.jsonl"""
        path = config.refinery_dir / "chunks" / f"{doc_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Chunks not found for '{doc_id}'. "
                f"Run the chunker first: python -m src.agents.chunker <pdf>"
            )
        chunks = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                bbox = None
                if d.get("bounding_box"):
                    b = d["bounding_box"]
                    bbox = BoundingBox(
                        x0=b.get("x0", 0), y0=b.get("y0", 0),
                        x1=b.get("x1", 0), y1=b.get("y1", 0),
                        page=b.get("page", 1),
                    )
                chunks.append(LDU(
                    chunk_id=d["chunk_id"],
                    doc_id=d["doc_id"],
                    chunk_index=d["chunk_index"],
                    content=d["content"],
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

    # ----------------------------------------------------------
    # Step 2 — PageIndex navigation
    # ----------------------------------------------------------

    def _navigate_index(
        self, query: str, index: PageIndex
    ) -> list[PageIndexNode]:
        """
        Search PageIndex by keyword overlap with query.
        Returns relevant section nodes to focus retrieval on.
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        scored = []

        for node in index.nodes:
            node_words = set(re.findall(r"\w+",
                (node.title + " " + node.summary).lower()))
            overlap = len(query_words & node_words)
            if overlap > 0:
                scored.append((overlap, node))

        scored.sort(key=lambda x: -x[0])

        # Return top 3 sections, or all if fewer
        top_nodes = [n for _, n in scored[:3]]

        # If nothing matched, return all nodes
        return top_nodes if top_nodes else index.nodes

    # ----------------------------------------------------------
    # Step 3 — Chunk retrieval
    # ----------------------------------------------------------

    def _retrieve_chunks(
        self,
        query: str,
        all_chunks: list[LDU],
        candidate_nodes: list[PageIndexNode],
    ) -> list[LDU]:
        """
        Score and rank chunks by keyword overlap with the query.
        Prioritises chunks from candidate sections.
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        candidate_chunk_ids = set(
            cid for node in candidate_nodes for cid in node.chunk_ids
        )

        scored = []
        for chunk in all_chunks:
            chunk_words = set(re.findall(r"\w+", chunk.content.lower()))
            overlap = len(query_words & chunk_words)

            # Boost score if chunk is in a candidate section
            boost = 2 if chunk.chunk_id in candidate_chunk_ids else 1
            score = overlap * boost

            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:self.top_k]]

    # ----------------------------------------------------------
    # Step 4 — Gemini call
    # ----------------------------------------------------------

    def _build_context(self, chunks: list[LDU]) -> str:
        """Build context string from top chunks."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            section = chunk.parent_section or "Unknown section"
            pages   = ", ".join(str(p) for p in chunk.page_refs)
            parts.append(
                f"[Source {i} | Section: {section} | Page: {pages}]\n"
                f"{chunk.content}"
            )
        return "\n\n".join(parts)

    def _call_gemini(self, query: str, context: str) -> str:
        """Call Google Gemini to generate a grounded answer."""
        api_key = os.getenv("GOOGLE_GEMINI_API_KEY", "")
        print(f"[DEBUG] API KEY: '{api_key}' (length: {len(api_key)})")

        if not api_key:
            return self._keyword_answer(query, context)

        try:
            model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

            prompt = (
                f"You are a document analyst. Answer the question using ONLY "
                f"the provided document excerpts. If the answer is not in the "
                f"excerpts, say 'Not found in the document'.\n\n"
                f"Document excerpts:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )

            payload = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}]
            }).encode()

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={api_key}"
            )
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        except Exception as e:
            return self._keyword_answer(query, context)

    def _keyword_answer(self, query: str, context: str) -> str:
        """
        Fallback when no API key — returns the most relevant excerpt directly.
        """
        return (
            f"[No API key — showing most relevant excerpt]\n\n{context[:800]}"
        )

    # ----------------------------------------------------------
    # Step 5 — Build ProvenanceChain
    # ----------------------------------------------------------

    def _build_citations(
        self, chunks: list[LDU], index: PageIndex
    ) -> list[SourceCitation]:
        """Build SourceCitation for each chunk used in the answer."""
        citations = []
        for chunk in chunks:
            # Find which PageIndex node this chunk belongs to
            node = next(
                (n for n in index.nodes if chunk.chunk_id in n.chunk_ids),
                None
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

    def _empty_response(self, query: str, reason: str) -> ProvenanceChain:
        return ProvenanceChain(
            query=query,
            answer=reason,
            citations=[],
            is_verified=False,
        )


# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------

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

    # Run full pipeline first if not already indexed
    pageindex_path = (
        config.refinery_dir / "pageindex" /
        f"{pdf_path.stem.lower().replace(' ', '_')}.json"
    )

    if not pageindex_path.exists():
        print("Running full pipeline first...\n")

        print("Stage 1 — Triage...")
        triage = TriageAgent()
        profile = triage.run(pdf_path)
        print(f"  strategy: {profile.estimated_extraction_cost}")

        print("Stage 2 — Extraction...")
        router = ExtractionRouter()
        extracted = router.run(pdf_path, profile)
        print(f"  strategy used: {extracted.strategy_used}")

        print("Stage 3 — Chunking...")
        chunker = ChunkingEngine()
        chunks = chunker.run(extracted)
        print(f"  chunks: {len(chunks)}")

        print("Stage 4 — PageIndex...")
        builder = PageIndexBuilder()
        index = builder.run(
            doc_id=profile.doc_id,
            filename=profile.filename,
            total_pages=profile.total_pages,
            chunks=chunks,
        )
        print(f"  sections: {index.total_sections}")
        doc_id = profile.doc_id
    else:
        doc_id = pdf_path.stem.lower().replace(" ", "_")
        print(f"Using existing index for '{doc_id}'")

    # Interactive query loop
    agent = QueryAgent(top_k=5)

    print(f"\n{'='*55}")
    print(f"  QUERY AGENT READY — document: {doc_id}")
    print(f"  Type your question. Type 'exit' to quit.")
    print(f"{'='*55}\n")

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            break

        result = agent.run(query, doc_id)

        print(f"\n{'='*55}")
        print(f"ANSWER:\n{result.answer}")
        print(f"\nSOURCES ({len(result.citations)} citations):")
        for i, cite in enumerate(result.citations, 1):
            print(f"  [{i}] Page {cite.page_number} | "
                  f"Section: {cite.section_title} | "
                  f"Chunk: {cite.chunk_id}")
            print(f"      \"{cite.excerpt[:100]}...\"")
        print(f"\nPageIndex nodes traversed: {result.pageindex_nodes_traversed}")
        print(f"Verified: {result.is_verified}")
        print(f"{'='*55}\n")
