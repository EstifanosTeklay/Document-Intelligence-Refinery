"""
PageIndex Builder — Stage 4 of the Document Intelligence Refinery

Takes the list of LDUs from Stage 3 and builds a hierarchical navigation
tree (PageIndex) over the document.

What it does:
  1. Detects section headings from LDUs to form the tree structure
  2. Maps every chunk to its parent section node
  3. Extracts key named entities per section (numbers, dates, orgs)
  4. Calls a fast LLM to generate a 2-3 sentence summary per section
  5. Saves the PageIndex to .refinery/pageindex/{doc_id}.json

The PageIndex lets the Query Agent navigate the document by section
before doing vector search — dramatically improving retrieval precision.
"""
import json
import re
import os
from pathlib import Path

from src.models.ldu import LDU, ChunkType
from src.models.pageindex import PageIndex, PageIndexNode, SectionEntity
from src.utils.config import config


class PageIndexBuilder:
    """
    Builds a hierarchical PageIndex from a list of LDUs.
    Section summaries are generated via OpenRouter API.
    Falls back to extractive summaries if no API key is set.
    """

    def __init__(self):
        self.cfg = config
        self.max_summary_tokens = self.cfg.get_raw("pageindex", "max_summary_tokens")
        self.summary_model      = self.cfg.get_raw("pageindex", "summary_model")

    # ----------------------------------------------------------
    # Public entry point
    # ----------------------------------------------------------

    def run(
        self,
        doc_id: str,
        filename: str,
        total_pages: int,
        chunks: list[LDU],
    ) -> PageIndex:
        """
        Build and save a PageIndex from a list of LDUs.
        Returns the completed PageIndex.
        """
        # Step 1 — Group chunks by section
        sections = self._group_by_section(chunks)

        # Step 2 — Build nodes
        nodes: list[PageIndexNode] = []
        root_node_ids: list[str] = []
        node_index = 0

        for section_title, section_chunks in sections.items():
            node_id = f"{doc_id}_node_{node_index:04d}"

            # Page span for this section
            all_pages = [p for c in section_chunks for p in c.page_refs]
            page_start = min(all_pages) if all_pages else 1
            page_end   = max(all_pages) if all_pages else 1

            # Heading level from the heading chunk if present
            level = self._detect_heading_level(section_title, section_chunks)

            # Chunk IDs in this section
            chunk_ids = [c.chunk_id for c in section_chunks]

            # Key entities
            section_text = " ".join(c.content for c in section_chunks)
            entities = self._extract_entities(section_text)

            # Data types present
            data_types = self._detect_data_types(section_chunks)

            # Summary — LLM or extractive fallback
            summary = self._generate_summary(section_title, section_text)

            node = PageIndexNode(
                node_id=node_id,
                doc_id=doc_id,
                title=section_title,
                level=level,
                page_start=page_start,
                page_end=page_end,
                summary=summary,
                key_entities=entities,
                data_types_present=data_types,
                chunk_ids=chunk_ids,
                parent_node_id=None,
                child_node_ids=[],
            )
            nodes.append(node)
            root_node_ids.append(node_id)
            node_index += 1

        # Step 3 — Build parent/child relationships
        nodes = self._build_hierarchy(nodes)

        # Step 4 — Document-level summary
        doc_summary = self._generate_document_summary(doc_id, nodes)

        # Step 5 — Assemble PageIndex
        index = PageIndex(
            doc_id=doc_id,
            filename=filename,
            total_pages=total_pages,
            nodes=nodes,
            root_node_ids=[n.node_id for n in nodes if n.parent_node_id is None],
            document_summary=doc_summary,
            total_sections=len(nodes),
            total_chunks_indexed=len(chunks),
        )

        # Save
        self._save_index(index)
        return index

    # ----------------------------------------------------------
    # Step 1 — Group chunks by section
    # ----------------------------------------------------------

    def _group_by_section(self, chunks: list[LDU]) -> dict[str, list[LDU]]:
        """Group chunks by their parent_section. Ungrouped chunks go to 'Document'."""
        sections: dict[str, list[LDU]] = {}
        for chunk in chunks:
            section = chunk.parent_section or "Document"
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk)
        return sections

    # ----------------------------------------------------------
    # Step 2 — Heading level detection
    # ----------------------------------------------------------

    def _detect_heading_level(self, title: str, chunks: list[LDU]) -> int:
        """
        Estimate heading depth from numbering patterns.
        '1.' = level 1, '1.1' = level 2, '1.1.1' = level 3
        """
        match = re.match(r"^(\d+)(\.(\d+))?(\.(\d+))?", title.strip())
        if not match:
            return 1
        if match.group(5):
            return 3
        if match.group(3):
            return 2
        return 1

    # ----------------------------------------------------------
    # Step 3 — Entity extraction
    # ----------------------------------------------------------

    def _extract_entities(self, text: str) -> list[SectionEntity]:
        """
        Extract key named entities using regex patterns.
        No external NLP library required.
        """
        entities = []
        seen = set()

        patterns = [
            # Money / financial figures
            (r"ETB\s?[\d,\.]+\s?(billion|million|trillion)?",        "MONEY"),
            (r"\$[\d,\.]+\s?(billion|million|trillion)?",             "MONEY"),
            (r"[\d,\.]+\s?(billion|million|trillion)\s?(birr|USD)?",  "MONEY"),
            # Percentages
            (r"\d+\.?\d*\s?%",                                        "PERCENT"),
            # Years / dates
            (r"\b(19|20)\d{2}[\/\-](19|20)?\d{2}\b",                 "DATE"),
            (r"\b(19|20)\d{2}\b",                                     "DATE"),
            # Organisations (capitalised multi-word)
            (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b",              "ORG"),
        ]

        for pattern, entity_type in patterns:
            for match in re.finditer(pattern, text):
                text_val = match.group(0).strip()
                key = (text_val, entity_type)
                if key not in seen and len(text_val) > 2:
                    seen.add(key)
                    entities.append(SectionEntity(
                        text=text_val,
                        entity_type=entity_type,
                    ))
                if len(entities) >= 10:
                    break
            if len(entities) >= 10:
                break

        return entities

    # ----------------------------------------------------------
    # Step 4 — Data type detection
    # ----------------------------------------------------------

    def _detect_data_types(self, chunks: list[LDU]) -> list[str]:
        """Detect what data types are present in this section."""
        types = set()
        for chunk in chunks:
            if chunk.chunk_type == ChunkType.TABLE or chunk.chunk_type == "table":
                types.add("tables")
            if chunk.chunk_type == ChunkType.FIGURE or chunk.chunk_type == "figure":
                types.add("figures")
            if chunk.chunk_type == ChunkType.LIST or chunk.chunk_type == "list":
                types.add("lists")
            if chunk.chunk_type == ChunkType.HEADING or chunk.chunk_type == "heading":
                types.add("headings")
            if chunk.relationships:
                types.add("cross_references")
        return sorted(types)

    # ----------------------------------------------------------
    # Step 5 — Summary generation
    # ----------------------------------------------------------

    def _generate_summary(self, title: str, text: str) -> str:
        """
        Generate a 2-3 sentence summary of a section.
        Uses OpenRouter LLM if API key is set, else extractive fallback.
        """
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if api_key and api_key != "your_openrouter_api_key_here":
            return self._llm_summary(title, text, api_key)
        return self._extractive_summary(text)

    def _llm_summary(self, title: str, text: str, api_key: str) -> str:
        """Call OpenRouter to generate a section summary."""
        try:
            import urllib.request
            import json

            # Truncate text to avoid excessive tokens
            truncated = text[:2000] if len(text) > 2000 else text

            payload = json.dumps({
                "model": self.summary_model,
                "max_tokens": self.max_summary_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Summarise this document section in 2-3 sentences. "
                            f"Section title: '{title}'\n\n{truncated}"
                        )
                    }
                ]
            }).encode()

            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            # Fall back to extractive if LLM call fails
            return self._extractive_summary(text)

    def _extractive_summary(self, text: str) -> str:
        """
        Extractive fallback — take first 2 sentences from the section text.
        Used when no API key is set.
        """
        if not text.strip():
            return "No content available for this section."
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        summary_sentences = [s.strip() for s in sentences[:2] if len(s.strip()) > 20]
        if not summary_sentences:
            return text[:200].strip()
        return " ".join(summary_sentences)

    def _generate_document_summary(self, doc_id: str, nodes: list[PageIndexNode]) -> str:
        """Generate a document-level summary from section titles and summaries."""
        if not nodes:
            return "Empty document."
        titles = [n.title for n in nodes[:5]]
        return f"Document with {len(nodes)} sections including: {', '.join(titles)}."

    # ----------------------------------------------------------
    # Hierarchy building
    # ----------------------------------------------------------

    def _build_hierarchy(self, nodes: list[PageIndexNode]) -> list[PageIndexNode]:
        """
        Assign parent/child relationships based on heading levels.
        Level 1 nodes are roots, level 2 are children of the nearest level 1, etc.
        """
        stack: list[PageIndexNode] = []

        for node in nodes:
            # Pop stack until we find a node at a higher level
            while stack and stack[-1].level >= node.level:
                stack.pop()

            if stack:
                parent = stack[-1]
                node.parent_node_id = parent.node_id
                parent.child_node_ids.append(node.node_id)

            stack.append(node)

        return nodes

    # ----------------------------------------------------------
    # Persist
    # ----------------------------------------------------------

    def _save_index(self, index: PageIndex) -> None:
        """Save PageIndex to .refinery/pageindex/{doc_id}.json"""
        pageindex_dir = config.refinery_dir / "pageindex"
        pageindex_dir.mkdir(parents=True, exist_ok=True)
        out_path = pageindex_dir / f"{index.doc_id}.json"
        with open(out_path, "w") as f:
            f.write(json.dumps(index.model_dump(), indent=2))


# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter
    from src.agents.chunker import ChunkingEngine

    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.indexer <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    # Stage 1 — Triage
    print("Running Triage Agent...")
    triage = TriageAgent()
    profile = triage.run(pdf_path)
    print(f"  strategy selected : {profile.estimated_extraction_cost}")

    # Stage 2 — Extraction
    print("\nRunning Extraction Router...")
    router = ExtractionRouter()
    extracted = router.run(pdf_path, profile)
    print(f"  strategy used     : {extracted.strategy_used}")
    print(f"  text_blocks       : {len(extracted.text_blocks)}")
    print(f"  tables            : {len(extracted.tables)}")

    # Stage 3 — Chunking
    print("\nRunning Chunking Engine...")
    chunker = ChunkingEngine()
    chunks = chunker.run(extracted)
    print(f"  total chunks      : {len(chunks)}")

    # Stage 4 — PageIndex
    print("\nBuilding PageIndex...")
    builder = PageIndexBuilder()
    index = builder.run(
        doc_id=profile.doc_id,
        filename=profile.filename,
        total_pages=profile.total_pages,
        chunks=chunks,
    )

    print("\n" + "="*50)
    print("  PAGE INDEX COMPLETE")
    print("="*50)
    print(f"  total sections    : {index.total_sections}")
    print(f"  total chunks      : {index.total_chunks_indexed}")
    print(f"  root sections     : {len(index.root_node_ids)}")
    print(f"  doc summary       : {index.document_summary}")
    print(f"\n  saved to          : .refinery/pageindex/{profile.doc_id}.json")
    print("="*50)

    # Show section tree
    print("\nSection Tree:")
    for node in index.nodes:
        indent = "  " * (node.level - 1)
        has_children = "+" if node.child_node_ids else "-"
        print(f"  {indent}[{has_children}] {node.title}  "
              f"(p{node.page_start}-{node.page_end}, "
              f"{len(node.chunk_ids)} chunks)")
        if node.summary:
            summary_preview = node.summary[:80] + "..." if len(node.summary) > 80 else node.summary
            print(f"  {indent}    {summary_preview}")
