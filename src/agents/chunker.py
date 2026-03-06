"""
Chunking Engine — Stage 3 of the Document Intelligence Refinery

Takes an ExtractedDocument from Stage 2 and produces a list of LDUs
(Logical Document Units) — the semantic atoms of the RAG pipeline.

The 5 inviolable chunking rules (from extraction_rules.yaml):
  Rule 1: Never split a table cell from its header
  Rule 2: Figure caption always stored as figure metadata, not standalone chunk
  Rule 3: Numbered/bulleted list = single LDU unless it exceeds max_tokens
  Rule 4: Section header stored as parent_section metadata on child chunks
  Rule 5: Cross-references resolved and stored as relationships on LDUs
"""
import re
import hashlib
from pathlib import Path

from src.models import ExtractedDocument
from src.models.ldu import LDU, ChunkType, ChunkRelationship
from src.utils.config import config


class ChunkingEngine:
    """
    Converts an ExtractedDocument into a list of LDUs following the
    5 chunking rules defined in extraction_rules.yaml.
    """

    def __init__(self):
        self.cfg = config
        self.max_tokens = self.cfg.get_raw("chunking", "max_tokens_per_chunk")
        self.min_tokens = self.cfg.get_raw("chunking", "min_tokens_per_chunk")
        self.overlap    = self.cfg.get_raw("chunking", "overlap_tokens")

    # ----------------------------------------------------------
    # Public entry point
    # ----------------------------------------------------------

    def run(self, extracted: ExtractedDocument) -> list[LDU]:
        """
        Chunk an ExtractedDocument into LDUs.
        Returns an ordered list of LDUs ready for embedding.
        """
        chunks: list[LDU] = []
        current_section: str | None = None
        current_section_page: int | None = None
        chunk_index = 0

        # Process in reading order
        for item_ref in extracted.reading_order:
            kind, ref_id = item_ref.split(":", 1)

            # ── Text blocks ──────────────────────────────────
            if kind == "text":
                idx = int(ref_id)
                if idx >= len(extracted.text_blocks):
                    continue
                block = extracted.text_blocks[idx]

                # Rule 4: detect section heading — store as metadata
                if block.is_heading:
                    current_section = block.text.strip()
                    current_section_page = block.bbox.page if block.bbox else None
                    # Heading becomes its own small chunk
                    chunks.append(self._make_chunk(
                        doc_id=extracted.doc_id,
                        index=chunk_index,
                        content=block.text,
                        chunk_type=ChunkType.HEADING,
                        page_refs=[block.bbox.page] if block.bbox else [1],
                        bbox=block.bbox,
                        parent_section=current_section,
                        parent_section_page=current_section_page,
                        strategy=extracted.strategy_used,
                    ))
                    chunk_index += 1
                    continue

                # Rule 3: detect lists — keep as single LDU
                if self._is_list(block.text):
                    chunks.append(self._make_chunk(
                        doc_id=extracted.doc_id,
                        index=chunk_index,
                        content=block.text,
                        chunk_type=ChunkType.LIST,
                        page_refs=[block.bbox.page] if block.bbox else [1],
                        bbox=block.bbox,
                        parent_section=current_section,
                        parent_section_page=current_section_page,
                        strategy=extracted.strategy_used,
                    ))
                    chunk_index += 1
                    continue

                # Regular text — split if over max_tokens
                text_chunks = self._split_text(block.text)
                for text in text_chunks:
                    if not text.strip():
                        continue
                    # Rule 5: detect and store cross-references
                    relationships = self._detect_cross_references(text, chunks)
                    chunks.append(self._make_chunk(
                        doc_id=extracted.doc_id,
                        index=chunk_index,
                        content=text,
                        chunk_type=ChunkType.TEXT,
                        page_refs=[block.bbox.page] if block.bbox else [1],
                        bbox=block.bbox,
                        parent_section=current_section,
                        parent_section_page=current_section_page,
                        strategy=extracted.strategy_used,
                        relationships=relationships,
                    ))
                    chunk_index += 1

            # ── Tables ───────────────────────────────────────
            elif kind == "table":
                table = next(
                    (t for t in extracted.tables if t.table_id == ref_id), None
                )
                if not table:
                    continue

                # Rule 1: table is always one LDU — headers + rows never split
                table_text = self._table_to_text(table.headers, table.rows)
                chunks.append(self._make_chunk(
                    doc_id=extracted.doc_id,
                    index=chunk_index,
                    content=table_text,
                    chunk_type=ChunkType.TABLE,
                    page_refs=[table.bbox.page] if table.bbox else [1],
                    bbox=table.bbox,
                    parent_section=current_section,
                    parent_section_page=current_section_page,
                    strategy=extracted.strategy_used,
                    table_data={
                        "headers": table.headers,
                        "rows": table.rows,
                    },
                ))
                chunk_index += 1

            # ── Figures ──────────────────────────────────────
            elif kind == "figure":
                figure_id = ref_id
                figure = next(
                    (f for f in extracted.figures if f.figure_id == figure_id), None
                )
                if not figure:
                    continue

                # Rule 2: caption stored as figure metadata, not separate chunk
                caption = figure.caption or ""
                content = f"[Figure] {caption}" if caption else "[Figure]"
                chunks.append(self._make_chunk(
                    doc_id=extracted.doc_id,
                    index=chunk_index,
                    content=content,
                    chunk_type=ChunkType.FIGURE,
                    page_refs=[figure.bbox.page] if figure.bbox else [1],
                    bbox=figure.bbox,
                    parent_section=current_section,
                    parent_section_page=current_section_page,
                    strategy=extracted.strategy_used,
                    figure_caption=caption,
                ))
                chunk_index += 1

        # Save chunks to disk
        self._save_chunks(extracted.doc_id, chunks)
        return chunks

    # ----------------------------------------------------------
    # Chunk factory
    # ----------------------------------------------------------

    def _make_chunk(
        self,
        doc_id: str,
        index: int,
        content: str,
        chunk_type: ChunkType,
        page_refs: list[int],
        bbox=None,
        parent_section: str | None = None,
        parent_section_page: int | None = None,
        strategy: str = "",
        relationships: list[ChunkRelationship] | None = None,
        table_data: dict | None = None,
        figure_caption: str | None = None,
    ) -> LDU:
        return LDU(
            chunk_id=f"{doc_id}_chunk_{index:04d}",
            doc_id=doc_id,
            chunk_index=index,
            content=content,
            chunk_type=chunk_type,
            page_refs=page_refs,
            bounding_box=bbox,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            parent_section=parent_section,
            parent_section_page=parent_section_page,
            token_count=self._estimate_tokens(content),
            relationships=relationships or [],
            table_data=table_data,
            figure_caption=figure_caption,
            strategy_used=strategy,
        )

    # ----------------------------------------------------------
    # Rule helpers
    # ----------------------------------------------------------

    def _is_list(self, text: str) -> bool:
        """Rule 3: detect numbered or bulleted lists."""
        lines = text.strip().splitlines()
        if len(lines) < 2:
            return False
        list_pattern = re.compile(r"^(\s*[\-\•\*\·]|\s*\d+[\.\)])\s+")
        matches = sum(1 for line in lines if list_pattern.match(line))
        return matches >= 2

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks respecting max_tokens. Adds overlap."""
        tokens = text.split()
        if len(tokens) <= self.max_tokens:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk = " ".join(tokens[start:end])
            chunks.append(chunk)
            if end == len(tokens):
                break
            start = end - self.overlap  # overlap between chunks
        return chunks

    def _table_to_text(self, headers: list[str], rows: list[list[str]]) -> str:
        """Rule 1: convert table to readable text keeping headers with rows."""
        if not headers and not rows:
            return "[Empty table]"
        lines = []
        if headers:
            lines.append(" | ".join(headers))
            lines.append("-" * len(lines[0]))
        for row in rows:
            lines.append(" | ".join(str(cell) for cell in row))
        return "\n".join(lines)

    def _detect_cross_references(
        self, text: str, existing_chunks: list[LDU]
    ) -> list[ChunkRelationship]:
        """
        Rule 5: detect references like 'see Table 3', 'Figure 2', 'Section 4'.
        Links them to existing chunks by type matching.
        """
        relationships = []
        patterns = [
            (r"\b[Tt]able\s+(\d+)", "TABLE"),
            (r"\b[Ff]igure\s+(\d+)", "FIGURE"),
            (r"\b[Ss]ection\s+(\d+)", "HEADING"),
        ]
        for pattern, chunk_type in patterns:
            for match in re.finditer(pattern, text):
                # Find a matching chunk of the same type
                target = next(
                    (c for c in existing_chunks
                     if c.chunk_type == chunk_type.lower()),
                    None
                )
                if target:
                    relationships.append(ChunkRelationship(
                        target_chunk_id=target.chunk_id,
                        relationship_type="references",
                    ))
        return relationships

    # ----------------------------------------------------------
    # Token estimation
    # ----------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Approximate token count — 1 token ≈ 4 characters."""
        return max(1, len(text) // 4)

    # ----------------------------------------------------------
    # Persist chunks
    # ----------------------------------------------------------

    def _save_chunks(self, doc_id: str, chunks: list[LDU]) -> None:
        """Save all LDUs to .refinery/chunks/{doc_id}.jsonl"""
        import json
        chunks_dir = config.refinery_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        out_path = chunks_dir / f"{doc_id}.jsonl"
        with open(out_path, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.model_dump()) + "\n")


# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path
    from src.agents.triage import TriageAgent
    from src.agents.extractor import ExtractionRouter

    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.chunker <path_to_pdf>")
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
    print(f"  confidence        : {extracted.extraction_confidence}")
    print(f"  text_blocks       : {len(extracted.text_blocks)}")
    print(f"  tables            : {len(extracted.tables)}")

    # Stage 3 — Chunking
    print("\nRunning Chunking Engine...")
    chunker = ChunkingEngine()
    chunks = chunker.run(extracted)

    print("\n" + "="*50)
    print("  CHUNKING COMPLETE")
    print("="*50)
    print(f"  total chunks      : {len(chunks)}")

    # Breakdown by type
    from collections import Counter
    type_counts = Counter(c.chunk_type for c in chunks)
    for chunk_type, count in type_counts.items():
        print(f"  {chunk_type:<18} : {count}")

    print(f"\n  avg tokens/chunk  : {sum(c.token_count for c in chunks) // max(len(chunks),1)}")
    print(f"  chunks with xrefs : {sum(1 for c in chunks if c.relationships)}")
    print(f"  chunks with parent: {sum(1 for c in chunks if c.parent_section)}")
    print(f"\n  saved to          : .refinery/chunks/{profile.doc_id}.jsonl")
    print("="*50)

    # Show first 3 chunks as a sample
    print("\nSample — first 3 chunks:")
    for chunk in chunks[:3]:
        print(f"\n  [{chunk.chunk_id}]")
        print(f"  type    : {chunk.chunk_type}")
        print(f"  pages   : {chunk.page_refs}")
        print(f"  section : {chunk.parent_section}")
        print(f"  tokens  : {chunk.token_count}")
        print(f"  content : {chunk.content[:120]}...")
