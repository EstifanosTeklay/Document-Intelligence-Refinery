"""
Strategy B — Layout-Aware Extractor
Tool: Docling (IBM Research)
Triggers: multi_column, table_heavy, figure_heavy, mixed layouts
Cost: Medium (runs local layout models)
"""
import time
from pathlib import Path

from src.models import (
    DocumentProfile,
    ExtractedDocument,
    ExtractedTable,
    ExtractedFigure,
    TextBlock,
    BoundingBox,
)
from .base import BaseExtractor


class LayoutExtractor(BaseExtractor):
    """
    Strategy B: Layout-aware extraction using Docling.
    Handles multi-column layouts, tables as structured JSON,
    figures with captions, and reading order reconstruction.
    """

    @property
    def strategy_name(self) -> str:
        return "B"

    def extract(
        self,
        file_path: Path,
        profile: DocumentProfile,
    ) -> ExtractedDocument:
        start = time.time()

        try:
            from docling.document_converter import DocumentConverter
            result = self._extract_with_docling(file_path, profile, start)
            return result
        except ImportError:
            # Docling not installed — degrade gracefully with a low-confidence result
            return self._fallback_extraction(file_path, profile, start)

    # ----------------------------------------------------------
    # Docling extraction
    # ----------------------------------------------------------

    def _extract_with_docling(
        self,
        file_path: Path,
        profile: DocumentProfile,
        start: float,
    ) -> ExtractedDocument:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        doc = result.document

        text_blocks: list[TextBlock] = []
        tables: list[ExtractedTable] = []
        figures: list[ExtractedFigure] = []
        pages_text: dict[int, str] = {}
        reading_order: list[str] = []

        # --- Text blocks ---
        for idx, text_item in enumerate(doc.texts):
            page_num = self._get_page_num(text_item)
            bbox = self._get_bbox(text_item, page_num)
            text_blocks.append(TextBlock(
                text=text_item.text,
                bbox=bbox,
                is_heading=self._is_heading(text_item),
            ))
            pages_text.setdefault(page_num, "")
            pages_text[page_num] += text_item.text + "\n"
            reading_order.append(f"text:{idx}")

        # --- Tables ---
        for t_idx, table in enumerate(doc.tables):
            page_num = self._get_page_num(table)
            bbox = self._get_bbox(table, page_num)
            table_id = f"table_{page_num}_{t_idx:02d}"

            headers, rows = self._parse_docling_table(table)
            tables.append(ExtractedTable(
                table_id=table_id,
                bbox=bbox,
                headers=headers,
                rows=rows,
                confidence=0.85,
            ))
            reading_order.append(f"table:{table_id}")

        # --- Figures ---
        for f_idx, figure in enumerate(doc.pictures):
            page_num = self._get_page_num(figure)
            bbox = self._get_bbox(figure, page_num)
            figure_id = f"fig_{page_num}_{f_idx:02d}"
            caption = self._get_caption(figure)
            figures.append(ExtractedFigure(
                figure_id=figure_id,
                bbox=bbox,
                caption=caption,
            ))
            reading_order.append(f"figure:{figure_id}")

        confidence = self._compute_confidence(text_blocks, tables, profile)

        return ExtractedDocument(
            doc_id=profile.doc_id,
            filename=profile.filename,
            total_pages=profile.total_pages,
            strategy_used=self.strategy_name,
            extraction_confidence=confidence,
            cost_estimate_usd=0.0,
            processing_time_seconds=round(time.time() - start, 2),
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            reading_order=reading_order,
            pages_text=pages_text,
        )

    # ----------------------------------------------------------
    # Docling helpers
    # ----------------------------------------------------------

    def _get_page_num(self, item) -> int:
        try:
            return item.prov[0].page_no if item.prov else 1
        except (AttributeError, IndexError):
            return 1

    def _get_bbox(self, item, page_num: int) -> BoundingBox:
        try:
            bbox = item.prov[0].bbox
            return BoundingBox(
                x0=bbox.l, y0=bbox.b,
                x1=bbox.r, y1=bbox.t,
                page=page_num
            )
        except (AttributeError, IndexError):
            return BoundingBox(x0=0, y0=0, x1=0, y1=0, page=page_num)

    def _is_heading(self, item) -> bool:
        try:
            label = str(item.label).lower()
            return "head" in label or "title" in label
        except AttributeError:
            return False

    def _get_caption(self, figure) -> str | None:
        try:
            if figure.captions:
                return figure.captions[0].text
        except (AttributeError, IndexError):
            pass
        return None

    def _parse_docling_table(self, table) -> tuple[list[str], list[list[str]]]:
        """Extract headers and rows from a Docling TableItem."""
        try:
            df = table.export_to_dataframe()
            headers = [str(col) for col in df.columns.tolist()]
            rows = [[str(cell) for cell in row] for row in df.values.tolist()]
            return headers, rows
        except Exception:
            return [], []

    def _compute_confidence(
        self,
        text_blocks: list[TextBlock],
        tables: list[ExtractedTable],
        profile: DocumentProfile,
    ) -> float:
        if not text_blocks and not tables:
            return 0.2
        if profile.avg_chars_per_page > 100:
            return 0.85
        return 0.60

    # ----------------------------------------------------------
    # Fallback when Docling not installed
    # ----------------------------------------------------------

    def _fallback_extraction(
        self,
        file_path: Path,
        profile: DocumentProfile,
        start: float,
    ) -> ExtractedDocument:
        """
        Graceful degradation: return a low-confidence empty extraction
        so the router knows to escalate to Strategy C.
        """
        return ExtractedDocument(
            doc_id=profile.doc_id,
            filename=profile.filename,
            total_pages=profile.total_pages,
            strategy_used=self.strategy_name,
            extraction_confidence=0.1,  # Forces escalation to Strategy C
            cost_estimate_usd=0.0,
            processing_time_seconds=round(time.time() - start, 2),
            text_blocks=[],
            tables=[],
            figures=[],
            reading_order=[],
            pages_text={},
        )
