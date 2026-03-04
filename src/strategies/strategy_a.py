"""
Strategy A — Fast Text Extractor
Tool: pdfplumber
Triggers: native_digital + single_column layouts
Cost: Low (no ML models, pure text extraction)
"""
import time
import uuid
from pathlib import Path

import pdfplumber

from src.models import (
    DocumentProfile,
    ExtractedDocument,
    ExtractedTable,
    TextBlock,
    BoundingBox,
)
from src.utils.config import config
from .base import BaseExtractor


class FastTextExtractor(BaseExtractor):
    """
    Strategy A: Fast, cheap text extraction using pdfplumber.
    Includes multi-signal confidence scoring.
    If confidence < threshold, the ExtractionRouter escalates to Strategy B.
    """

    @property
    def strategy_name(self) -> str:
        return "A"

    def extract(
        self,
        file_path: Path,
        profile: DocumentProfile,
    ) -> ExtractedDocument:
        start = time.time()

        text_blocks: list[TextBlock] = []
        tables: list[ExtractedTable] = []
        pages_text: dict[int, str] = {}
        reading_order: list[str] = []
        confidence_signals: list[float] = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_num = page.page_number
                page_area = (page.width or 1) * (page.height or 1)

                # --- Extract text blocks ---
                text = page.extract_text() or ""
                pages_text[page_num] = text

                if text.strip():
                    block_idx = len(text_blocks)
                    text_blocks.append(TextBlock(
                        text=text,
                        bbox=BoundingBox(
                            x0=0, y0=0,
                            x1=page.width, y1=page.height,
                            page=page_num
                        ),
                        font_name=None,
                        font_size=None,
                        is_heading=False,
                    ))
                    reading_order.append(f"text:{block_idx}")

                # --- Extract tables ---
                raw_tables = page.extract_tables() or []
                for t_idx, raw_table in enumerate(raw_tables):
                    if not raw_table:
                        continue
                    table_id = f"table_{page_num}_{t_idx:02d}"
                    headers = [str(cell or "") for cell in raw_table[0]]
                    rows = [
                        [str(cell or "") for cell in row]
                        for row in raw_table[1:]
                    ]
                    tables.append(ExtractedTable(
                        table_id=table_id,
                        bbox=BoundingBox(
                            x0=0, y0=0,
                            x1=page.width, y1=page.height,
                            page=page_num
                        ),
                        headers=headers,
                        rows=rows,
                        confidence=0.7,  # pdfplumber table extraction is decent but not perfect
                    ))
                    reading_order.append(f"table:{table_id}")

                # --- Confidence signal for this page ---
                confidence_signals.append(
                    self._score_page_confidence(
                        char_count=len(text.strip()),
                        page_area=page_area,
                        image_count=len(page.images or []),
                        table_count=len(raw_tables),
                    )
                )

        overall_confidence = (
            sum(confidence_signals) / len(confidence_signals)
            if confidence_signals else 0.0
        )

        return ExtractedDocument(
            doc_id=profile.doc_id,
            filename=profile.filename,
            total_pages=profile.total_pages,
            strategy_used=self.strategy_name,
            extraction_confidence=round(overall_confidence, 4),
            cost_estimate_usd=0.0,  # Strategy A is free
            processing_time_seconds=round(time.time() - start, 2),
            text_blocks=text_blocks,
            tables=tables,
            figures=[],
            reading_order=reading_order,
            pages_text=pages_text,
        )

    # ----------------------------------------------------------
    # Confidence scoring — multi-signal
    # ----------------------------------------------------------

    def _score_page_confidence(
        self,
        char_count: int,
        page_area: float,
        image_count: int,
        table_count: int,
    ) -> float:
        """
        Compute a confidence score [0.0, 1.0] for a single page extraction.
        Weights match extraction_rules.yaml confidence_scoring section.
        """
        cfg = config

        # Signal 1: character density (chars per point of page area)
        char_density = char_count / max(page_area, 1)
        # Normalize: 0.001 chars/pt² is reasonable for a text-heavy page
        char_density_score = min(char_density / 0.001, 1.0)

        # Signal 2: image area penalty
        # We don't have image area here directly, so use image count as proxy
        image_penalty = min(image_count * 0.15, 1.0)
        image_score = 1.0 - image_penalty

        # Signal 3: font metadata (pdfplumber always has it for digital PDFs)
        # Strategy A only runs on digital PDFs so we give full score here
        font_score = 1.0

        # Signal 4: table completeness
        # If tables were found and extracted → good signal
        table_score = min(0.5 + (table_count * 0.25), 1.0) if table_count > 0 else 0.5

        # Weighted sum using config weights
        score = (
            char_density_score * cfg.get_raw("confidence_scoring", "char_density_weight")
            + image_score * cfg.get_raw("confidence_scoring", "image_area_weight")
            + font_score * cfg.get_raw("confidence_scoring", "font_metadata_weight")
            + table_score * cfg.get_raw("confidence_scoring", "table_completeness_weight")
        )

        return round(min(score, 1.0), 4)
