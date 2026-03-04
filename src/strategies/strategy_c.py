"""
Strategy C — Vision-Augmented Extractor
Tool: VLM via OpenRouter (Gemini Flash / GPT-4o-mini)
Triggers: scanned_image OR low confidence from A/B
Cost: High — budget guard enforced per document
"""
import base64
import json
import time
from pathlib import Path

import httpx
import pdfplumber
from PIL import Image
import io

from src.models import (
    DocumentProfile,
    ExtractedDocument,
    ExtractedTable,
    TextBlock,
    BoundingBox,
)
from src.utils.config import config
from .base import BaseExtractor


# Cost estimate per page for vision models (approximate)
COST_PER_PAGE_USD = 0.002  # ~$0.002 per page for Gemini Flash


class VisionExtractor(BaseExtractor):
    """
    Strategy C: Vision-augmented extraction using a multimodal LLM.
    Used for scanned documents or when A/B confidence is too low.
    Enforces a per-document cost budget cap.
    """

    @property
    def strategy_name(self) -> str:
        return "C"

    def extract(
        self,
        file_path: Path,
        profile: DocumentProfile,
    ) -> ExtractedDocument:
        start = time.time()
        cfg = config

        text_blocks: list[TextBlock] = []
        tables: list[ExtractedTable] = []
        pages_text: dict[int, str] = {}
        reading_order: list[str] = []
        total_cost = 0.0

        max_pages = min(profile.total_pages, cfg.strategy_c_max_pages)

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_num = page.page_number
                if page_num > max_pages:
                    break

                # Budget guard — stop if we exceed cost cap
                if total_cost >= cfg.strategy_c_max_cost_usd:
                    break

                # Convert page to image
                page_image_b64 = self._page_to_base64(page)
                if not page_image_b64:
                    continue

                # Call VLM
                vlm_result = self._call_vlm(page_image_b64, page_num)
                total_cost += COST_PER_PAGE_USD

                if not vlm_result:
                    continue

                # Parse VLM response
                page_text = vlm_result.get("text", "")
                pages_text[page_num] = page_text

                if page_text.strip():
                    block_idx = len(text_blocks)
                    text_blocks.append(TextBlock(
                        text=page_text,
                        bbox=BoundingBox(
                            x0=0, y0=0,
                            x1=page.width or 612,
                            y1=page.height or 792,
                            page=page_num
                        ),
                    ))
                    reading_order.append(f"text:{block_idx}")

                # Parse tables from VLM
                for t_idx, raw_table in enumerate(vlm_result.get("tables", [])):
                    table_id = f"table_{page_num}_{t_idx:02d}"
                    headers = raw_table.get("headers", [])
                    rows = raw_table.get("rows", [])
                    tables.append(ExtractedTable(
                        table_id=table_id,
                        bbox=BoundingBox(
                            x0=0, y0=0,
                            x1=page.width or 612,
                            y1=page.height or 792,
                            page=page_num
                        ),
                        headers=headers,
                        rows=rows,
                        confidence=0.80,
                    ))
                    reading_order.append(f"table:{table_id}")

        confidence = 0.80 if text_blocks else 0.2

        return ExtractedDocument(
            doc_id=profile.doc_id,
            filename=profile.filename,
            total_pages=profile.total_pages,
            strategy_used=self.strategy_name,
            extraction_confidence=confidence,
            cost_estimate_usd=round(total_cost, 4),
            processing_time_seconds=round(time.time() - start, 2),
            text_blocks=text_blocks,
            tables=tables,
            figures=[],
            reading_order=reading_order,
            pages_text=pages_text,
        )

    # ----------------------------------------------------------
    # Page to base64 image
    # ----------------------------------------------------------

    def _page_to_base64(self, page) -> str | None:
        """Convert a pdfplumber page to a base64-encoded PNG."""
        try:
            img = page.to_image(resolution=150)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception:
            return None

    # ----------------------------------------------------------
    # VLM call via OpenRouter
    # ----------------------------------------------------------

    def _call_vlm(self, image_b64: str, page_num: int) -> dict | None:
        """
        Send a page image to the VLM and parse the structured response.
        Returns: {"text": "...", "tables": [{"headers": [...], "rows": [[...]]}]}
        """
        try:
            cfg = config
            prompt = (
                "You are a document extraction expert. Analyze this document page and extract:\n"
                "1. All text content in reading order\n"
                "2. Any tables as structured data with headers and rows\n\n"
                "Respond ONLY with valid JSON in this exact format:\n"
                '{"text": "full page text here", "tables": [{"headers": ["col1", "col2"], '
                '"rows": [["val1", "val2"]]}]}\n'
                "No markdown, no explanation, just the JSON."
            )

            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {cfg.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": cfg.vision_model,
                    "max_tokens": 2000,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}"
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                },
                timeout=60.0,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            return None
