"""
Triage Agent — Stage 1 of the Document Intelligence Refinery

Classifies every document before any extraction begins.
Produces a DocumentProfile that governs all downstream decisions.
"""
from pathlib import Path

import pdfplumber

from src.models import (
    DocumentProfile,
    OriginType,
    LayoutComplexity,
    DomainHint,
    ExtractionCost,
)
from src.utils.config import config


class TriageAgent:
    """
    Classifies a document across 4 dimensions:
    1. origin_type       — digital vs scanned vs mixed
    2. layout_complexity — single column vs multi-column vs table-heavy etc.
    3. domain_hint       — financial, legal, technical, medical, general
    4. extraction_cost   — which strategy tier is needed

    Output: DocumentProfile saved to .refinery/profiles/{doc_id}.json
    """

    def __init__(self):
        self.cfg = config

    # ----------------------------------------------------------
    # Public entry point
    # ----------------------------------------------------------

    def run(self, file_path: str | Path) -> DocumentProfile:
        """
        Classify a document and return its DocumentProfile.
        Also saves the profile to .refinery/profiles/{doc_id}.json.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        doc_id = self._make_doc_id(file_path)

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            pages_data = self._extract_pages_data(pdf)

        full_text = " ".join(p["text"] for p in pages_data)

        origin_type, digital_count, scanned_count = self._detect_origin_type(pages_data)
        layout_complexity, table_count, figure_count = self._detect_layout_complexity(pages_data)
        domain_hint = self._detect_domain_hint(full_text)
        extraction_cost = self._estimate_extraction_cost(origin_type, layout_complexity)

        avg_chars = sum(p["char_count"] for p in pages_data) / max(total_pages, 1)
        avg_image_ratio = sum(p["image_area_ratio"] for p in pages_data) / max(total_pages, 1)

        profile = DocumentProfile(
            doc_id=doc_id,
            filename=file_path.name,
            file_path=str(file_path.resolve()),
            total_pages=total_pages,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            domain_hint=domain_hint,
            estimated_extraction_cost=extraction_cost,
            avg_chars_per_page=round(avg_chars, 2),
            avg_image_area_ratio=round(avg_image_ratio, 4),
            scanned_page_count=scanned_count,
            digital_page_count=digital_count,
            table_page_count=table_count,
            figure_page_count=figure_count,
        )

        self._save_profile(profile)
        return profile

    # ----------------------------------------------------------
    # Step 1 — Origin type detection
    # ----------------------------------------------------------

    def _detect_origin_type(
        self,
        pages_data: list[dict]
    ) -> tuple[OriginType, int, int]:
        """
        Analyze character density and image area per page.
        Returns: (origin_type, digital_page_count, scanned_page_count)
        """
        digital_count = 0
        scanned_count = 0

        for page in pages_data:
            is_scanned = (
                page["char_count"] < self.cfg.min_chars_per_page
                and page["image_area_ratio"] > self.cfg.max_image_area_ratio
            )
            if is_scanned:
                scanned_count += 1
            else:
                digital_count += 1

        total = len(pages_data)
        scanned_ratio = scanned_count / max(total, 1)

        if scanned_ratio >= self.cfg.scanned_page_threshold and digital_count == 0:
            origin_type = OriginType.SCANNED_IMAGE
        elif scanned_ratio >= self.cfg.scanned_page_threshold:
            origin_type = OriginType.MIXED
        else:
            origin_type = OriginType.NATIVE_DIGITAL

        return origin_type, digital_count, scanned_count

    # ----------------------------------------------------------
    # Step 2 — Layout complexity detection
    # ----------------------------------------------------------

    def _detect_layout_complexity(
        self,
        pages_data: list[dict]
    ) -> tuple[LayoutComplexity, int, int]:
        """
        Analyze column count, table presence, figure presence per page.
        Returns: (layout_complexity, table_page_count, figure_page_count)
        """
        total = max(len(pages_data), 1)
        table_pages = sum(1 for p in pages_data if p["has_tables"])
        figure_pages = sum(1 for p in pages_data if p["has_figures"])
        col_counts = [self._estimate_column_count(p["words"]) for p in pages_data]
        avg_cols = sum(col_counts) / total

        table_ratio = table_pages / total
        figure_ratio = figure_pages / total

        if table_ratio >= self.cfg.table_heavy_page_ratio and figure_ratio >= self.cfg.figure_heavy_page_ratio:
            layout = LayoutComplexity.MIXED
        elif table_ratio >= self.cfg.table_heavy_page_ratio:
            layout = LayoutComplexity.TABLE_HEAVY
        elif figure_ratio >= self.cfg.figure_heavy_page_ratio:
            layout = LayoutComplexity.FIGURE_HEAVY
        elif avg_cols >= self.cfg.multi_column_threshold:
            layout = LayoutComplexity.MULTI_COLUMN
        else:
            layout = LayoutComplexity.SINGLE_COLUMN

        return layout, table_pages, figure_pages

    # ----------------------------------------------------------
    # Step 3 — Domain hint classification
    # ----------------------------------------------------------

    def _detect_domain_hint(self, full_text: str) -> DomainHint:
        """
        Keyword-based domain classification.
        Pluggable — can be swapped for VLM classification later.
        """
        text_lower = full_text.lower()
        scores: dict[str, int] = {}

        for domain, keywords in self.cfg.domain_keywords.items():
            scores[domain] = sum(1 for kw in keywords if kw.lower() in text_lower)

        best_domain = max(scores, key=lambda d: scores[d])

        if scores[best_domain] == 0:
            return DomainHint.GENERAL

        try:
            return DomainHint(best_domain)
        except ValueError:
            return DomainHint.GENERAL

    # ----------------------------------------------------------
    # Step 4 — Extraction cost estimation
    # ----------------------------------------------------------

    def _estimate_extraction_cost(
        self,
        origin_type: OriginType,
        layout_complexity: LayoutComplexity,
    ) -> ExtractionCost:
        """Map origin + layout → which strategy tier is needed."""
        if origin_type == OriginType.SCANNED_IMAGE:
            return ExtractionCost.NEEDS_VISION_MODEL

        if layout_complexity in (
            LayoutComplexity.MULTI_COLUMN,
            LayoutComplexity.TABLE_HEAVY,
            LayoutComplexity.FIGURE_HEAVY,
            LayoutComplexity.MIXED,
        ):
            return ExtractionCost.NEEDS_LAYOUT_MODEL

        if origin_type == OriginType.MIXED:
            return ExtractionCost.NEEDS_LAYOUT_MODEL

        return ExtractionCost.FAST_TEXT_SUFFICIENT

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _extract_pages_data(self, pdf: pdfplumber.PDF) -> list[dict]:
        """
        Extract raw measurements from each page.
        """
        pages_data = []
        for page in pdf.pages:
            width = page.width or 1
            height = page.height or 1
            page_area = width * height

            text = page.extract_text() or ""
            char_count = len(text.strip())

            images = page.images or []
            image_area = sum(
                (img.get("width", 0) * img.get("height", 0))
                for img in images
            )
            image_area_ratio = min(image_area / page_area, 1.0)

            tables = page.extract_tables() or []
            has_tables = len(tables) > 0
            has_figures = len(images) > 0
            words = page.extract_words() or []

            pages_data.append({
                "page_number": page.page_number,
                "char_count": char_count,
                "page_area": page_area,
                "image_area_ratio": image_area_ratio,
                "has_tables": has_tables,
                "has_figures": has_figures,
                "words": words,
                "text": text,
            })
        return pages_data

    def _estimate_column_count(self, words: list[dict]) -> int:
        """Estimate number of columns by clustering word x-coordinates."""
        if not words:
            return 1
        x_centers = [(w["x0"] + w["x1"]) / 2 for w in words]
        x_min, x_max = min(x_centers), max(x_centers)
        mid = (x_min + x_max) / 2
        left = sum(1 for x in x_centers if x < mid)
        right = sum(1 for x in x_centers if x >= mid)
        total = len(x_centers)
        left_ratio = left / total
        right_ratio = right / total
        if 0.25 <= left_ratio <= 0.75 and 0.25 <= right_ratio <= 0.75:
            return 2
        return 1

    def _make_doc_id(self, file_path: Path) -> str:
        """Generate a stable doc_id from the filename."""
        return file_path.stem.lower().replace(" ", "_")

    def _save_profile(self, profile: DocumentProfile) -> None:
        """Persist profile to .refinery/profiles/{doc_id}.json"""
        out_path = self.cfg.profiles_dir / f"{profile.doc_id}.json"
        with open(out_path, "w") as f:
            f.write(profile.model_dump_json(indent=2))
