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


# Supported file extensions — extend here when adding new format support
SUPPORTED_EXTENSIONS = {".pdf"}


class TriageAgent:
    """
    Classifies a document across 4 dimensions:
    1. origin_type       — digital vs scanned vs mixed
    2. layout_complexity — single column vs multi-column vs table-heavy etc.
    3. domain_hint       — financial, legal, technical, medical, general
    4. extraction_cost   — which strategy tier is needed

    Edge cases handled:
    - Non-existent file         → FileNotFoundError with clear message
    - Unsupported file type     → ValueError listing supported types
    - Password-protected PDF    → graceful fallback to SCANNED_IMAGE + triage_notes
    - Empty document (0 pages)  → ValueError
    - Single-page document      → handled correctly (no division-by-zero)
    - Pages with no text/images → classified as scanned individually
    - Corrupt/unreadable PDF    → RuntimeError with original exception chained

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

        # --- Edge case: file does not exist ---
        if not file_path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}. "
                f"Check the path and ensure the file has been uploaded to data/corpus/."
            )

        # --- Edge case: unsupported file type ---
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: '{file_path.suffix}'. "
                f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}. "
                f"Convert the document to PDF before running the refinery."
            )

        doc_id = self._make_doc_id(file_path)

        # --- Edge case: password-protected or corrupt PDF ---
        try:
            with pdfplumber.open(file_path) as pdf:

                # --- Edge case: empty document ---
                if len(pdf.pages) == 0:
                    raise ValueError(
                        f"Document '{file_path.name}' has 0 pages. "
                        f"The file may be corrupt or an empty placeholder."
                    )

                total_pages = len(pdf.pages)
                pages_data = self._extract_pages_data(pdf)

        except ValueError:
            raise  # re-raise our own ValueError unchanged

        except Exception as exc:
            # Catch pdfplumber errors — most commonly password protection
            error_msg = str(exc).lower()
            if "password" in error_msg or "encrypted" in error_msg:
                # Graceful fallback: return a profile flagging the issue
                profile = DocumentProfile(
                    doc_id=doc_id,
                    filename=file_path.name,
                    file_path=str(file_path.resolve()),
                    total_pages=0,
                    origin_type=OriginType.SCANNED_IMAGE,
                    layout_complexity=LayoutComplexity.SINGLE_COLUMN,
                    domain_hint=DomainHint.GENERAL,
                    estimated_extraction_cost=ExtractionCost.NEEDS_VISION_MODEL,
                    triage_confidence=0.0,
                    triage_notes=(
                        "PASSWORD_PROTECTED: Document is encrypted. "
                        "Decrypt the PDF before running the refinery. "
                        "Routed to Strategy C as a conservative fallback."
                    ),
                )
                self._save_profile(profile)
                return profile
            raise RuntimeError(
                f"Failed to open '{file_path.name}': {exc}. "
                f"The file may be corrupt. Verify it opens correctly in a PDF viewer."
            ) from exc

        # --- Normal classification flow ---
        full_text = " ".join(p["text"] for p in pages_data)

        origin_type, digital_count, scanned_count = self._detect_origin_type(pages_data)
        layout_complexity, table_count, figure_count = self._detect_layout_complexity(pages_data)
        domain_hint = self._detect_domain_hint(full_text)
        extraction_cost = self._estimate_extraction_cost(origin_type, layout_complexity)

        # --- Single-page edge case: note in triage_notes ---
        triage_notes = None
        if total_pages == 1:
            triage_notes = (
                "SINGLE_PAGE: Document has only 1 page. "
                "Layout complexity detection may be less reliable — "
                "column heuristics require multiple pages for stable estimation."
            )

        # --- Edge case: no text and no images on any page ---
        avg_chars = sum(p["char_count"] for p in pages_data) / max(total_pages, 1)
        avg_image_ratio = sum(p["image_area_ratio"] for p in pages_data) / max(total_pages, 1)

        if avg_chars == 0 and avg_image_ratio == 0:
            triage_notes = (
                "EMPTY_CONTENT: No text or images detected on any page. "
                "Document may use non-standard encoding or be a blank template. "
                "Routed to Strategy C for visual inspection."
            )
            extraction_cost = ExtractionCost.NEEDS_VISION_MODEL
            origin_type = OriginType.SCANNED_IMAGE

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
            triage_notes=triage_notes,
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

        Edge cases:
        - Pages with zero chars AND zero images → classified as scanned
          (blank pages or non-standard encoding)
        - Single-page documents → single page determines origin type
        """
        digital_count = 0
        scanned_count = 0

        for page in pages_data:
            char_count = page["char_count"]
            image_ratio = page["image_area_ratio"]

            # A page is scanned if it has few characters AND significant image area
            # OR if it has zero of both (blank / non-standard)
            is_scanned = (
                char_count < self.cfg.min_chars_per_page
                and image_ratio > self.cfg.max_image_area_ratio
            ) or (
                char_count == 0 and image_ratio == 0
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

        Edge cases:
        - All pages empty (scanned) → defaults to SINGLE_COLUMN
        - Pages with only images, no words → column count defaults to 1
        """
        # Only compute layout on pages that have content
        content_pages = [p for p in pages_data if p["char_count"] > 0 or p["has_tables"]]

        # Edge case: all pages are scanned / empty
        if not content_pages:
            total_all = max(len(pages_data), 1)
            table_pages = sum(1 for p in pages_data if p["has_tables"])
            figure_pages = sum(1 for p in pages_data if p["has_figures"])
            return LayoutComplexity.SINGLE_COLUMN, table_pages, figure_pages

        total = max(len(content_pages), 1)
        table_pages = sum(1 for p in pages_data if p["has_tables"])
        figure_pages = sum(1 for p in pages_data if p["has_figures"])

        col_counts = [self._estimate_column_count(p["words"]) for p in content_pages]
        avg_cols = sum(col_counts) / total

        table_ratio = table_pages / max(len(pages_data), 1)
        figure_ratio = figure_pages / max(len(pages_data), 1)

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

        Edge case: empty text (scanned doc) → returns GENERAL
        """
        if not full_text.strip():
            return DomainHint.GENERAL

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
        """
        Map origin + layout → which strategy tier is needed.

        Edge cases:
        - FORM_FILLABLE: treated as native_digital but routed to layout model
          because form fields require structured extraction
        """
        if origin_type == OriginType.SCANNED_IMAGE:
            return ExtractionCost.NEEDS_VISION_MODEL

        if origin_type == OriginType.FORM_FILLABLE:
            return ExtractionCost.NEEDS_LAYOUT_MODEL

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
        Handles per-page errors gracefully — a single corrupt page
        does not abort the entire document.
        """
        pages_data = []
        for page in pdf.pages:
            try:
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
                    "error": None,
                })

            except Exception as exc:
                # Log corrupt page but continue processing remaining pages
                pages_data.append({
                    "page_number": page.page_number,
                    "char_count": 0,
                    "page_area": 1,
                    "image_area_ratio": 0.0,
                    "has_tables": False,
                    "has_figures": False,
                    "words": [],
                    "text": "",
                    "error": str(exc),
                })

        return pages_data

    def _estimate_column_count(self, words: list[dict]) -> int:
        """
        Estimate number of columns by clustering word x-coordinates.
        Edge case: single word or all words on same x → returns 1.
        """
        if not words:
            return 1

        x_centers = [(w["x0"] + w["x1"]) / 2 for w in words]

        # Edge case: all words at same x position
        if len(set(x_centers)) <= 1:
            return 1

        x_min, x_max = min(x_centers), max(x_centers)

        # Edge case: page width is negligible
        if x_max - x_min < 10:
            return 1

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
