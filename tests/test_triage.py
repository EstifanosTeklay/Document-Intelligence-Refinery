"""
Unit tests for TriageAgent and extraction confidence scoring.
Run with: pytest tests/ -v
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models import OriginType, LayoutComplexity, DomainHint, ExtractionCost
from src.agents.triage import TriageAgent
from src.strategies.strategy_a import FastTextExtractor


# ----------------------------------------------------------
# TriageAgent — origin type detection
# ----------------------------------------------------------

class TestOriginTypeDetection:

    def setup_method(self):
        self.agent = TriageAgent()

    def test_all_digital_pages(self):
        pages_data = [
            {"char_count": 500, "image_area_ratio": 0.05} for _ in range(10)
        ]
        origin, digital, scanned = self.agent._detect_origin_type(pages_data)
        assert origin == OriginType.NATIVE_DIGITAL
        assert digital == 10
        assert scanned == 0

    def test_all_scanned_pages(self):
        pages_data = [
            {"char_count": 10, "image_area_ratio": 0.95} for _ in range(10)
        ]
        origin, digital, scanned = self.agent._detect_origin_type(pages_data)
        assert origin == OriginType.SCANNED_IMAGE
        assert scanned == 10

    def test_mixed_pages(self):
        pages_data = (
            [{"char_count": 10, "image_area_ratio": 0.95}] * 4 +
            [{"char_count": 500, "image_area_ratio": 0.05}] * 6
        )
        origin, digital, scanned = self.agent._detect_origin_type(pages_data)
        assert origin == OriginType.MIXED
        assert digital == 6
        assert scanned == 4

    def test_mostly_digital_not_mixed(self):
        pages_data = (
            [{"char_count": 10, "image_area_ratio": 0.95}] * 1 +
            [{"char_count": 500, "image_area_ratio": 0.05}] * 9
        )
        origin, _, _ = self.agent._detect_origin_type(pages_data)
        assert origin == OriginType.NATIVE_DIGITAL

    # --- Edge cases ---

    def test_blank_page_zero_chars_zero_images_is_scanned(self):
        """A page with nothing on it should be treated as scanned."""
        pages_data = [{"char_count": 0, "image_area_ratio": 0.0}] * 5
        origin, digital, scanned = self.agent._detect_origin_type(pages_data)
        assert scanned == 5

    def test_single_digital_page(self):
        """Single page document — digital."""
        pages_data = [{"char_count": 300, "image_area_ratio": 0.05}]
        origin, digital, scanned = self.agent._detect_origin_type(pages_data)
        assert origin == OriginType.NATIVE_DIGITAL
        assert digital == 1

    def test_single_scanned_page(self):
        """Single page document — scanned."""
        pages_data = [{"char_count": 5, "image_area_ratio": 0.90}]
        origin, digital, scanned = self.agent._detect_origin_type(pages_data)
        assert origin == OriginType.SCANNED_IMAGE
        assert scanned == 1


# ----------------------------------------------------------
# TriageAgent — layout complexity detection
# ----------------------------------------------------------

class TestLayoutComplexityDetection:

    def setup_method(self):
        self.agent = TriageAgent()

    def _make_pages(self, n, has_tables=False, has_figures=False, words=None, char_count=200):
        return [
            {
                "has_tables": has_tables,
                "has_figures": has_figures,
                "words": words or [],
                "char_count": char_count,
            }
            for _ in range(n)
        ]

    def test_single_column_no_tables(self):
        pages = self._make_pages(10)
        layout, t_count, f_count = self.agent._detect_layout_complexity(pages)
        assert layout == LayoutComplexity.SINGLE_COLUMN
        assert t_count == 0

    def test_table_heavy(self):
        pages = (
            self._make_pages(4, has_tables=True) +
            self._make_pages(6, has_tables=False)
        )
        layout, t_count, _ = self.agent._detect_layout_complexity(pages)
        assert layout == LayoutComplexity.TABLE_HEAVY
        assert t_count == 4

    def test_figure_heavy(self):
        pages = (
            self._make_pages(3, has_figures=True) +
            self._make_pages(7, has_figures=False)
        )
        layout, _, f_count = self.agent._detect_layout_complexity(pages)
        assert layout == LayoutComplexity.FIGURE_HEAVY

    def test_all_scanned_pages_defaults_to_single_column(self):
        """When all pages are scanned (no content), layout should default gracefully."""
        pages = self._make_pages(10, char_count=0)
        layout, _, _ = self.agent._detect_layout_complexity(pages)
        assert layout == LayoutComplexity.SINGLE_COLUMN


# ----------------------------------------------------------
# TriageAgent — domain hint detection
# ----------------------------------------------------------

class TestDomainHintDetection:

    def setup_method(self):
        self.agent = TriageAgent()

    def test_financial_domain(self):
        text = "The balance sheet shows revenue and income statement for fiscal year ended June 2024."
        assert self.agent._detect_domain_hint(text) == DomainHint.FINANCIAL

    def test_legal_domain(self):
        text = "Pursuant to the agreement, jurisdiction and liability are hereby established."
        assert self.agent._detect_domain_hint(text) == DomainHint.LEGAL

    def test_general_domain_no_keywords(self):
        text = "The quick brown fox jumped over the lazy dog."
        assert self.agent._detect_domain_hint(text) == DomainHint.GENERAL

    # --- Edge cases ---

    def test_empty_text_returns_general(self):
        """Scanned documents produce no text — should not crash."""
        assert self.agent._detect_domain_hint("") == DomainHint.GENERAL

    def test_whitespace_only_returns_general(self):
        assert self.agent._detect_domain_hint("   \n\t  ") == DomainHint.GENERAL


# ----------------------------------------------------------
# TriageAgent — extraction cost estimation
# ----------------------------------------------------------

class TestExtractionCostEstimation:

    def setup_method(self):
        self.agent = TriageAgent()

    def test_scanned_always_needs_vision(self):
        cost = self.agent._estimate_extraction_cost(
            OriginType.SCANNED_IMAGE, LayoutComplexity.SINGLE_COLUMN
        )
        assert cost == ExtractionCost.NEEDS_VISION_MODEL

    def test_digital_single_column_is_fast(self):
        cost = self.agent._estimate_extraction_cost(
            OriginType.NATIVE_DIGITAL, LayoutComplexity.SINGLE_COLUMN
        )
        assert cost == ExtractionCost.FAST_TEXT_SUFFICIENT

    def test_digital_table_heavy_needs_layout(self):
        cost = self.agent._estimate_extraction_cost(
            OriginType.NATIVE_DIGITAL, LayoutComplexity.TABLE_HEAVY
        )
        assert cost == ExtractionCost.NEEDS_LAYOUT_MODEL

    def test_mixed_origin_needs_layout(self):
        cost = self.agent._estimate_extraction_cost(
            OriginType.MIXED, LayoutComplexity.SINGLE_COLUMN
        )
        assert cost == ExtractionCost.NEEDS_LAYOUT_MODEL

    # --- Edge cases ---

    def test_form_fillable_needs_layout(self):
        """Form-fillable PDFs need structured extraction even if single-column."""
        cost = self.agent._estimate_extraction_cost(
            OriginType.FORM_FILLABLE, LayoutComplexity.SINGLE_COLUMN
        )
        assert cost == ExtractionCost.NEEDS_LAYOUT_MODEL

    def test_scanned_multi_column_still_vision(self):
        """Scanned always wins over layout complexity."""
        cost = self.agent._estimate_extraction_cost(
            OriginType.SCANNED_IMAGE, LayoutComplexity.MULTI_COLUMN
        )
        assert cost == ExtractionCost.NEEDS_VISION_MODEL


# ----------------------------------------------------------
# TriageAgent — run() edge cases (file-level)
# ----------------------------------------------------------

class TestTriageRunEdgeCases:

    def setup_method(self):
        self.agent = TriageAgent()

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Document not found"):
            self.agent.run("/nonexistent/path/document.pdf")

    def test_unsupported_extension_raises_value_error(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            f.write(b"fake content")
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                self.agent.run(tmp_path)
        finally:
            os.unlink(tmp_path)


# ----------------------------------------------------------
# TriageAgent — column count estimation edge cases
# ----------------------------------------------------------

class TestColumnCountEdgeCases:

    def setup_method(self):
        self.agent = TriageAgent()

    def test_no_words_returns_1(self):
        assert self.agent._estimate_column_count([]) == 1

    def test_single_word_returns_1(self):
        words = [{"x0": 100, "x1": 200}]
        assert self.agent._estimate_column_count(words) == 1

    def test_all_words_same_x_returns_1(self):
        """Edge case: all words at identical x — no column split possible."""
        words = [{"x0": 100, "x1": 110}] * 10
        assert self.agent._estimate_column_count(words) == 1

    def test_negligible_page_width_returns_1(self):
        """Edge case: x_max - x_min < 10 — too narrow to split."""
        words = [{"x0": 100, "x1": 105}, {"x0": 101, "x1": 106}]
        assert self.agent._estimate_column_count(words) == 1

    def test_two_column_layout_detected(self):
        """Words split evenly left/right should detect 2 columns."""
        left_words = [{"x0": 50 + i, "x1": 100 + i} for i in range(10)]
        right_words = [{"x0": 350 + i, "x1": 400 + i} for i in range(10)]
        words = left_words + right_words
        assert self.agent._estimate_column_count(words) == 2


# ----------------------------------------------------------
# Strategy A — confidence scoring
# ----------------------------------------------------------

class TestStrategyAConfidence:

    def setup_method(self):
        self.extractor = FastTextExtractor()

    def test_high_confidence_text_page(self):
        score = self.extractor._score_page_confidence(
            char_count=800,
            page_area=595 * 842,
            image_count=0,
            table_count=1,
        )
        assert score > 0.7, f"Expected high confidence, got {score}"

    def test_low_confidence_image_heavy_page(self):
        score = self.extractor._score_page_confidence(
            char_count=5,
            page_area=595 * 842,
            image_count=5,
            table_count=0,
        )
        assert score < 0.6, f"Expected low confidence, got {score}"

    def test_zero_char_zero_image_very_low(self):
        """Completely empty page should have near-zero confidence."""
        score = self.extractor._score_page_confidence(
            char_count=0,
            page_area=595 * 842,
            image_count=0,
            table_count=0,
        )
        assert score < 0.5
