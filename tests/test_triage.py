"""
Unit tests for TriageAgent and extraction confidence scoring.
Run with: pytest tests/ -v
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

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
        # 4 scanned out of 10 = 40% > 30% threshold → MIXED (has digital pages too)
        pages_data = (
            [{"char_count": 10, "image_area_ratio": 0.95}] * 4 +
            [{"char_count": 500, "image_area_ratio": 0.05}] * 6
        )
        origin, digital, scanned = self.agent._detect_origin_type(pages_data)
        assert origin == OriginType.MIXED
        assert digital == 6
        assert scanned == 4

    def test_mostly_digital_not_mixed(self):
        # 1 scanned out of 10 = 10% < 30% threshold → NATIVE_DIGITAL
        pages_data = (
            [{"char_count": 10, "image_area_ratio": 0.95}] * 1 +
            [{"char_count": 500, "image_area_ratio": 0.05}] * 9
        )
        origin, _, _ = self.agent._detect_origin_type(pages_data)
        assert origin == OriginType.NATIVE_DIGITAL


# ----------------------------------------------------------
# TriageAgent — layout complexity detection
# ----------------------------------------------------------

class TestLayoutComplexityDetection:

    def setup_method(self):
        self.agent = TriageAgent()

    def _make_pages(self, n, has_tables=False, has_figures=False, words=None):
        return [
            {
                "has_tables": has_tables,
                "has_figures": has_figures,
                "words": words or [],
            }
            for _ in range(n)
        ]

    def test_single_column_no_tables(self):
        pages = self._make_pages(10, has_tables=False, has_figures=False)
        layout, t_count, f_count = self.agent._detect_layout_complexity(pages)
        assert layout == LayoutComplexity.SINGLE_COLUMN
        assert t_count == 0

    def test_table_heavy(self):
        # 4 out of 10 pages have tables = 40% > 30% threshold
        pages = (
            self._make_pages(4, has_tables=True) +
            self._make_pages(6, has_tables=False)
        )
        layout, t_count, f_count = self.agent._detect_layout_complexity(pages)
        assert layout == LayoutComplexity.TABLE_HEAVY
        assert t_count == 4

    def test_figure_heavy(self):
        # 3 out of 10 pages have figures = 30% >= 25% threshold
        pages = (
            self._make_pages(3, has_figures=True) +
            self._make_pages(7, has_figures=False)
        )
        layout, t_count, f_count = self.agent._detect_layout_complexity(pages)
        assert layout == LayoutComplexity.FIGURE_HEAVY


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


# ----------------------------------------------------------
# Strategy A — confidence scoring
# ----------------------------------------------------------

class TestStrategyAConfidence:

    def setup_method(self):
        self.extractor = FastTextExtractor()

    def test_high_confidence_text_page(self):
        score = self.extractor._score_page_confidence(
            char_count=800,
            page_area=595 * 842,  # A4 in points
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
