"""
Unit tests for ExtractionRouter — profile-based selection
and confidence-gated escalation.
Run with: pytest tests/ -v
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.models import (
    DocumentProfile, ExtractedDocument,
    OriginType, LayoutComplexity, DomainHint, ExtractionCost,
)
from src.models.routing import RoutingDecision, StrategyAttempt
from src.agents.extractor import ExtractionRouter


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def make_profile(cost: ExtractionCost) -> DocumentProfile:
    return DocumentProfile(
        doc_id="test_doc",
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        total_pages=10,
        origin_type=OriginType.NATIVE_DIGITAL,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        domain_hint=DomainHint.GENERAL,
        estimated_extraction_cost=cost,
    )


def make_extracted(strategy: str, confidence: float) -> ExtractedDocument:
    return ExtractedDocument(
        doc_id="test_doc",
        filename="test.pdf",
        total_pages=10,
        strategy_used=strategy,
        extraction_confidence=confidence,
        cost_estimate_usd=0.0,
        processing_time_seconds=0.1,
    )


# ----------------------------------------------------------
# Profile-based initial strategy selection
# ----------------------------------------------------------

class TestInitialStrategySelection:

    def setup_method(self):
        self.router = ExtractionRouter()

    def test_fast_text_sufficient_selects_a(self):
        profile = make_profile(ExtractionCost.FAST_TEXT_SUFFICIENT)
        assert self.router._select_initial_strategy(profile) == "A"

    def test_needs_layout_selects_b(self):
        profile = make_profile(ExtractionCost.NEEDS_LAYOUT_MODEL)
        assert self.router._select_initial_strategy(profile) == "B"

    def test_needs_vision_selects_c(self):
        profile = make_profile(ExtractionCost.NEEDS_VISION_MODEL)
        assert self.router._select_initial_strategy(profile) == "C"


# ----------------------------------------------------------
# Confidence-gated escalation
# ----------------------------------------------------------

class TestConfidenceGatedEscalation:

    def setup_method(self):
        self.router = ExtractionRouter()
        self.profile = make_profile(ExtractionCost.FAST_TEXT_SUFFICIENT)

    def test_strategy_a_passes_no_escalation(self):
        """A returns high confidence → no escalation."""
        self.router.strategy_a.extract = MagicMock(
            return_value=make_extracted("A", confidence=0.90)
        )
        _, routing = self.router._run_with_escalation(
            Path("/tmp/test.pdf"), self.profile, "A"
        )
        assert routing.final_strategy == "A"
        assert routing.escalation_occurred is False
        assert routing.total_attempts == 1
        assert routing.attempts[0].passed is True

    def test_strategy_a_low_confidence_escalates_to_b(self):
        """A returns low confidence → escalates to B which passes."""
        self.router.strategy_a.extract = MagicMock(
            return_value=make_extracted("A", confidence=0.50)
        )
        self.router.strategy_b.extract = MagicMock(
            return_value=make_extracted("B", confidence=0.80)
        )
        _, routing = self.router._run_with_escalation(
            Path("/tmp/test.pdf"), self.profile, "A"
        )
        assert routing.final_strategy == "B"
        assert routing.escalation_occurred is True
        assert routing.total_attempts == 2
        assert routing.attempts[0].passed is False
        assert routing.attempts[1].passed is True

    def test_a_and_b_low_confidence_escalates_to_c(self):
        """A and B both fail → falls through to C."""
        self.router.strategy_a.extract = MagicMock(
            return_value=make_extracted("A", confidence=0.30)
        )
        self.router.strategy_b.extract = MagicMock(
            return_value=make_extracted("B", confidence=0.40)
        )
        self.router.strategy_c.extract = MagicMock(
            return_value=make_extracted("C", confidence=0.85)
        )
        _, routing = self.router._run_with_escalation(
            Path("/tmp/test.pdf"), self.profile, "A"
        )
        assert routing.final_strategy == "C"
        assert routing.escalation_occurred is True
        assert routing.total_attempts == 3
        assert routing.escalation_path == "A → B → C"

    def test_start_at_b_passes_no_escalation(self):
        """Profile says B → B passes → no escalation to C."""
        profile = make_profile(ExtractionCost.NEEDS_LAYOUT_MODEL)
        self.router.strategy_b.extract = MagicMock(
            return_value=make_extracted("B", confidence=0.75)
        )
        _, routing = self.router._run_with_escalation(
            Path("/tmp/test.pdf"), profile, "B"
        )
        assert routing.final_strategy == "B"
        assert routing.total_attempts == 1
        assert routing.escalation_occurred is False

    def test_start_at_b_low_confidence_escalates_to_c(self):
        """Profile says B → B fails → escalates to C."""
        profile = make_profile(ExtractionCost.NEEDS_LAYOUT_MODEL)
        self.router.strategy_b.extract = MagicMock(
            return_value=make_extracted("B", confidence=0.30)
        )
        self.router.strategy_c.extract = MagicMock(
            return_value=make_extracted("C", confidence=0.80)
        )
        _, routing = self.router._run_with_escalation(
            Path("/tmp/test.pdf"), profile, "B"
        )
        assert routing.final_strategy == "C"
        assert routing.escalation_occurred is True
        assert routing.escalation_path == "B → C"

    def test_start_at_c_always_passes(self):
        """C is the final fallback — always passes regardless of confidence."""
        profile = make_profile(ExtractionCost.NEEDS_VISION_MODEL)
        self.router.strategy_c.extract = MagicMock(
            return_value=make_extracted("C", confidence=0.20)
        )
        _, routing = self.router._run_with_escalation(
            Path("/tmp/test.pdf"), profile, "C"
        )
        assert routing.final_strategy == "C"
        assert routing.total_attempts == 1
        assert routing.attempts[0].passed is True


# ----------------------------------------------------------
# RoutingDecision metadata
# ----------------------------------------------------------

class TestRoutingDecisionMetadata:

    def setup_method(self):
        self.router = ExtractionRouter()

    def test_routing_decision_attached_to_result(self):
        """run() must attach routing_decision to the ExtractedDocument."""
        profile = make_profile(ExtractionCost.FAST_TEXT_SUFFICIENT)
        self.router.strategy_a.extract = MagicMock(
            return_value=make_extracted("A", confidence=0.90)
        )
        with patch.object(self.router, '_write_ledger'):
            result = self.router.run(Path("/tmp/test.pdf"), profile)

        assert result.routing_decision is not None
        assert result.routing_decision["final_strategy"] == "A"
        assert result.routing_decision["initial_strategy_selected"] == "A"
        assert result.routing_decision["escalation_occurred"] is False

    def test_escalation_reason_recorded(self):
        """Failed attempts must record why they were rejected."""
        profile = make_profile(ExtractionCost.FAST_TEXT_SUFFICIENT)
        self.router.strategy_a.extract = MagicMock(
            return_value=make_extracted("A", confidence=0.50)
        )
        self.router.strategy_b.extract = MagicMock(
            return_value=make_extracted("B", confidence=0.80)
        )
        _, routing = self.router._run_with_escalation(
            Path("/tmp/test.pdf"), profile, "A"
        )
        failed_attempt = routing.attempts[0]
        assert failed_attempt.escalation_reason is not None
        assert "0.500" in failed_attempt.escalation_reason
        assert "escalating" in failed_attempt.escalation_reason.lower()

    def test_invalid_strategy_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            self.router._get_extractor("Z")

    def test_escalation_path_single_strategy(self):
        attempts = [StrategyAttempt(
            strategy="B", confidence_achieved=0.80,
            confidence_threshold=0.60, passed=True
        )]
        routing = RoutingDecision(
            doc_id="x", profile_origin_type="native_digital",
            profile_layout_complexity="table_heavy",
            profile_estimated_cost="needs_layout_model",
            initial_strategy_selected="B",
            attempts=attempts,
            final_strategy="B",
        )
        assert routing.escalation_path == "B"

    def test_escalation_path_full_chain(self):
        attempts = [
            StrategyAttempt(strategy="A", confidence_achieved=0.3, confidence_threshold=0.75, passed=False),
            StrategyAttempt(strategy="B", confidence_achieved=0.4, confidence_threshold=0.60, passed=False),
            StrategyAttempt(strategy="C", confidence_achieved=0.8, confidence_threshold=0.0, passed=True),
        ]
        routing = RoutingDecision(
            doc_id="x", profile_origin_type="native_digital",
            profile_layout_complexity="single_column",
            profile_estimated_cost="fast_text_sufficient",
            initial_strategy_selected="A",
            attempts=attempts,
            final_strategy="C",
            escalation_occurred=True,
            total_attempts=3,
        )
        assert routing.escalation_path == "A → B → C"
