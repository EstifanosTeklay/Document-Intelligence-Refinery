"""
RoutingDecision — typed metadata produced by the ExtractionRouter.

This model makes the router's decisions explicit and auditable.
Every escalation attempt is recorded here, proving the router
orchestrated profile-based selection and confidence-gated escalation.
"""
from typing import Optional
from pydantic import BaseModel, Field


class StrategyAttempt(BaseModel):
    """
    A single strategy attempt within one routing decision.
    Records what was tried, what confidence was achieved, and why it escalated.
    """
    strategy: str = Field(..., description="A | B | C")
    confidence_achieved: float = Field(..., ge=0.0, le=1.0)
    confidence_threshold: float = Field(..., description="Threshold required to pass")
    passed: bool = Field(..., description="True if confidence >= threshold")
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Why this strategy was rejected. None if it passed."
    )
    cost_usd: float = Field(default=0.0)
    processing_time_seconds: float = Field(default=0.0)


class RoutingDecision(BaseModel):
    """
    The complete routing decision record for one document.

    Exposes:
    - Which strategy was selected from the DocumentProfile
    - Every escalation attempt with confidence scores
    - The final strategy that produced the accepted ExtractedDocument
    - Why each intermediate strategy was rejected

    Stored inside ExtractedDocument.routing_decision and
    also written to extraction_ledger.jsonl.
    """
    doc_id: str
    profile_origin_type: str
    profile_layout_complexity: str
    profile_estimated_cost: str

    # The strategy the profile recommended as starting point
    initial_strategy_selected: str = Field(
        ...,
        description="Strategy selected from DocumentProfile: A | B | C"
    )

    # Full escalation trail — all attempts in order
    attempts: list[StrategyAttempt] = Field(
        default_factory=list,
        description="All strategy attempts in order. Length > 1 means escalation occurred."
    )

    # Final outcome
    final_strategy: str = Field(
        ...,
        description="Strategy whose output was accepted as the final ExtractedDocument"
    )
    escalation_occurred: bool = Field(
        default=False,
        description="True if the router escalated beyond the initial strategy"
    )
    total_attempts: int = Field(default=1)
    total_cost_usd: float = Field(default=0.0)

    @property
    def escalation_path(self) -> str:
        """Human-readable escalation path. E.g. 'A → B → C' or 'B → C'."""
        strategies = [a.strategy for a in self.attempts]
        return " → ".join(strategies) if len(strategies) > 1 else strategies[0]
