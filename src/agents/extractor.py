"""
Extraction Router — Stage 2 of the Document Intelligence Refinery

Orchestrates profile-based strategy selection and confidence-gated
escalation across all three extraction strategies.

Routing logic (explicit):
  1. Read DocumentProfile.estimated_extraction_cost → select starting strategy
  2. Run selected strategy → measure extraction_confidence
  3. If confidence < threshold → escalate to next strategy tier
  4. Repeat until confidence passes OR Strategy C (final fallback) is reached
  5. Attach RoutingDecision to the final ExtractedDocument
  6. Write full ledger entry to .refinery/extraction_ledger.jsonl
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from src.models import DocumentProfile, ExtractedDocument, ExtractionCost
from src.models.routing import RoutingDecision, StrategyAttempt
from src.strategies.strategy_a import FastTextExtractor
from src.strategies.strategy_b import LayoutExtractor
from src.strategies.strategy_c import VisionExtractor
from src.utils.config import config


class ExtractionRouter:
    """
    Orchestrates document extraction across Strategy A, B, and C.

    Key behaviours:
    - Profile-based selection: DocumentProfile.estimated_extraction_cost
      determines the STARTING strategy — never blindly defaults to A.
    - Confidence-gated escalation: each strategy must meet its threshold
      or the router escalates automatically to the next tier.
    - Full audit trail: every attempt (including failed ones) recorded
      in RoutingDecision and written to extraction_ledger.jsonl.
    - Routing metadata exposed: RoutingDecision attached to the returned
      ExtractedDocument so downstream stages can inspect routing history.

    Escalation chain:
        A (threshold 0.75) → B (threshold 0.60) → C (final fallback)
    """

    def __init__(self):
        self.cfg = config
        self.strategy_a = FastTextExtractor()
        self.strategy_b = LayoutExtractor()
        self.strategy_c = VisionExtractor()

        # Thresholds from config — never hardcoded
        self._thresholds = {
            "A": self.cfg.strategy_a_min_confidence,
            "B": self.cfg.strategy_b_min_confidence,
            "C": 0.0,  # C is the final fallback — always passes
        }

        # Escalation chain — order matters
        self._chain = ["A", "B", "C"]

    # ----------------------------------------------------------
    # Public entry point
    # ----------------------------------------------------------

    def run(
        self,
        file_path: str | Path,
        profile: DocumentProfile,
    ) -> ExtractedDocument:
        """
        Route a document through extraction with confidence-gated escalation.

        Args:
            file_path: Path to the source document
            profile:   DocumentProfile produced by TriageAgent

        Returns:
            ExtractedDocument with routing_decision attached,
            reflecting all strategies attempted and the final outcome.
        """
        file_path = Path(file_path)

        # Step 1: Select starting strategy from DocumentProfile
        initial_strategy = self._select_initial_strategy(profile)

        # Step 2: Run with escalation, collecting full attempt trail
        result, routing = self._run_with_escalation(
            file_path, profile, initial_strategy
        )

        # Step 3: Attach routing metadata to result
        result.routing_decision = routing.model_dump()

        # Step 4: Write full ledger entry
        self._write_ledger(profile, result, routing)

        return result

    # ----------------------------------------------------------
    # Step 1: Profile-based strategy selection
    # ----------------------------------------------------------

    def _select_initial_strategy(self, profile: DocumentProfile) -> str:
        """
        Map DocumentProfile.estimated_extraction_cost → starting strategy.

        This is the explicit profile-based selection the router performs
        BEFORE any extraction attempt — not a default or guess.
        """
        cost = profile.estimated_extraction_cost

        if cost == ExtractionCost.FAST_TEXT_SUFFICIENT:
            return "A"
        elif cost == ExtractionCost.NEEDS_LAYOUT_MODEL:
            return "B"
        else:
            # NEEDS_VISION_MODEL — scanned document, go straight to C
            return "C"

    # ----------------------------------------------------------
    # Step 2: Confidence-gated escalation
    # ----------------------------------------------------------

    def _run_with_escalation(
        self,
        file_path: Path,
        profile: DocumentProfile,
        start_strategy: str,
    ) -> tuple[ExtractedDocument, RoutingDecision]:
        """
        Run extraction starting at start_strategy.
        Escalate through the chain until confidence passes or C is reached.

        Returns both the final ExtractedDocument and the full RoutingDecision.
        """
        attempts: list[StrategyAttempt] = []
        start_idx = self._chain.index(start_strategy)

        final_result: ExtractedDocument | None = None
        final_strategy: str = start_strategy

        for strategy_name in self._chain[start_idx:]:
            threshold = self._thresholds[strategy_name]
            extractor = self._get_extractor(strategy_name)

            # Run the strategy
            result = extractor.extract(file_path, profile)
            confidence = result.extraction_confidence
            passed = (confidence >= threshold) or (strategy_name == "C")

            # Record this attempt
            attempts.append(StrategyAttempt(
                strategy=strategy_name,
                confidence_achieved=confidence,
                confidence_threshold=threshold,
                passed=passed,
                escalation_reason=(
                    None if passed else
                    f"Confidence {confidence:.3f} below threshold {threshold:.3f} "
                    f"— escalating to next strategy tier"
                ),
                cost_usd=result.cost_estimate_usd,
                processing_time_seconds=result.processing_time_seconds,
            ))

            final_result = result
            final_strategy = strategy_name

            if passed:
                break  # Confidence gate passed — stop escalating

        # Build the RoutingDecision
        routing = RoutingDecision(
            doc_id=profile.doc_id,
            profile_origin_type=str(profile.origin_type),
            profile_layout_complexity=str(profile.layout_complexity),
            profile_estimated_cost=str(profile.estimated_extraction_cost),
            initial_strategy_selected=start_strategy,
            attempts=attempts,
            final_strategy=final_strategy,
            escalation_occurred=len(attempts) > 1,
            total_attempts=len(attempts),
            total_cost_usd=sum(a.cost_usd for a in attempts),
        )

        return final_result, routing

    # ----------------------------------------------------------
    # Strategy registry
    # ----------------------------------------------------------

    def _get_extractor(self, strategy_name: str):
        """Return the correct extractor instance by strategy name."""
        extractors = {
            "A": self.strategy_a,
            "B": self.strategy_b,
            "C": self.strategy_c,
        }
        if strategy_name not in extractors:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Valid strategies: {list(extractors.keys())}"
            )
        return extractors[strategy_name]

    # ----------------------------------------------------------
    # Ledger logging
    # ----------------------------------------------------------

    def _write_ledger(
        self,
        profile: DocumentProfile,
        result: ExtractedDocument,
        routing: RoutingDecision,
    ) -> None:
        """
        Append one self-contained JSON line to extraction_ledger.jsonl.
        Includes the full routing decision so the ledger is a complete audit trail.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Document identity
            "doc_id": profile.doc_id,
            "filename": profile.filename,
            "total_pages": profile.total_pages,
            # Profile classification
            "origin_type": str(profile.origin_type),
            "layout_complexity": str(profile.layout_complexity),
            "domain_hint": str(profile.domain_hint),
            "profile_estimated_cost": str(profile.estimated_extraction_cost),
            # Routing decisions
            "initial_strategy_selected": routing.initial_strategy_selected,
            "final_strategy_used": routing.final_strategy,
            "escalation_occurred": routing.escalation_occurred,
            "total_attempts": routing.total_attempts,
            "escalation_path": routing.escalation_path,
            "attempts": [a.model_dump() for a in routing.attempts],
            # Final extraction result
            "extraction_confidence": result.extraction_confidence,
            "total_cost_usd": routing.total_cost_usd,
            "processing_time_seconds": result.processing_time_seconds,
            "text_blocks_count": len(result.text_blocks),
            "tables_count": len(result.tables),
            "figures_count": len(result.figures),
        }

        with open(self.cfg.ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
if __name__ == "__main__":
    import sys
    from pathlib import Path
    from src.agents.triage import TriageAgent

    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.extractor <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    # Step 1 - triage first
    print("Running Triage Agent...")
    triage = TriageAgent()
    profile = triage.run(pdf_path)
    print(f"  Strategy selected : {profile.estimated_extraction_cost}")

    # Step 2 - extraction router
    print("\nRunning Extraction Router...")
    router = ExtractionRouter()
    result = router.run(pdf_path, profile)

    routing = result.routing_decision
    print("\n" + "="*50)
    print("  EXTRACTION COMPLETE")
    print("="*50)
    print(f"  strategy_used     : {result.strategy_used}")
    print(f"  confidence        : {result.extraction_confidence}")
    print(f"  escalation        : {routing['escalation_occurred']}")
    print(f"  escalation_path   : {routing.get('escalation_path', 'none')}")
    print(f"  text_blocks       : {len(result.text_blocks)}")
    print(f"  tables            : {len(result.tables)}")
    print(f"  processing_time   : {result.processing_time_seconds}s")
    print(f"  ledger written to : .refinery/extraction_ledger.jsonl")
    print("="*50)