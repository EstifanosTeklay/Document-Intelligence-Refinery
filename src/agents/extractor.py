"""
Extraction Router — Stage 2 of the Document Intelligence Refinery

Reads the DocumentProfile, selects the correct extraction strategy,
and escalates automatically if confidence falls below threshold.
Logs every decision to .refinery/extraction_ledger.jsonl
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.models import DocumentProfile, ExtractedDocument, ExtractionCost
from src.strategies.strategy_a import FastTextExtractor
from src.strategies.strategy_b import LayoutExtractor
from src.strategies.strategy_c import VisionExtractor
from src.utils.config import config


class ExtractionRouter:
    """
    Strategy pattern router with confidence-gated escalation.

    Routing logic:
      DocumentProfile.estimated_extraction_cost → selects starting strategy
      If extraction_confidence < threshold → escalate to next strategy
      Strategy C is always the final fallback — never escalates further.

    Every extraction attempt is logged to extraction_ledger.jsonl.
    """

    def __init__(self):
        self.cfg = config
        self.strategy_a = FastTextExtractor()
        self.strategy_b = LayoutExtractor()
        self.strategy_c = VisionExtractor()

    # ----------------------------------------------------------
    # Public entry point
    # ----------------------------------------------------------

    def run(
        self,
        file_path: str | Path,
        profile: DocumentProfile,
    ) -> ExtractedDocument:
        """
        Route a document to the correct extraction strategy.
        Escalates automatically on low confidence.
        Returns the best ExtractedDocument obtained.
        """
        file_path = Path(file_path)
        cost = profile.estimated_extraction_cost

        # Select starting strategy based on DocumentProfile
        if cost == ExtractionCost.FAST_TEXT_SUFFICIENT:
            result = self._run_with_escalation(
                file_path, profile,
                start_strategy="A"
            )
        elif cost == ExtractionCost.NEEDS_LAYOUT_MODEL:
            result = self._run_with_escalation(
                file_path, profile,
                start_strategy="B"
            )
        else:
            # NEEDS_VISION_MODEL — go straight to C
            result = self._run_with_escalation(
                file_path, profile,
                start_strategy="C"
            )

        self._write_ledger(file_path, profile, result)
        return result

    # ----------------------------------------------------------
    # Escalation logic
    # ----------------------------------------------------------

    def _run_with_escalation(
        self,
        file_path: Path,
        profile: DocumentProfile,
        start_strategy: str,
    ) -> ExtractedDocument:
        """
        Run extraction starting at start_strategy.
        Escalate A → B → C if confidence is below threshold.
        """
        if start_strategy == "A":
            result = self.strategy_a.extract(file_path, profile)
            if result.extraction_confidence >= self.cfg.strategy_a_min_confidence:
                return result
            # Escalation A → B
            result = self.strategy_b.extract(file_path, profile)
            if result.extraction_confidence >= self.cfg.strategy_b_min_confidence:
                return result
            # Escalation B → C
            return self.strategy_c.extract(file_path, profile)

        elif start_strategy == "B":
            result = self.strategy_b.extract(file_path, profile)
            if result.extraction_confidence >= self.cfg.strategy_b_min_confidence:
                return result
            # Escalation B → C
            return self.strategy_c.extract(file_path, profile)

        else:
            # Strategy C — final fallback, no further escalation
            return self.strategy_c.extract(file_path, profile)

    # ----------------------------------------------------------
    # Ledger logging
    # ----------------------------------------------------------

    def _write_ledger(
        self,
        file_path: Path,
        profile: DocumentProfile,
        result: ExtractedDocument,
    ) -> None:
        """
        Append one line to extraction_ledger.jsonl.
        Each line is a self-contained JSON record.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "doc_id": profile.doc_id,
            "filename": profile.filename,
            "total_pages": profile.total_pages,
            "origin_type": profile.origin_type,
            "layout_complexity": profile.layout_complexity,
            "domain_hint": profile.domain_hint,
            "strategy_used": result.strategy_used,
            "extraction_confidence": result.extraction_confidence,
            "cost_estimate_usd": result.cost_estimate_usd,
            "processing_time_seconds": result.processing_time_seconds,
            "text_blocks_count": len(result.text_blocks),
            "tables_count": len(result.tables),
            "figures_count": len(result.figures),
        }

        ledger_path = self.cfg.ledger_path
        with open(ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
