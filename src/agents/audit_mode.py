"""
AuditMode — Explicit Claim Verification API

Takes a claim string, searches relevant chunks and FactTable facts,
checks for supporting evidence, and returns a structured verification
result with status, citations, and confidence score.

Verification statuses:
    SUPPORTED       — evidence found in chunks or facts
    CONTRADICTED    — evidence found that contradicts the claim
    UNVERIFIABLE    — no relevant evidence found in the document
    PARTIAL         — some supporting evidence but incomplete

Usage:
    auditor = AuditMode(doc_id="annual_report_2023")
    result  = auditor.verify("Total revenue was ETB 2.4 billion in 2023")
    print(result.status)        # SUPPORTED
    print(result.citations)     # list of SourceCitation
    print(result.confidence)    # 0.87
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.models.ldu import LDU
from src.models.pageindex import PageIndex, PageIndexNode
from src.models.provenance import SourceCitation
from src.models.extracted_document import BoundingBox
from src.utils.config import config


# ════════════════════════════════════════════════════════════
# Verification result
# ════════════════════════════════════════════════════════════

@dataclass
class VerificationResult:
    """
    The output of a single claim verification.

    Fields:
        claim           : The original claim string submitted for audit
        status          : SUPPORTED | CONTRADICTED | UNVERIFIABLE | PARTIAL
        confidence      : 0.0–1.0 confidence in the verification status
        citations       : Source citations that support or contradict the claim
        supporting_text : Excerpts from chunks that directly support the claim
        contradicting_text : Excerpts that contradict the claim (if any)
        fact_evidence   : Relevant numerical facts found in FactTable
        unverifiable_reason : Explanation if status is UNVERIFIABLE
        chunks_searched : Total chunks examined
        facts_searched  : Total facts examined
    """
    claim:                str
    status:               str   = "UNVERIFIABLE"
    confidence:           float = 0.0
    citations:            list[SourceCitation]  = field(default_factory=list)
    supporting_text:      list[str]             = field(default_factory=list)
    contradicting_text:   list[str]             = field(default_factory=list)
    fact_evidence:        list[dict]            = field(default_factory=list)
    unverifiable_reason:  str                   = ""
    chunks_searched:      int                   = 0
    facts_searched:       int                   = 0

    def summary(self) -> str:
        lines = [
            f"Claim      : {self.claim}",
            f"Status     : {self.status}",
            f"Confidence : {self.confidence:.0%}",
            f"Citations  : {len(self.citations)}",
            f"Facts      : {len(self.fact_evidence)}",
        ]
        if self.supporting_text:
            lines.append("Supporting evidence:")
            for t in self.supporting_text[:2]:
                lines.append(f"  \"{t[:120]}\"")
        if self.contradicting_text:
            lines.append("Contradicting evidence:")
            for t in self.contradicting_text[:2]:
                lines.append(f"  \"{t[:120]}\"")
        if self.unverifiable_reason:
            lines.append(f"Reason     : {self.unverifiable_reason}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# AuditMode
# ════════════════════════════════════════════════════════════

class AuditMode:
    """
    Verifies factual claims against a document's chunks and FactTable.

    Steps:
        1. Extract key terms and numerical values from the claim
        2. Search VectorStore/chunks for relevant passages (semantic_search)
        3. Query FactTable for matching numerical facts (structured_query)
        4. Score evidence as supporting, contradicting, or absent
        5. Return VerificationResult with status, confidence, and citations
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self._chunks: list[LDU]      = []
        self._index:  Optional[PageIndex] = None
        self._load()

    def _load(self) -> None:
        """Load chunks and PageIndex from disk."""
        # Load chunks
        chunk_path = config.refinery_dir / "chunks" / f"{self.doc_id}.jsonl"
        if chunk_path.exists():
            with open(chunk_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d    = json.loads(line)
                    bbox = None
                    if d.get("bounding_box"):
                        b    = d["bounding_box"]
                        bbox = BoundingBox(
                            x0=b.get("x0", 0), y0=b.get("y0", 0),
                            x1=b.get("x1", 0), y1=b.get("y1", 0),
                            page=b.get("page", 1),
                        )
                    self._chunks.append(LDU(
                        chunk_id=d["chunk_id"], doc_id=d["doc_id"],
                        chunk_index=d["chunk_index"], content=d["content"],
                        chunk_type=d["chunk_type"],
                        page_refs=d.get("page_refs", [1]),
                        bounding_box=bbox,
                        content_hash=d.get("content_hash", ""),
                        parent_section=d.get("parent_section"),
                        parent_section_page=d.get("parent_section_page"),
                        token_count=d.get("token_count", 0),
                        strategy_used=d.get("strategy_used", ""),
                    ))

        # Load PageIndex
        index_path = config.refinery_dir / "pageindex" / f"{self.doc_id}.json"
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
            nodes = []
            for n in data.get("nodes", []):
                nodes.append(PageIndexNode(
                    node_id=n["node_id"], doc_id=n["doc_id"],
                    title=n["title"], level=n.get("level", 1),
                    page_start=n["page_start"], page_end=n["page_end"],
                    summary=n.get("summary", ""), chunk_ids=n.get("chunk_ids", []),
                    parent_node_id=n.get("parent_node_id"),
                    child_node_ids=n.get("child_node_ids", []),
                    data_types_present=n.get("data_types_present", []),
                ))
            self._index = PageIndex(
                doc_id=data["doc_id"], filename=data["filename"],
                total_pages=data["total_pages"], nodes=nodes,
                root_node_ids=data.get("root_node_ids", []),
                document_summary=data.get("document_summary", ""),
                total_sections=data.get("total_sections", 0),
                total_chunks_indexed=data.get("total_chunks_indexed", 0),
            )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def verify(self, claim: str) -> VerificationResult:
        """
        Verify a claim against the document.

        Args:
            claim: A factual claim to verify, e.g.
                   "The company had 500 employees in 2023"
                   "Total revenue exceeded ETB 2 billion"

        Returns:
            VerificationResult with status, confidence, and citations.
        """
        result = VerificationResult(claim=claim)

        # Step 1 — Extract claim components
        claim_terms  = self._extract_claim_terms(claim)
        claim_values = self._extract_numerical_values(claim)

        # Step 2 — Search chunks
        relevant_chunks = self._search_chunks(claim_terms)
        result.chunks_searched = len(self._chunks)

        # Step 3 — Search FactTable
        relevant_facts = self._search_facts(claim_terms, claim_values)
        result.facts_searched = self._count_facts()

        # Step 4 — Score evidence
        supporting    = []
        contradicting = []

        for chunk in relevant_chunks:
            score = self._score_chunk_vs_claim(chunk.content, claim_terms, claim_values)
            if score > 0.4:
                supporting.append(chunk)
                result.supporting_text.append(chunk.content[:200])
            elif score < -0.2:
                contradicting.append(chunk)
                result.contradicting_text.append(chunk.content[:200])

        for fact in relevant_facts:
            fact_support = self._score_fact_vs_claim(fact, claim_values)
            if fact_support > 0:
                result.fact_evidence.append(fact)

        # Step 5 — Determine status and confidence
        result.citations = self._build_citations(supporting)
        result.status, result.confidence = self._determine_status(
            supporting, contradicting, result.fact_evidence, claim
        )

        if result.status == "UNVERIFIABLE":
            result.unverifiable_reason = (
                f"No matching evidence found for claim terms: "
                f"{', '.join(list(claim_terms)[:5])}"
            )

        return result

    def verify_batch(self, claims: list[str]) -> list[VerificationResult]:
        """Verify multiple claims at once."""
        return [self.verify(claim) for claim in claims]

    # ----------------------------------------------------------
    # Claim parsing
    # ----------------------------------------------------------

    def _extract_claim_terms(self, claim: str) -> set[str]:
        """Extract meaningful keywords from the claim."""
        STOPWORDS = {
            "the","a","an","and","or","in","on","at","to","for","of",
            "with","by","is","are","was","were","be","been","has","have",
            "had","will","would","that","this","it","its","we","our",
            "approximately","about","around","over","under","more","less",
        }
        tokens = re.findall(r"[a-z]{2,}", claim.lower())
        return {t for t in tokens if t not in STOPWORDS}

    def _extract_numerical_values(self, claim: str) -> list[float]:
        """Extract all numerical values from the claim."""
        values = []
        # Match numbers with optional multipliers
        pattern = re.compile(
            r"([\d,\.]+)\s*(billion|million|thousand|[BMK])?", re.IGNORECASE
        )
        multipliers = {
            "billion": 1e9, "million": 1e6, "thousand": 1e3,
            "b": 1e9, "m": 1e6, "k": 1e3,
        }
        for match in pattern.finditer(claim):
            try:
                val  = float(match.group(1).replace(",", ""))
                mult = multipliers.get((match.group(2) or "").lower(), 1)
                values.append(val * mult)
            except ValueError:
                pass
        return values

    # ----------------------------------------------------------
    # Evidence search
    # ----------------------------------------------------------

    def _search_chunks(self, claim_terms: set[str]) -> list[LDU]:
        """Find chunks with keyword overlap to the claim."""
        scored = []
        for chunk in self._chunks:
            chunk_words = set(re.findall(r"[a-z]{2,}", chunk.content.lower()))
            overlap = len(claim_terms & chunk_words)
            if overlap >= 2:  # require at least 2 term matches
                scored.append((overlap, chunk))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:10]]

    def _search_facts(
        self, claim_terms: set[str], claim_values: list[float]
    ) -> list[dict]:
        """Query FactTable for facts relevant to the claim."""
        try:
            from src.storage.fact_table import FactTable
            ft = FactTable()

            # Search by label keywords
            all_facts = []
            for term in list(claim_terms)[:3]:
                facts = ft.query_facts(doc_id=self.doc_id, fact_label=term, limit=10)
                all_facts.extend(facts)

            # Deduplicate by fact id
            seen = set()
            unique = []
            for f in all_facts:
                if f["id"] not in seen:
                    seen.add(f["id"])
                    unique.append(f)
            return unique

        except Exception:
            return []

    def _count_facts(self) -> int:
        """Count total facts in the FactTable for this document."""
        try:
            from src.storage.fact_table import FactTable
            ft    = FactTable()
            facts = ft.query_facts(doc_id=self.doc_id, limit=10000)
            return len(facts)
        except Exception:
            return 0

    # ----------------------------------------------------------
    # Evidence scoring
    # ----------------------------------------------------------

    def _score_chunk_vs_claim(
        self, chunk_text: str, claim_terms: set[str], claim_values: list[float]
    ) -> float:
        """
        Score a chunk as supporting (+) or contradicting (−) the claim.
        Returns a float in [-1, 1].
        """
        chunk_lower = chunk_text.lower()
        chunk_words = set(re.findall(r"[a-z]{2,}", chunk_lower))

        # Term overlap
        term_overlap = len(claim_terms & chunk_words) / max(len(claim_terms), 1)

        # Numerical value match
        value_score = 0.0
        if claim_values:
            chunk_nums = re.findall(r"[\d,\.]+", chunk_text)
            chunk_vals = []
            for n in chunk_nums:
                try:
                    chunk_vals.append(float(n.replace(",", "")))
                except ValueError:
                    pass

            for cv in claim_values:
                for chunk_val in chunk_vals:
                    if cv > 0 and abs(chunk_val - cv) / cv < 0.05:
                        value_score = 0.5  # within 5% match
                        break

        # Negation detection — if chunk negates the claim
        negation_pattern = re.compile(
            r"\b(not|no|never|neither|nor|without|failed|declined|"
            r"decreased|reduced|below|under|less than)\b",
            re.IGNORECASE,
        )
        negation_count = len(negation_pattern.findall(chunk_text))
        negation_penalty = min(negation_count * 0.15, 0.4)

        score = (term_overlap * 0.5 + value_score) - negation_penalty
        return max(-1.0, min(1.0, score))

    def _score_fact_vs_claim(self, fact: dict, claim_values: list[float]) -> float:
        """Check if a fact's value matches any of the claim's numerical values."""
        if not claim_values:
            return 0.5  # no values to check — label match alone is evidence

        fact_val = fact.get("fact_value", 0)
        for cv in claim_values:
            if cv > 0 and abs(fact_val - cv) / cv < 0.10:  # within 10%
                return 1.0
        return 0.0

    # ----------------------------------------------------------
    # Status determination
    # ----------------------------------------------------------

    def _determine_status(
        self,
        supporting:   list[LDU],
        contradicting: list[LDU],
        facts:         list[dict],
        claim:         str,
    ) -> tuple[str, float]:
        """
        Determine verification status and confidence from evidence.

        Returns: (status, confidence)
        """
        has_support     = len(supporting) > 0 or len(facts) > 0
        has_contradiction = len(contradicting) > 0

        if has_support and not has_contradiction:
            confidence = min(0.5 + len(supporting) * 0.1 + len(facts) * 0.05, 0.95)
            return "SUPPORTED", confidence

        if has_support and has_contradiction:
            # More support than contradiction
            if len(supporting) > len(contradicting):
                return "PARTIAL", 0.55
            return "CONTRADICTED", 0.6

        if not has_support and has_contradiction:
            return "CONTRADICTED", 0.7

        return "UNVERIFIABLE", 0.0

    # ----------------------------------------------------------
    # Build citations
    # ----------------------------------------------------------

    def _build_citations(self, chunks: list[LDU]) -> list[SourceCitation]:
        citations = []
        for chunk in chunks:
            node = None
            if self._index:
                node = next(
                    (n for n in self._index.nodes if chunk.chunk_id in n.chunk_ids),
                    None
                )
            citations.append(SourceCitation(
                doc_id=chunk.doc_id,
                document_name=self._index.filename if self._index else chunk.doc_id,
                page_number=chunk.page_refs[0] if chunk.page_refs else 1,
                bbox=chunk.bounding_box,
                chunk_id=chunk.chunk_id,
                content_hash=chunk.content_hash,
                excerpt=chunk.content[:200],
                strategy_used=chunk.strategy_used,
                section_title=chunk.parent_section,
                section_node_id=node.node_id if node else None,
            ))
        return citations


# ════════════════════════════════════════════════════════════
# Entry point — test audit mode from command line
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.agents.audit_mode <doc_id> '<claim>'")
        print("Example: python -m src.agents.audit_mode mcf 'Contributions are paid back after internship ends'")
        sys.exit(1)

    doc_id = sys.argv[1]
    claim  = sys.argv[2]

    print(f"\nAudit Mode — verifying claim against document '{doc_id}'")
    print(f"Claim: {claim}\n")

    auditor = AuditMode(doc_id=doc_id)
    result  = auditor.verify(claim)

    print("=" * 55)
    print(result.summary())
    print("=" * 55)

    if result.citations:
        print(f"\nCitations ({len(result.citations)}):")
        for i, c in enumerate(result.citations, 1):
            print(f"  [{i}] Page {c.page_number} | Section: {c.section_title}")
            print(f"      \"{c.excerpt[:100]}...\"")

    if result.fact_evidence:
        print(f"\nFact evidence ({len(result.fact_evidence)}):")
        for f in result.fact_evidence[:5]:
            print(f"  {f['fact_label']}: {f['fact_value']} {f['fact_unit']} ({f['fact_type']})")
