"""
ChunkValidator — validates all 5 chunking rules on candidate LDUs
before they are emitted from the ChunkingEngine.

For each rule violation found, the validator either:
  - Auto-repairs the chunk (where possible)
  - Flags the chunk with a validation_note
  - Rejects the chunk (rare — only for empty/corrupt content)

Validation results are returned alongside the validated chunks so the
pipeline can log or surface any issues.
"""
import re
from dataclasses import dataclass, field
from src.models.ldu import LDU, ChunkType


# ════════════════════════════════════════════════════════════
# Validation result
# ════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Result of validating a single LDU against all 5 rules."""
    chunk_id:         str
    passed:           bool
    violations:       list[str] = field(default_factory=list)
    repairs:          list[str] = field(default_factory=list)
    rejected:         bool      = False
    rejection_reason: str       = ""


@dataclass
class ValidationReport:
    """Summary of the full validation pass over all candidate LDUs."""
    total_input:     int
    total_passed:    int
    total_repaired:  int
    total_rejected:  int
    results:         list[ValidationResult] = field(default_factory=list)

    @property
    def violation_count(self) -> int:
        return sum(len(r.violations) for r in self.results)

    def summary_lines(self) -> list[str]:
        lines = [
            f"  input chunks   : {self.total_input}",
            f"  passed clean   : {self.total_passed}",
            f"  auto-repaired  : {self.total_repaired}",
            f"  rejected       : {self.total_rejected}",
            f"  total violations: {self.violation_count}",
        ]
        for r in self.results:
            if r.violations or r.repairs or r.rejected:
                lines.append(f"    [{r.chunk_id}]")
                for v in r.violations:
                    lines.append(f"      VIOLATION: {v}")
                for rp in r.repairs:
                    lines.append(f"      REPAIRED : {rp}")
                if r.rejected:
                    lines.append(f"      REJECTED : {r.rejection_reason}")
        return lines


# ════════════════════════════════════════════════════════════
# Validator
# ════════════════════════════════════════════════════════════

class ChunkValidator:
    """
    Validates candidate LDUs against all 5 inviolable chunking rules.

    Usage:
        validator = ChunkValidator()
        validated_chunks, report = validator.validate(candidate_chunks)

    Rules checked:
        Rule 1: Table integrity   — TABLE chunks must have table_data with headers
        Rule 2: Figure caption    — FIGURE chunks must carry caption as metadata
        Rule 3: List unity        — LIST chunks must not be split mid-list
        Rule 4: Section metadata  — All non-heading chunks must have parent_section
        Rule 5: Cross-references  — Relationships must point to existing chunk_ids
    """

    def validate(
        self, chunks: list[LDU]
    ) -> tuple[list[LDU], ValidationReport]:
        """
        Validate and repair all candidate LDUs.

        Returns:
            (validated_chunks, report)
            validated_chunks excludes rejected chunks.
        """
        valid_chunk_ids = {c.chunk_id for c in chunks}
        results: list[ValidationResult] = []
        output:  list[LDU]             = []

        for chunk in chunks:
            result = ValidationResult(chunk_id=chunk.chunk_id, passed=True)

            # Run all 5 rule checks — each may mutate chunk in place
            chunk = self._rule1_table_integrity(chunk, result)
            chunk = self._rule2_figure_caption(chunk, result)
            chunk = self._rule3_list_unity(chunk, result)
            chunk = self._rule4_section_metadata(chunk, result)
            chunk = self._rule5_cross_references(chunk, result, valid_chunk_ids)

            # Reject completely empty or corrupt chunks
            if not chunk.content or not chunk.content.strip():
                result.rejected       = True
                result.rejection_reason = "Empty content after validation"
                result.passed         = False
            elif len(chunk.content.strip()) < 3:
                result.rejected       = True
                result.rejection_reason = "Content too short to be meaningful"
                result.passed         = False

            if result.violations or result.repairs:
                result.passed = len(result.violations) == 0

            results.append(result)

            if not result.rejected:
                output.append(chunk)

        total_passed   = sum(1 for r in results if r.passed and not r.rejected)
        total_repaired = sum(1 for r in results if r.repairs)
        total_rejected = sum(1 for r in results if r.rejected)

        report = ValidationReport(
            total_input    = len(chunks),
            total_passed   = total_passed,
            total_repaired = total_repaired,
            total_rejected = total_rejected,
            results        = results,
        )

        return output, report

    # ----------------------------------------------------------
    # Rule 1 — Table integrity
    # ----------------------------------------------------------

    def _rule1_table_integrity(
        self, chunk: LDU, result: ValidationResult
    ) -> LDU:
        """
        TABLE chunks must have table_data with at least a headers list.
        If table_data is missing but content looks like a table, rebuild it.
        If headers are absent but rows exist, flag as violation.
        """
        if chunk.chunk_type not in (ChunkType.TABLE, "table"):
            return chunk

        # Missing table_data entirely
        if not chunk.table_data:
            # Try to reconstruct from content (pipe-separated text)
            reconstructed = self._reconstruct_table_data(chunk.content)
            if reconstructed:
                chunk.table_data = reconstructed
                result.repairs.append(
                    "Rule 1: table_data missing — reconstructed from pipe-separated content"
                )
            else:
                result.violations.append(
                    "Rule 1: TABLE chunk has no table_data and content is not parseable"
                )
            return chunk

        # Has table_data but no headers
        headers = chunk.table_data.get("headers", [])
        rows    = chunk.table_data.get("rows", [])

        if not headers and rows:
            # Promote first row to headers as a repair
            chunk.table_data["headers"] = rows[0] if rows else []
            chunk.table_data["rows"]    = rows[1:] if len(rows) > 1 else []
            result.repairs.append(
                "Rule 1: No headers found — promoted first row to headers"
            )

        return chunk

    def _reconstruct_table_data(self, content: str) -> dict | None:
        """Try to parse pipe-separated table text back into headers + rows."""
        lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            return None
        # Check if it looks like a table (has pipe separators)
        if not any("|" in line for line in lines):
            return None

        rows = []
        for line in lines:
            if re.match(r"^[-| ]+$", line):
                continue  # separator line
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells:
                rows.append(cells)

        if not rows:
            return None

        return {
            "headers": rows[0],
            "rows":    rows[1:],
        }

    # ----------------------------------------------------------
    # Rule 2 — Figure caption
    # ----------------------------------------------------------

    def _rule2_figure_caption(
        self, chunk: LDU, result: ValidationResult
    ) -> LDU:
        """
        FIGURE chunks must carry caption as figure_caption metadata.
        If caption is embedded in content, extract it to metadata.
        """
        if chunk.chunk_type not in (ChunkType.FIGURE, "figure"):
            return chunk

        # Caption should be in figure_caption field
        if chunk.figure_caption is None:
            # Try to extract from content: "[Figure] Some caption text"
            match = re.match(r"\[Figure\]\s*(.+)", chunk.content.strip(), re.DOTALL)
            if match and match.group(1).strip():
                caption = match.group(1).strip()
                chunk.figure_caption = caption
                result.repairs.append(
                    f"Rule 2: Extracted caption from content to figure_caption metadata"
                )
            else:
                result.violations.append(
                    "Rule 2: FIGURE chunk has no figure_caption metadata"
                )

        # Ensure caption is NOT a standalone separate chunk (checked by type)
        # If content is ONLY the caption text with no [Figure] marker, flag it
        if chunk.figure_caption and chunk.content.strip() == chunk.figure_caption.strip():
            # Content should be "[Figure] caption", not just the caption
            chunk.content = f"[Figure] {chunk.figure_caption}"
            result.repairs.append(
                "Rule 2: Content was bare caption — prefixed with [Figure] marker"
            )

        return chunk

    # ----------------------------------------------------------
    # Rule 3 — List unity
    # ----------------------------------------------------------

    def _rule3_list_unity(
        self, chunk: LDU, result: ValidationResult
    ) -> LDU:
        """
        LIST chunks must not be truncated mid-list.
        Checks that the list appears complete (last line is not a dangling item).
        Edge case: nested lists are detected and flagged.
        """
        if chunk.chunk_type not in (ChunkType.LIST, "list"):
            return chunk

        lines = chunk.content.strip().splitlines()
        if not lines:
            return chunk

        list_pattern = re.compile(r"^(\s*[\-\•\*\·]|\s*\d+[\.\)])\s+")

        # Check for nested lists (indented items)
        has_nested = any(
            re.match(r"^\s{4,}", line) and list_pattern.match(line.strip())
            for line in lines
        )
        if has_nested:
            result.violations.append(
                "Rule 3: LIST chunk contains nested list items — "
                "consider splitting into separate LDUs"
            )

        # Check last line — if it ends mid-sentence without punctuation,
        # the list may be truncated
        last_line = lines[-1].strip()
        if last_line and not last_line[-1] in ".;:,)]":
            # Only flag if it looks like a list item (not a heading)
            if list_pattern.match(last_line):
                result.violations.append(
                    "Rule 3: LIST chunk may be truncated — "
                    f"last item ends without terminal punctuation: '{last_line[:60]}'"
                )

        return chunk

    # ----------------------------------------------------------
    # Rule 4 — Section metadata
    # ----------------------------------------------------------

    def _rule4_section_metadata(
        self, chunk: LDU, result: ValidationResult
    ) -> LDU:
        """
        All non-HEADING chunks must carry parent_section metadata.
        HEADING chunks are exempt (they ARE the section).
        """
        if chunk.chunk_type in (ChunkType.HEADING, "heading"):
            return chunk  # headings are exempt

        if not chunk.parent_section:
            result.violations.append(
                f"Rule 4: {chunk.chunk_type} chunk has no parent_section — "
                "section metadata missing"
            )
            # Repair: assign a fallback
            chunk.parent_section = "Document"
            result.repairs.append(
                "Rule 4: Assigned fallback parent_section = 'Document'"
            )

        return chunk

    # ----------------------------------------------------------
    # Rule 5 — Cross-reference integrity
    # ----------------------------------------------------------

    def _rule5_cross_references(
        self, chunk: LDU, result: ValidationResult, valid_ids: set[str]
    ) -> LDU:
        """
        All relationships must point to a chunk_id that exists in the document.
        Dangling references are removed with a violation logged.
        """
        if not chunk.relationships:
            return chunk

        valid_rels  = []
        dangling    = []

        for rel in chunk.relationships:
            if rel.target_chunk_id in valid_ids:
                valid_rels.append(rel)
            else:
                dangling.append(rel.target_chunk_id)

        if dangling:
            result.violations.append(
                f"Rule 5: {len(dangling)} dangling cross-reference(s) removed: "
                + ", ".join(dangling[:3])
            )
            result.repairs.append(
                f"Rule 5: Removed {len(dangling)} dangling relationship(s)"
            )
            chunk.relationships = valid_rels

        return chunk
