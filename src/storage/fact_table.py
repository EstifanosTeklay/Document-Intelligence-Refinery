"""
FactTable — Data Persistence Layer (Storage Path 2)

Extracts key-value numerical facts from document chunks and stores them
in a SQLite database with a defined schema. Enables precise SQL-based
retrieval of financial and numerical facts.

Schema:
    facts(
        id, doc_id, chunk_id, page_number, section_title,
        fact_label, fact_value, fact_unit, fact_year,
        fact_type, content_hash, confidence, inserted_at
    )

Fact types detected:
    - CURRENCY   : ETB 2.4 billion, $1.2M, USD 500,000
    - PERCENTAGE : 12.5%, 8 percent
    - COUNT      : 1,234 employees, 45 branches
    - RATIO      : 1:4, 3.2x
    - DATE_RANGE : 2022/23, FY2023, Q3 2024
    - QUANTITY   : 500 tonnes, 120 MW

Storage location: .refinery/facttable/facts.db
"""
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.models.ldu import LDU
from src.utils.config import config


# ════════════════════════════════════════════════════════════
# Fact dataclass
# ════════════════════════════════════════════════════════════

@dataclass
class Fact:
    """A single extracted numerical fact."""
    doc_id:        str
    chunk_id:      str
    page_number:   int
    section_title: str
    fact_label:    str          # e.g. "Total Revenue", "Net Profit Margin"
    fact_value:    float        # normalised numeric value
    fact_unit:     str          # e.g. "ETB", "%", "employees", "MW"
    fact_year:     str          # e.g. "2023", "2022/23", "Q3 2024", ""
    fact_type:     str          # CURRENCY | PERCENTAGE | COUNT | RATIO | QUANTITY | DATE_RANGE
    content_hash:  str          # SHA-256 of source chunk — provenance
    confidence:    float = 1.0  # 0.0–1.0 extraction confidence
    raw_text:      str  = ""    # original matched text for audit


# ════════════════════════════════════════════════════════════
# Extraction patterns
# ════════════════════════════════════════════════════════════

# Each pattern: (regex, fact_type, unit_group, value_group)
FACT_PATTERNS: list[tuple] = [

    # ── Currency ────────────────────────────────────────────
    # ETB 2.4 billion / ETB2,400,000
    (
        r"(?P<label>[A-Za-z ]{2,40}?)[:\s]+ETB\s*(?P<value>[\d,\.]+)\s*(?P<mult>billion|million|thousand)?",
        "CURRENCY", "ETB", None
    ),
    # $1.2M / USD 500,000
    (
        r"(?P<label>[A-Za-z ]{2,40}?)[:\s]+(?:USD|\$)\s*(?P<value>[\d,\.]+)\s*(?P<mult>billion|million|thousand|[MBK])?",
        "CURRENCY", "USD", None
    ),
    # Birr 3,500,000
    (
        r"(?P<label>[A-Za-z ]{2,40}?)[:\s]+(?P<value>[\d,\.]+)\s*(?:birr|Birr)\b",
        "CURRENCY", "ETB", None
    ),

    # ── Percentage ──────────────────────────────────────────
    (
        r"(?P<label>[A-Za-z ]{2,50}?)[:\s]+(?P<value>[\d,\.]+)\s*(?:%|percent|per cent)",
        "PERCENTAGE", "%", None
    ),

    # ── Count ───────────────────────────────────────────────
    (
        r"(?P<value>[\d,]+)\s+(?P<label>employees|staff|branches|offices|customers|"
        r"beneficiaries|students|patients|members|projects|loans)",
        "COUNT", None, None
    ),

    # ── Ratio ───────────────────────────────────────────────
    (
        r"(?P<label>[A-Za-z ]{2,40}?)[:\s]+(?P<value>[\d\.]+)\s*(?:x|times|:1)",
        "RATIO", "x", None
    ),

    # ── Quantity ────────────────────────────────────────────
    (
        r"(?P<value>[\d,\.]+)\s+(?P<label>tonnes|tons|MW|GW|km|hectares|acres|"
        r"litres|liters|units|items)",
        "QUANTITY", None, None
    ),

    # ── Standalone year references ───────────────────────────
    (
        r"(?:FY|fiscal year|year)\s*(?P<value>20\d{2}(?:[\/\-]\d{2,4})?)",
        "DATE_RANGE", "FY", None
    ),
]

# Multiplier lookup
MULTIPLIERS = {
    "billion": 1_000_000_000,
    "million": 1_000_000,
    "thousand": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
    "K": 1_000,
}

# Year pattern for extracting year context from surrounding text
YEAR_PATTERN = re.compile(
    r"\b(20\d{2}(?:[\/\-]\d{2,4})?|FY\s*20\d{2}|Q[1-4]\s*20\d{2})\b"
)


# ════════════════════════════════════════════════════════════
# FactTable class
# ════════════════════════════════════════════════════════════

class FactTable:
    """
    Extracts numerical facts from LDUs and stores them in SQLite.

    Schema columns:
        id            INTEGER PRIMARY KEY
        doc_id        TEXT    — source document identifier
        chunk_id      TEXT    — source LDU chunk_id
        page_number   INTEGER — 1-indexed source page
        section_title TEXT    — parent section from PageIndex
        fact_label    TEXT    — human-readable fact name
        fact_value    REAL    — normalised numeric value
        fact_unit     TEXT    — unit of measurement
        fact_year     TEXT    — temporal context
        fact_type     TEXT    — CURRENCY|PERCENTAGE|COUNT|RATIO|QUANTITY|DATE_RANGE
        content_hash  TEXT    — SHA-256 of source chunk
        confidence    REAL    — extraction confidence 0–1
        raw_text      TEXT    — original matched string
        inserted_at   TEXT    — ISO timestamp

    Usage:
        ft = FactTable()
        n  = ft.extract_and_store(chunks, doc_id="annual_report_2023")
        results = ft.query_facts(doc_id="annual_report_2023", fact_type="CURRENCY")
    """

    def __init__(self):
        ft_dir = config.refinery_dir / "facttable"
        ft_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = ft_dir / "facts.db"
        self._init_schema()

    # ----------------------------------------------------------
    # Schema initialisation
    # ----------------------------------------------------------

    def _init_schema(self) -> None:
        """
        Create the FactTable schema if it does not exist.
        All column names and types are fixed — not dynamic.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id        TEXT    NOT NULL,
                    chunk_id      TEXT    NOT NULL,
                    page_number   INTEGER NOT NULL DEFAULT 1,
                    section_title TEXT    NOT NULL DEFAULT '',
                    fact_label    TEXT    NOT NULL,
                    fact_value    REAL    NOT NULL,
                    fact_unit     TEXT    NOT NULL DEFAULT '',
                    fact_year     TEXT    NOT NULL DEFAULT '',
                    fact_type     TEXT    NOT NULL,
                    content_hash  TEXT    NOT NULL,
                    confidence    REAL    NOT NULL DEFAULT 1.0,
                    raw_text      TEXT    NOT NULL DEFAULT '',
                    inserted_at   TEXT    DEFAULT (datetime('now'))
                );

                -- Indices for fast retrieval
                CREATE INDEX IF NOT EXISTS idx_facts_doc
                    ON facts(doc_id);
                CREATE INDEX IF NOT EXISTS idx_facts_type
                    ON facts(fact_type);
                CREATE INDEX IF NOT EXISTS idx_facts_label
                    ON facts(fact_label);
                CREATE INDEX IF NOT EXISTS idx_facts_year
                    ON facts(fact_year);
                CREATE INDEX IF NOT EXISTS idx_facts_doc_type
                    ON facts(doc_id, fact_type);
                CREATE INDEX IF NOT EXISTS idx_facts_value
                    ON facts(fact_value);
            """)

    # ----------------------------------------------------------
    # Extraction
    # ----------------------------------------------------------

    def extract_and_store(self, chunks: list[LDU], doc_id: str) -> int:
        """
        Extract numerical facts from all chunks and write to SQLite.
        Returns the total number of facts stored.
        """
        all_facts: list[Fact] = []

        for chunk in chunks:
            facts = self._extract_facts_from_chunk(chunk, doc_id)
            all_facts.extend(facts)

        self._write_facts(all_facts, doc_id)
        return len(all_facts)

    def _extract_facts_from_chunk(self, chunk: LDU, doc_id: str) -> list[Fact]:
        """Run all extraction patterns against a single chunk."""
        facts: list[Fact] = []
        text  = chunk.content
        page  = chunk.page_refs[0] if chunk.page_refs else 1
        section = chunk.parent_section or ""

        # Extract year context from the chunk text
        year_ctx = self._extract_year_context(text)

        for pattern, fact_type, unit_override, _ in FACT_PATTERNS:
            try:
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    groups = m.groupdict()
                    raw    = m.group(0).strip()

                    # Parse numeric value
                    value_str = groups.get("value", "0").replace(",", "")
                    try:
                        value = float(value_str)
                    except ValueError:
                        continue

                    # Apply multiplier if present
                    mult_str = groups.get("mult", "") or ""
                    mult     = MULTIPLIERS.get(mult_str.strip(), 1)
                    value   *= mult

                    # Determine unit
                    unit = unit_override or groups.get("label", "").strip().lower()
                    unit = unit[:30]  # cap length

                    # Determine label
                    if fact_type == "COUNT":
                        label = groups.get("label", "count").strip().lower()
                    elif fact_type == "QUANTITY":
                        label = groups.get("label", "quantity").strip().lower()
                    else:
                        label = groups.get("label", "").strip()
                        label = re.sub(r"\s+", " ", label)[:80]

                    if not label:
                        label = fact_type.lower()

                    # Year — from match or surrounding context
                    year = groups.get("value", "") if fact_type == "DATE_RANGE" else year_ctx

                    facts.append(Fact(
                        doc_id=doc_id,
                        chunk_id=chunk.chunk_id,
                        page_number=page,
                        section_title=section,
                        fact_label=label,
                        fact_value=value,
                        fact_unit=unit,
                        fact_year=year,
                        fact_type=fact_type,
                        content_hash=chunk.content_hash,
                        confidence=0.9 if len(label) > 4 else 0.6,
                        raw_text=raw[:200],
                    ))
            except Exception:
                continue

        return facts

    def _extract_year_context(self, text: str) -> str:
        """Extract the most prominent year reference from text."""
        matches = YEAR_PATTERN.findall(text)
        if matches:
            return matches[0]
        return ""

    # ----------------------------------------------------------
    # Write
    # ----------------------------------------------------------

    def _write_facts(self, facts: list[Fact], doc_id: str) -> None:
        """Write all facts to SQLite, replacing existing facts for this doc."""
        rows = [
            (
                f.doc_id, f.chunk_id, f.page_number, f.section_title,
                f.fact_label, f.fact_value, f.fact_unit, f.fact_year,
                f.fact_type, f.content_hash, f.confidence, f.raw_text,
            )
            for f in facts
        ]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM facts WHERE doc_id = ?", (doc_id,))
            conn.executemany("""
                INSERT INTO facts (
                    doc_id, chunk_id, page_number, section_title,
                    fact_label, fact_value, fact_unit, fact_year,
                    fact_type, content_hash, confidence, raw_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)

    # ----------------------------------------------------------
    # Query functions
    # ----------------------------------------------------------

    def query_facts(
        self,
        doc_id:     Optional[str]   = None,
        fact_type:  Optional[str]   = None,
        fact_label: Optional[str]   = None,
        year:       Optional[str]   = None,
        min_value:  Optional[float] = None,
        max_value:  Optional[float] = None,
        section:    Optional[str]   = None,
        limit:      int             = 50,
    ) -> list[dict]:
        """
        Precise SQL retrieval of numerical facts with multiple filter axes.

        Args:
            doc_id:     Filter by document
            fact_type:  CURRENCY | PERCENTAGE | COUNT | RATIO | QUANTITY | DATE_RANGE
            fact_label: Keyword match on fact label (LIKE %keyword%)
            year:       Exact or partial year match
            min_value:  Minimum numeric value
            max_value:  Maximum numeric value
            section:    Keyword match on section title
            limit:      Maximum rows returned

        Returns:
            List of fact dicts ordered by fact_value DESC
        """
        conditions = []
        params:     list = []

        if doc_id:
            conditions.append("doc_id = ?")
            params.append(doc_id)
        if fact_type:
            conditions.append("fact_type = ?")
            params.append(fact_type.upper())
        if fact_label:
            conditions.append("fact_label LIKE ?")
            params.append(f"%{fact_label}%")
        if year:
            conditions.append("fact_year LIKE ?")
            params.append(f"%{year}%")
        if min_value is not None:
            conditions.append("fact_value >= ?")
            params.append(min_value)
        if max_value is not None:
            conditions.append("fact_value <= ?")
            params.append(max_value)
        if section:
            conditions.append("section_title LIKE ?")
            params.append(f"%{section}%")

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
            SELECT
                id, doc_id, chunk_id, page_number, section_title,
                fact_label, fact_value, fact_unit, fact_year,
                fact_type, content_hash, confidence, raw_text, inserted_at
            FROM facts
            {where}
            ORDER BY fact_value DESC
            LIMIT ?
        """
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        return [dict(row) for row in rows]

    def query_top_values(
        self,
        doc_id:    str,
        fact_type: str = "CURRENCY",
        top_k:     int = 10,
    ) -> list[dict]:
        """
        Return the top-k largest numerical facts of a given type.
        Useful for 'what are the largest revenue items?' queries.
        """
        return self.query_facts(
            doc_id=doc_id,
            fact_type=fact_type,
            limit=top_k,
        )

    def query_by_section(self, doc_id: str, section_keyword: str) -> list[dict]:
        """Return all facts from sections matching a keyword."""
        return self.query_facts(doc_id=doc_id, section=section_keyword)

    def aggregate(
        self,
        doc_id:    str,
        fact_type: str = "CURRENCY",
        agg:       str = "SUM",
    ) -> float:
        """
        Run an aggregate SQL query (SUM, AVG, MAX, MIN) over fact values.

        Example:
            ft.aggregate("annual_report_2023", "CURRENCY", "SUM")
            → total of all currency facts in the document
        """
        agg = agg.upper()
        if agg not in ("SUM", "AVG", "MAX", "MIN", "COUNT"):
            raise ValueError(f"Unsupported aggregation: {agg}")

        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                f"SELECT {agg}(fact_value) FROM facts "
                f"WHERE doc_id = ? AND fact_type = ?",
                (doc_id, fact_type.upper()),
            ).fetchone()

        return result[0] or 0.0

    def summary(self, doc_id: str) -> dict:
        """
        Return a statistical summary of all facts for a document.
        Groups by fact_type with count and total value.
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT
                    fact_type,
                    COUNT(*)       AS count,
                    SUM(fact_value) AS total,
                    AVG(fact_value) AS average,
                    MAX(fact_value) AS maximum
                FROM facts
                WHERE doc_id = ?
                GROUP BY fact_type
                ORDER BY count DESC
            """, (doc_id,)).fetchall()

        return {
            row[0]: {
                "count":   row[1],
                "total":   row[2],
                "average": row[3],
                "maximum": row[4],
            }
            for row in rows
        }

    def list_documents(self) -> list[str]:
        """Return all doc_ids in the FactTable."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT doc_id FROM facts ORDER BY doc_id"
            ).fetchall()
        return [r[0] for r in rows]
