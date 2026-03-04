from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .routing import RoutingDecision


# ------------------------------------------------------------
# Building blocks — smallest units first
# ------------------------------------------------------------

class BoundingBox(BaseModel):
    """
    Spatial coordinates of an element on a page.
    Uses pdfplumber's coordinate system: origin (0,0) is bottom-left.
    Units are PDF points (1 point = 1/72 inch).
    """
    x0: float = Field(..., description="Left edge")
    y0: float = Field(..., description="Bottom edge")
    x1: float = Field(..., description="Right edge")
    y1: float = Field(..., description="Top edge")
    page: int = Field(..., description="1-indexed page number")


class TextBlock(BaseModel):
    """A block of continuous text with its spatial location."""
    text: str
    bbox: BoundingBox
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    is_heading: bool = False


class TableCell(BaseModel):
    """A single cell within a table."""
    row: int
    col: int
    text: str
    is_header: bool = False


class ExtractedTable(BaseModel):
    """
    A table extracted as structured data.
    Headers are always stored separately to enforce chunking rule #1.
    """
    table_id: str = Field(..., description="Unique ID within this document")
    bbox: BoundingBox
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list, description="Each row is a list of cell strings")
    cells: list[TableCell] = Field(default_factory=list)
    caption: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ExtractedFigure(BaseModel):
    """A figure or image with its caption."""
    figure_id: str
    bbox: BoundingBox
    caption: Optional[str] = None          # always stored as metadata (chunking rule #2)
    image_bytes: Optional[bytes] = None    # raw image if extracted
    alt_text: Optional[str] = None


# ------------------------------------------------------------
# ExtractedDocument — the normalized contract between
# all 3 extraction strategies and the chunking engine
# ------------------------------------------------------------

class ExtractedDocument(BaseModel):
    """
    The single normalized representation that Strategy A, B, and C
    must all produce. The ChunkingEngine only ever sees this schema —
    never the raw output of pdfplumber, Docling, or a VLM.
    """

    # Identity — linked back to the DocumentProfile
    doc_id: str
    filename: str
    total_pages: int

    # Extraction provenance
    strategy_used: str = Field(..., description="A | B | C")
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    cost_estimate_usd: float = Field(default=0.0)
    processing_time_seconds: float = Field(default=0.0)

    # Content
    text_blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    figures: list[ExtractedFigure] = Field(default_factory=list)

    # Reading order: list of element IDs in the order they should be read
    # Each entry is like "text:0", "table:table_001", "figure:fig_001"
    reading_order: list[str] = Field(default_factory=list)

    # Raw full text per page (fallback for simple queries)
    pages_text: dict[int, str] = Field(
        default_factory=dict,
        description="page_number → full page text"
    )

    # Routing metadata — attached by ExtractionRouter after orchestration
    routing_decision: Optional[dict] = Field(
        default=None,
        description=(
            "RoutingDecision dict attached by ExtractionRouter. "
            "Records strategies attempted, confidence scores, and escalation reason."
        )
    )

    class Config:
        arbitrary_types_allowed = True  # needed for bytes in ExtractedFigure
