from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ------------------------------------------------------------
# Enums — all valid classification values
# ------------------------------------------------------------

class OriginType(str, Enum):
    NATIVE_DIGITAL = "native_digital"
    SCANNED_IMAGE = "scanned_image"
    MIXED = "mixed"
    FORM_FILLABLE = "form_fillable"


class LayoutComplexity(str, Enum):
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"
    FIGURE_HEAVY = "figure_heavy"
    MIXED = "mixed"


class DomainHint(str, Enum):
    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    GENERAL = "general"


class ExtractionCost(str, Enum):
    FAST_TEXT_SUFFICIENT = "fast_text_sufficient"       # Strategy A
    NEEDS_LAYOUT_MODEL = "needs_layout_model"           # Strategy B
    NEEDS_VISION_MODEL = "needs_vision_model"           # Strategy C


# ------------------------------------------------------------
# DocumentProfile — the Triage Agent's output
# ------------------------------------------------------------

class DocumentProfile(BaseModel):
    """
    Classification profile produced by the Triage Agent for every document.
    Governs which extraction strategy all downstream stages will use.
    Stored at: .refinery/profiles/{doc_id}.json
    """

    # Identity
    doc_id: str = Field(..., description="Unique document identifier (stem of filename)")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Absolute path to the source file")
    total_pages: int = Field(..., description="Total page count")

    # Classification dimensions
    origin_type: OriginType = Field(..., description="How the document was created")
    layout_complexity: LayoutComplexity = Field(..., description="Visual/structural complexity")
    language: str = Field(default="en", description="Detected language code (ISO 639-1)")
    language_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    domain_hint: DomainHint = Field(default=DomainHint.GENERAL)
    estimated_extraction_cost: ExtractionCost = Field(...)

    # Diagnostic signals (raw measurements that drove the classification)
    avg_chars_per_page: float = Field(default=0.0, description="Average character count per page")
    avg_image_area_ratio: float = Field(default=0.0, description="Average image area as fraction of page")
    scanned_page_count: int = Field(default=0, description="Number of pages classified as scanned")
    digital_page_count: int = Field(default=0, description="Number of pages classified as digital")
    table_page_count: int = Field(default=0, description="Number of pages containing tables")
    figure_page_count: int = Field(default=0, description="Number of pages containing figures")

    # Confidence in the triage classification itself
    triage_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="How confident the Triage Agent is in this profile"
    )
    triage_notes: Optional[str] = Field(
        default=None,
        description="Human-readable notes about edge cases or uncertainty"
    )

    class Config:
        use_enum_values = True
