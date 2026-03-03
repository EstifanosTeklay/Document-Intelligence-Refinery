import hashlib
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator

from .extracted_document import BoundingBox


# ------------------------------------------------------------
# Chunk type — what kind of content this LDU contains
# ------------------------------------------------------------

class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    HEADING = "heading"
    LIST = "list"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    MIXED = "mixed"


# ------------------------------------------------------------
# Cross-reference — resolved "see Table 3" style links
# ------------------------------------------------------------

class ChunkRelationship(BaseModel):
    """A resolved cross-reference between two LDUs."""
    target_chunk_id: str
    relationship_type: str = Field(
        ...,
        description="e.g. 'references', 'continues_from', 'caption_of'"
    )


# ------------------------------------------------------------
# LDU — Logical Document Unit
# ------------------------------------------------------------

class LDU(BaseModel):
    """
    A semantically coherent, self-contained unit of document content.
    This is the atom of the RAG pipeline — what gets embedded and retrieved.

    Every LDU carries full provenance so any claim can be traced back
    to an exact location in the source document.
    """

    # Identity
    chunk_id: str = Field(..., description="Unique ID: {doc_id}_chunk_{index}")
    doc_id: str
    chunk_index: int = Field(..., description="Position in document reading order")

    # Content
    content: str = Field(..., description="The text content of this chunk")
    chunk_type: ChunkType

    # Provenance — the audit trail
    page_refs: list[int] = Field(
        ...,
        description="1-indexed page numbers this chunk spans"
    )
    bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Spatial location of this chunk on the page"
    )
    content_hash: str = Field(
        default="",
        description="SHA-256 of content — enables provenance verification even if pages shift"
    )

    # Structural context — stored as metadata per chunking rules #3 and #4
    parent_section: Optional[str] = Field(
        default=None,
        description="Title of the section this chunk belongs to (chunking rule #4)"
    )
    parent_section_page: Optional[int] = Field(
        default=None,
        description="Page where the parent section heading appears"
    )

    # Token count — needed for RAG retrieval budget
    token_count: int = Field(default=0)

    # Relationships — resolved cross-references (chunking rule #5)
    relationships: list[ChunkRelationship] = Field(default_factory=list)

    # For table chunks — structured data preserved alongside text
    table_data: Optional[dict] = Field(
        default=None,
        description="For chunk_type=TABLE: {headers: [...], rows: [[...]]}"
    )

    # For figure chunks — caption always travels with the figure
    figure_caption: Optional[str] = Field(
        default=None,
        description="For chunk_type=FIGURE: caption stored as metadata (chunking rule #2)"
    )

    # Extraction lineage
    strategy_used: str = Field(default="", description="A | B | C — which extractor produced this")

    @model_validator(mode="after")
    def compute_content_hash(self) -> "LDU":
        """Auto-compute SHA-256 hash of content if not already set."""
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()
        return self

    class Config:
        use_enum_values = True
