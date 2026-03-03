from typing import Optional
from pydantic import BaseModel, Field

from .extracted_document import BoundingBox


# ------------------------------------------------------------
# Single source citation
# ------------------------------------------------------------

class SourceCitation(BaseModel):
    """
    A single traceable reference back to the source document.
    Every claim in a query answer must have at least one of these.
    """
    doc_id: str
    document_name: str
    page_number: int
    bounding_box: Optional[BoundingBox] = None
    content_hash: str = Field(..., description="SHA-256 of the source chunk — verifiable")
    excerpt: str = Field(..., description="Short excerpt from the source (max 200 chars)")
    strategy_used: str = Field(default="", description="A | B | C — extraction strategy")


# ------------------------------------------------------------
# ProvenanceChain — attached to every query answer
# ------------------------------------------------------------

class ProvenanceChain(BaseModel):
    """
    The full audit trail for a single query answer.
    Attached to every response from the Query Agent.
    Answers the question: 'Where exactly does this come from?'
    """
    query: str = Field(..., description="The original user query")
    answer: str = Field(..., description="The answer produced by the Query Agent")
    citations: list[SourceCitation] = Field(
        ...,
        description="Ordered list of sources that support the answer"
    )
    is_verified: bool = Field(
        default=False,
        description="True if every claim maps to a citation"
    )
    unverifiable_claims: list[str] = Field(
        default_factory=list,
        description="Any claims in the answer that could not be sourced"
    )

    class Config:
        use_enum_values = True
