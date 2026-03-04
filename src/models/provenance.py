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

    # Spatial provenance — bbox on the citation itself (rubric requirement)
    bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Bounding box of the source content on the page"
    )

    # LDU linkage — ties citation back to the exact chunk
    chunk_id: str = Field(..., description="ID of the source LDU chunk")
    content_hash: str = Field(..., description="SHA-256 of the source chunk — verifiable")

    excerpt: str = Field(..., description="Short excerpt from the source (max 200 chars)")
    strategy_used: str = Field(default="", description="A | B | C — extraction strategy")

    # PageIndex linkage — which section this came from
    section_title: Optional[str] = Field(
        default=None,
        description="Title of the PageIndex section this chunk belongs to"
    )
    section_node_id: Optional[str] = Field(
        default=None,
        description="PageIndex node ID for navigation back to section"
    )


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

    # Primary bbox — direct top-level field for the most relevant source
    # This is the bbox of the first / highest-confidence citation
    bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Bounding box of the primary source on its page — direct top-level field for audit access"
    )
    primary_page: Optional[int] = Field(
        default=None,
        description="Page number of the primary source citation"
    )
    primary_document: Optional[str] = Field(
        default=None,
        description="Filename of the primary source document"
    )

    citations: list[SourceCitation] = Field(
        ...,
        description="Ordered list of all sources that support the answer"
    )
    is_verified: bool = Field(
        default=False,
        description="True if every claim maps to a citation"
    )
    unverifiable_claims: list[str] = Field(
        default_factory=list,
        description="Any claims in the answer that could not be sourced"
    )

    # Audit mode fields
    verified_chunk_ids: list[str] = Field(
        default_factory=list,
        description="chunk_ids successfully verified against source"
    )
    pageindex_nodes_traversed: list[str] = Field(
        default_factory=list,
        description="PageIndex node IDs visited during retrieval"
    )

    def set_primary_from_citations(self) -> None:
        """
        Convenience method: populate top-level bbox, primary_page,
        and primary_document from the first citation.
        Call this after building the citations list.
        """
        if self.citations:
            first = self.citations[0]
            self.bbox = first.bbox
            self.primary_page = first.page_number
            self.primary_document = first.document_name

    class Config:
        use_enum_values = True
