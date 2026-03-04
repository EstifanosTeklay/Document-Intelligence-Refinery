"""
Base interface for all extraction strategies.
Strategy A, B, and C must all implement this contract.
The ExtractionRouter only ever calls this interface — never a concrete class directly.
"""
from abc import ABC, abstractmethod
from pathlib import Path

from src.models import DocumentProfile, ExtractedDocument


class BaseExtractor(ABC):
    """
    Abstract base class for all extraction strategies.
    Every strategy must produce an ExtractedDocument — the normalized schema.
    """

    @abstractmethod
    def extract(
        self,
        file_path: Path,
        profile: DocumentProfile,
    ) -> ExtractedDocument:
        """
        Extract content from a document and return a normalized ExtractedDocument.

        Args:
            file_path: Path to the source document
            profile: DocumentProfile from the Triage Agent

        Returns:
            ExtractedDocument with extraction_confidence set
        """
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return 'A', 'B', or 'C'."""
        ...
