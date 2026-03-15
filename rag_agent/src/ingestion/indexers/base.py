from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class IndexerType(Enum):
    """Type of indexer"""
    SPARSE = "sparse"  # BM25, lexical
    DENSE = "dense"    # Vector embeddings


@dataclass
class IngestionDocument:
    """
    Strongly-typed document for indexing.
    
    This is the contract between chunkers/parsers and indexers.
    No more mystery meat Dict[str, Any]!
    """
    # Required fields
    child_id: str
    parent_id: str
    text: str  # For sparse (BM25) indexing
    
    # Optional fields
    vector: Optional[List[float]] = None  # For dense indexing
    chunk_type: str = "text_chunk"
    chunk_index: int = 0
    
    # Source metadata
    source_file: str = ""
    page_number: Optional[int] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate document"""
        if not self.child_id:
            raise ValueError("child_id is required")
        if not self.parent_id:
            raise ValueError("parent_id is required")
        if not self.text and not self.vector:
            raise ValueError("Either text or vector must be provided")


@dataclass
class IndexResult:
    """Result from indexing operation"""
    document_id: str
    success: bool
    indexer_type: IndexerType
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BatchIndexResult:
    """Result from batch indexing operation"""
    results: List[IndexResult]
    total_success: int
    total_failed: int
    indexer_type: IndexerType
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.total_success + self.total_failed
        return self.total_success / total if total > 0 else 0.0
    
    @property
    def failed_ids(self) -> List[str]:
        """Get list of failed document IDs"""
        return [r.document_id for r in self.results if not r.success]


class BaseIndexer(ABC):
    """
    Abstract base class for indexers.
    
    Indexers store processed data in a searchable format.
    """
    
    @abstractmethod
    def get_indexer_type(self) -> IndexerType:
        """Return the type of this indexer"""
        pass
    
    @abstractmethod
    async def index(self, document: IngestionDocument) -> IndexResult:
        """
        Index a single document.
        
        Args:
            document: IngestionDocument with all required fields
            
        Returns:
            IndexResult
        """
        pass
    
    @abstractmethod
    async def index_batch(
        self,
        documents: List[IngestionDocument],
        transaction: bool = True
    ) -> BatchIndexResult:
        """
        Index multiple documents in batch.
        
        Args:
            documents: List of IngestionDocument objects
            transaction: If True, rollback all on any failure
            
        Returns:
            BatchIndexResult
        """
        pass
    
    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """
        Delete a document from index.
        
        Args:
            document_id: Document to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def delete_batch(self, document_ids: List[str]) -> int:
        """
        Delete multiple documents.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            Number of documents actually deleted
        """
        pass