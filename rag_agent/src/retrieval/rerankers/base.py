from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class CandidateDocument:
    """
    Type-safe document for reranking.
    
    No more mystery meat Dict[str, Any]!
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional fields from retrieval
    score: Optional[float] = None  # Original retrieval score
    source: Optional[str] = None   # "dense" or "sparse"
    
    def __post_init__(self):
        """Validate document"""
        if not self.id:
            raise ValueError("Document id cannot be empty")
        if not self.text:
            raise ValueError("Document text cannot be empty")


@dataclass
class RerankResult:
    """Single reranking result"""
    document_id: str
    text: str
    score: float
    original_rank: int
    reranked_rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResponse:
    """Complete reranking response"""
    results: List[RerankResult]
    query: str
    total_candidates: int
    total_reranked: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def top_result(self) -> Optional[RerankResult]:
        """Get top-ranked result"""
        return self.results[0] if self.results else None
    
    @property
    def document_ids(self) -> List[str]:
        """Get list of document IDs in reranked order"""
        return [r.document_id for r in self.results]
    
    @property
    def scores(self) -> List[float]:
        """Get list of scores in reranked order"""
        return [r.score for r in self.results]


class BaseReranker(ABC):
    """
    Abstract base class for rerankers.
    
    Rerankers take an initial set of retrieved documents
    and reorder them based on relevance to the query.
    
    IMPORTANT: All methods are async to avoid blocking the event loop.
    """
    
    @abstractmethod
    async def rerank(  # ← ASYNC!
        self,
        query: str,
        documents: List[CandidateDocument],  # ← Type-safe!
        top_k: Optional[int] = None
    ) -> RerankResponse:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of CandidateDocument objects
            top_k: Number of top results to return (None = all)
            
        Returns:
            RerankResponse with reranked results
        """
        pass
    
    @abstractmethod
    def get_reranker_name(self) -> str:
        """Return the name/model of this reranker"""
        pass