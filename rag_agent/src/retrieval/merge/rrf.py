from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    """Single retrieval candidate with scores from different sources"""
    document_id: str
    text: str
    
    # Scores from different retrievers
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    
    # Rankings
    dense_rank: int = 0
    sparse_rank: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
    weights: Optional[Dict[str, float]] = None
) -> List[RetrievalCandidate]:
    """
    Merge dense and sparse retrieval results using Reciprocal Rank Fusion.
    
    RRF Formula:
        RRF_score(d) = Σ (weight_i / (k + rank_i(d)))
    
    Where:
        - k is a constant (typically 60)
        - rank_i(d) is the rank of document d in retriever i
        - weight_i is the weight for retriever i
    
    Args:
        dense_results: Results from dense retriever (Milvus)
            Format: [{'child_id': '...', 'score': 0.95, 'text': '...'}, ...]
        sparse_results: Results from sparse retriever (BM25)
            Format: [{'child_id': '...', 'score': 5.2, 'text': '...'}, ...]
        k: RRF constant (default 60)
        weights: Weights for each retriever
            Format: {'dense': 0.7, 'sparse': 0.3}
            If None or partial, defaults are used
    
    Returns:
        List of RetrievalCandidate objects sorted by RRF score (highest first)
    
    References:
        Cormack et al. (2009) "Reciprocal Rank Fusion outperforms Condorcet and
        individual Rank Learning Methods"
    """
    # Handle missing or partial weights safely
    if weights is None:
        weights = {}
    
    # Use .get() with defaults to avoid KeyError
    dense_weight = weights.get('dense', 0.5)
    sparse_weight = weights.get('sparse', 0.5)
    
    # Validate weights
    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError(
            f"Weights must be non-negative: dense={dense_weight}, sparse={sparse_weight}"
        )
    
    if dense_weight == 0 and sparse_weight == 0:
        raise ValueError("At least one weight must be non-zero")
    
    # Normalize weights to sum to 1
    total_weight = dense_weight + sparse_weight
    dense_weight /= total_weight
    sparse_weight /= total_weight
    
    logger.debug(f"RRF weights: dense={dense_weight:.2f}, sparse={sparse_weight:.2f}, k={k}")
    
    # Build lookup by document_id
    candidates: Dict[str, RetrievalCandidate] = {}
    
    # Track skipped documents
    dense_empty_count = 0
    sparse_empty_count = 0
    
    # Process dense results
    for rank, result in enumerate(dense_results, start=1):
        doc_id = result['child_id']
        text = result.get('text', '').strip()
        
        # Only skip if BOTH empty AND new document
        if not text:
            dense_empty_count += 1
            logger.debug(f"Dense result {doc_id} (rank {rank}) has empty text")
            # Don't create candidate yet - wait to see if sparse has text
            # But don't continue - we might still want the rank!
        
        if doc_id not in candidates:
            # Only create if we have text
            if text:
                candidates[doc_id] = RetrievalCandidate(
                    document_id=doc_id,
                    text=text,
                    metadata=result.get('metadata', {})
                )
            else:
                # No text from dense, but we'll create a placeholder
                # in case sparse provides text
                candidates[doc_id] = RetrievalCandidate(
                    document_id=doc_id,
                    text="",  # Empty for now
                    metadata=result.get('metadata', {})
                )
        
        # Always update scores and ranks (even if text is empty)
        candidates[doc_id].dense_score = result.get('score', 0.0)
        candidates[doc_id].dense_rank = rank
        candidates[doc_id].rrf_score += dense_weight / (k + rank)
    
    # Process sparse results
    for rank, result in enumerate(sparse_results, start=1):
        doc_id = result['child_id']
        text = result.get('text', '').strip()
        
        # Check if document already exists
        if not text:
            sparse_empty_count += 1
            logger.debug(f"Sparse result {doc_id} (rank {rank}) has empty text")
            
            # If document doesn't exist yet, skip it (truly useless)
            if doc_id not in candidates:
                logger.debug(f"Skipping {doc_id}: empty text and not in dense results")
                continue  # ← Only skip if it's a NEW document with empty text
            
            # Document exists from dense - keep the rank, but note the issue
            logger.debug(
                f"Preserving {doc_id} rank from sparse (rank {rank}) "
                f"despite empty text (has text from dense)"
            )
        
        if doc_id not in candidates:
            # New candidate from sparse only
            if text:  # Only create if we have text
                candidates[doc_id] = RetrievalCandidate(
                    document_id=doc_id,
                    text=text,
                    metadata=result.get('metadata', {})
                )
            else:
                # Skip - sparse-only document with no text
                continue
        else:
            # Document exists - update text if current is empty but new one isn't
            if not candidates[doc_id].text and text:
                candidates[doc_id].text = text
                logger.debug(f"Updated empty text for {doc_id} from sparse results")
        
        # Always update scores and ranks (even if text is empty)
        candidates[doc_id].sparse_score = result.get('score', 0.0)
        candidates[doc_id].sparse_rank = rank
        candidates[doc_id].rrf_score += sparse_weight / (k + rank)
    
    # Final validation - remove any with empty text
    valid_candidates = {
        doc_id: candidate
        for doc_id, candidate in candidates.items()
        if candidate.text and candidate.text.strip()
    }
    
    empty_count = len(candidates) - len(valid_candidates)
    
    # Logging summary
    if dense_empty_count > 0:
        logger.warning(f"Dense retrieval: {dense_empty_count} results had empty text")
    if sparse_empty_count > 0:
        logger.warning(f"Sparse retrieval: {sparse_empty_count} results had empty text")
    if empty_count > 0:
        logger.warning(
            f"RRF: Removed {empty_count} candidates with empty text after merge "
            f"(no valid text from either retriever)"
        )
    
    # Sort by RRF score (highest first)
    sorted_candidates = sorted(
        valid_candidates.values(),
        key=lambda x: x.rrf_score,
        reverse=True
    )
    
    logger.debug(f"RRF merged {len(sorted_candidates)} valid candidates")
    
    return sorted_candidates