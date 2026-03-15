"""
Core LSH-based lexical matcher.
Generic engine that can be adapted for different use cases.
"""

from typing import List, Dict, Set, Optional, Iterable, Any
from dataclasses import dataclass
from datasketch import MinHash, MinHashLSH
from .shingling import create_shingles, normalize_text, stable_hash


@dataclass
class MatchResult:
    """Result from LSH matching."""
    key: str
    original_value: str  # String representation
    normalized_value: str
    score: float
    
    def __repr__(self):
        return f"Match('{self.original_value}', score={self.score:.3f})"


class LexicalLSHMatcher:
    """
    Core LSH matcher for approximate string matching.
    Generic, type-agnostic engine.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        num_perm: int = 128,
        k: int = 3,
        normalize_separators: bool = True,
    ):
        """
        Initialize matcher.
        
        Args:
            threshold: LSH similarity threshold (0-1)
            num_perm: Number of hash permutations
            k: Shingle size
            normalize_separators: Whether to normalize _ and - to spaces
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.k = k
        self.normalize_separators = normalize_separators
        
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        
        # String-only mappings
        self.key_to_original: Dict[str, str] = {}
        self.key_to_normalized: Dict[str, str] = {}
        self.key_to_shingles: Dict[str, Set[str]] = {}
    
    def index_values(
        self,
        values: Iterable[Any],
        prefix: str = "",
    ) -> int:
        """
        Index a list of values.
        
        Args:
            values: Iterable of values to index (will be converted to strings)
            prefix: Key prefix (use "|" separator, e.g., "table|column")
            
        Returns:
            Number of values indexed
        """
        indexed = 0
        
        for original_value in values:
            # Skip None
            if original_value is None:
                continue
            
            # Convert to string and strip
            str_value = str(original_value).strip()
            if not str_value:
                continue
            
            # Normalize
            normalized = normalize_text(
                str_value, 
                normalize_separators=self.normalize_separators
            )
            if not normalized:
                continue
            
            # Create shingles
            shingles = create_shingles(
                normalized, 
                k=self.k, 
                normalize=False,
                normalize_separators=False,  # Already normalized
            )
            if not shingles:
                continue
            
            # Create stable, deterministic key
            # Use hash of normalized form + hash of original to handle collisions
            # e.g., "L.A." and "LA" both normalize to "la" but have different originals
            norm_hash = stable_hash(normalized)
            orig_hash = stable_hash(str_value)
            
            # Key format: prefix|normalized_hash|original_hash
            if prefix:
                key = f"{prefix}|{norm_hash}|{orig_hash}"
            else:
                key = f"{norm_hash}|{orig_hash}"
            
            # Deduplicate: skip if already indexed
            if key in self.key_to_original:
                continue
            
            # Create MinHash
            m = MinHash(num_perm=self.num_perm)
            for shingle in shingles:
                m.update(shingle.encode('utf-8'))
            
            # Store mappings
            self.key_to_original[key] = str_value
            self.key_to_normalized[key] = normalized
            self.key_to_shingles[key] = shingles
            
            # Insert into LSH
            self.lsh.insert(key, m)
            indexed += 1
        
        return indexed
    
    def query(
        self,
        literal: str,
        top_k: Optional[int] = None,
        exact_rerank: bool = True,
    ) -> List[MatchResult]:
        """
        Find matching values for a query literal.
        
        Args:
            literal: Query string
            top_k: Return top k results (None = all, 0 = none)
            exact_rerank: Whether to rerank with exact Jaccard
            
        Returns:
            List of MatchResult, sorted by score descending
        """
        if literal is None:
            return []
        
        str_literal = str(literal).strip()
        if not str_literal:
            return []
        
        normalized_query = normalize_text(
            str_literal,
            normalize_separators=self.normalize_separators
        )
        if not normalized_query:
            return []
        
        query_shingles = create_shingles(
            normalized_query, 
            k=self.k, 
            normalize=False,
            normalize_separators=False,  # Already normalized
        )
        if not query_shingles:
            return []
        
        # Create MinHash
        m_query = MinHash(num_perm=self.num_perm)
        for shingle in query_shingles:
            m_query.update(shingle.encode('utf-8'))
        
        # Query LSH
        candidate_keys = self.lsh.query(m_query)
        if not candidate_keys:
            return []
        
        # Score candidates
        results = []
        for key in candidate_keys:
            original_value = self.key_to_original[key]
            normalized_value = self.key_to_normalized[key]
            
            if exact_rerank:
                # Exact Jaccard using shingle sets
                candidate_shingles = self.key_to_shingles[key]
                intersection = len(query_shingles & candidate_shingles)
                union = len(query_shingles | candidate_shingles)
                score = intersection / union if union > 0 else 0.0
            else:
                score = self.threshold
            
            results.append(MatchResult(
                key=key,
                original_value=original_value,
                normalized_value=normalized_value,
                score=score,
            ))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top k
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def __len__(self) -> int:
        return len(self.key_to_original)
    
    def __repr__(self):
        return f"LexicalLSHMatcher({len(self)} values, threshold={self.threshold})"