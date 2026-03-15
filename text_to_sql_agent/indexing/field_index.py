"""
FAISS-based field index for semantic similarity search.
Indexes field descriptions for schema linking.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from .embeddings import EmbeddingModel, create_embedding_model


@dataclass
class FieldIndexEntry:
    """Entry in the field index."""
    table: str
    column: str
    description: str
    embedding: Optional[np.ndarray] = None
    
    def key(self) -> Tuple[str, str]:
        """Get (table, column) key."""
        return (self.table, self.column)
    
    def __repr__(self):
        return f"FieldIndexEntry({self.table}.{self.column})"


@dataclass
class SemanticMatch:
    """Result from semantic similarity search."""
    table: str
    column: str
    description: str
    score: float  # Cosine similarity (0-1, higher is better)
    
    def __repr__(self):
        return f"SemanticMatch({self.table}.{self.column}, score={self.score:.3f})"


class FieldIndex:
    """
    FAISS-based index for semantic field search.
    
    Paper-aligned: Uses FAISS over field descriptions (long summaries)
    for semantic similarity matching.
    """
    
    def __init__(
        self, 
        embedding_model: Optional[EmbeddingModel] = None,
        use_gpu: bool = False
    ):
        """
        Initialize field index.
        
        Args:
            embedding_model: Model for embeddings (default: all-MiniLM-L6-v2)
            use_gpu: Whether to use GPU for FAISS (if available)
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss not installed. "
                "Install with: pip install faiss-cpu (or faiss-gpu for GPU support)"
            )
        
        self.faiss = faiss
        self.embedding_model = embedding_model or create_embedding_model()
        self.use_gpu = use_gpu
        
        # Index storage
        self.entries: List[FieldIndexEntry] = []
        self.index: Optional[faiss.Index] = None
        
        # Mapping from FAISS index position to entry
        self.idx_to_entry: Dict[int, FieldIndexEntry] = {}
    
    def build_from_metadata(
        self,
        metadata_list: List,  # List[FieldMetadata]
        use_full_description: bool = True,
        show_progress: bool = True
    ) -> int:
        """
        Build index from FieldMetadata objects.
        
        Args:
            metadata_list: List of FieldMetadata
            use_full_description: If True, use full_description (SME + LLM)
                                 If False, use maximal_description (LLM only)
            show_progress: Show progress bar
            
        Returns:
            Number of fields indexed
        """
        # Extract descriptions
        entries = []
        texts = []
        
        for metadata in metadata_list:
            profile = metadata.profile
            
            # Get description
            if use_full_description:
                description = metadata.full_description
            else:
                description = metadata.maximal_description
            
            if not description:
                continue
            
            entry = FieldIndexEntry(
                table=profile.table_name,
                column=profile.column_name,
                description=description
            )
            entries.append(entry)
            texts.append(description)
        
        if not entries:
            return 0
        
        # Encode descriptions
        if show_progress:
            print(f"Encoding {len(texts)} field descriptions...")
        
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=show_progress
        )
        
        # Store embeddings
        for entry, emb in zip(entries, embeddings):
            entry.embedding = emb
        
        # Build FAISS index
        self._build_faiss_index(entries, embeddings)
        
        return len(entries)
    
    def _build_faiss_index(
        self, 
        entries: List[FieldIndexEntry], 
        embeddings: np.ndarray
    ):
        """Build FAISS index from embeddings."""
        dim = embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (inner product after normalization)
        index = self.faiss.IndexFlatIP(dim)
        
        # Normalize embeddings for cosine similarity
        self.faiss.normalize_L2(embeddings)
        
        # Add to index
        index.add(embeddings.astype('float32'))
        
        # Store
        self.entries = entries
        self.index = index
        self.idx_to_entry = {i: entry for i, entry in enumerate(entries)}
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SemanticMatch]:
        """
        Search for fields semantically similar to query.
        
        Args:
            query: Query text (e.g., user question or extracted phrase)
            top_k: Number of results to return
            
        Returns:
            List of SemanticMatch, sorted by score (highest first)
        """
        if self.index is None:
            return []
        
        if not query.strip():
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            show_progress_bar=False
        )
        
        # Normalize for cosine similarity
        self.faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(top_k, len(self.entries))
        )
        
        # Convert to matches
        matches = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break
            
            entry = self.idx_to_entry[idx]
            matches.append(SemanticMatch(
                table=entry.table,
                column=entry.column,
                description=entry.description,
                score=float(score)  # Inner product of normalized embeddings = cosine similarity
                                    # Range: theoretically [-1, 1], practically [0, 1] for text
            ))
        
        return matches
    
    def __len__(self) -> int:
        """Number of indexed fields."""
        return len(self.entries)
    
    def __repr__(self):
        return f"FieldIndex({len(self)} fields, model={self.embedding_model.model_name})"