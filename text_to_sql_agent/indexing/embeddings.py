"""
Configurable embedding models for semantic similarity.
Makes it easy to swap between different embedding models.
"""

from typing import List, Protocol
from abc import ABC, abstractmethod
import numpy as np


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""
    
    def encode(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress
            
        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        ...
    
    @property
    def embedding_dim(self) -> int:
        """Dimensionality of embeddings."""
        ...


class SentenceTransformerModel:
    """
    Wrapper for sentence-transformers models.
    
    Default model: all-MiniLM-L6-v2 (384 dims, fast, good quality)
    Alternative: all-mpnet-base-v2 (768 dims, slower, better quality)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize sentence transformer.
        
        Args:
            model_name: HuggingFace model name
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self._embedding_dim
    
    def __repr__(self):
        return f"SentenceTransformerModel(model={self.model_name}, dim={self.embedding_dim})"


def create_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """
    Factory function to create embedding model.
    
    Makes it easy to switch models in one place.
    
    Args:
        model_name: Name of model to use
        
    Returns:
        EmbeddingModel instance
        
    Examples:
        # Fast, good quality (default)
        model = create_embedding_model("all-MiniLM-L6-v2")
        
        # Better quality, slower
        model = create_embedding_model("all-mpnet-base-v2")
        
        # Multilingual
        model = create_embedding_model("paraphrase-multilingual-MiniLM-L12-v2")
    """
    return SentenceTransformerModel(model_name)


# Convenience: Pre-configured models
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384 dims, fast
BEST_QUALITY_MODEL = "all-mpnet-base-v2"  # 768 dims, slower but better
MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 384 dims, multilingual