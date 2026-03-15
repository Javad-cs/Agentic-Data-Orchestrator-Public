from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class EmbeddingResult:
    """Result from embedding operation"""
    text: str
    embedding: np.ndarray  # Dense vector
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchEmbeddingResult:
    """Result from batch embedding operation"""
    results: List[EmbeddingResult]
    total_tokens: int = 0
    
    @property
    def embeddings(self) -> List[np.ndarray]:
        """Get all embeddings as list"""
        return [r.embedding for r in self.results]
    
    @property
    def texts(self) -> List[str]:
        """Get all texts as list"""
        return [r.text for r in self.results]


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedders.
    
    All embedders must implement embed() and embed_batch() methods.
    """
    
    @abstractmethod
    def embed(self, text: str) -> EmbeddingResult:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with dense vector
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> BatchEmbeddingResult:
        """
        Embed multiple texts in batch (more efficient).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            BatchEmbeddingResult with all embeddings
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension"""
        pass
    
    def validate_text(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Validate and optionally truncate text by tokens (not characters).
        
        Args:
            text: Input text
            max_length: Maximum TOKENS (None = no limit)
            
        Returns:
            Validated text
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        text = text.strip()
        
        if max_length:
            # Truncate by tokens, not characters
            import tiktoken
            encoder = tiktoken.get_encoding('cl100k_base')
            tokens = encoder.encode(text)
            
            if len(tokens) > max_length:
                truncated_tokens = tokens[:max_length]
                text = encoder.decode(truncated_tokens)
        
        return text