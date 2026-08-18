import logging
import requests
import numpy as np
from typing import List, Optional, Literal
import time
import tiktoken

from .base import BaseEmbedder, EmbeddingResult, BatchEmbeddingResult
from src.config.models import UpstageConfig

logger = logging.getLogger(__name__)


class UpstageEmbedder(BaseEmbedder):
    """
    Upstage Embedding API integration.
    
    Supports dual-encoder setup:
    - Passage mode: For indexing documents (solar-embedding-1-large-passage)
    - Query mode: For search queries (solar-embedding-1-large-query)
    
    API Reference:
    https://developers.upstage.ai/docs/apis/embeddings
    
    Features:
    - Dual-encoder support (passage/query)
    - Batch embedding (up to 100 texts per request)
    - Automatic retry on rate limits
    - Token usage tracking
    """
    
    API_ENDPOINT = "https://api.upstage.ai/v1/solar/embeddings"
    MAX_BATCH_SIZE = 100  # Upstage limit
    MAX_TOKENS = 3800 # Safe margin below upstage limit 4000
    
    def __init__(
        self,
        config: Optional[UpstageConfig] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        mode: Literal["passage", "query"] = "passage",  # ← Use Literal for type safety
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Upstage embedder.
        
        Can be initialized in two ways:
        1. With config object (recommended):
           # For indexing documents
           embedder = UpstageEmbedder(config=system_config.upstage, mode="passage")
           
           # For encoding search queries
           embedder = UpstageEmbedder(config=system_config.upstage, mode="query")
        
        2. With individual parameters:
           embedder = UpstageEmbedder(
               api_key="xxx",
               mode="passage"  # or "query"
           )
        
        Args:
            config: UpstageConfig object (takes precedence)
            api_key: Upstage API key (used if config not provided)
            model: Model name (overrides mode if provided)
            dimension: Expected embedding dimension (default 4096)
            mode: "passage" for documents, "query" for search queries
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
        """
        # Priority: config > individual parameters
        if config:
            self.api_key = config.api_key
            
            # If model explicitly provided, use it
            if model:
                self.model = model
            else:
                # Otherwise, use mode to select model
                if mode == "query":
                    self.model = config.embedding_model_query
                elif mode == "passage":
                    self.model = config.embedding_model_passage
                else:
                    raise ValueError(f"Invalid mode: {mode}. Must be 'passage' or 'query'")
            
            self.dimension = config.embedding_dimension
        else:
            if not api_key:
                raise ValueError("Either config or api_key must be provided")
            
            self.api_key = api_key
            
            # If model explicitly provided, use it
            if model:
                self.model = model
            else:
                # Otherwise, use mode to select default model
                if mode == "query":
                    self.model = "solar-embedding-1-large-query"
                elif mode == "passage":
                    self.model = "solar-embedding-1-large-passage"
                else:
                    raise ValueError(f"Invalid mode: {mode}. Must be 'passage' or 'query'")
            
            self.dimension = dimension or 4096
        
        self.mode = mode  # ← Store for logging
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Statistics
        self.total_tokens_used = 0
        self.total_requests = 0
        
        try:
            from transformers import AutoTokenizer
            self.upstage_tokenizer = AutoTokenizer.from_pretrained("upstage/solar-1-mini-tokenizer")
            logger.info(" Loaded Upstage solar-1-mini-tokenizer for accurate token counting")
        except Exception as e:
            logger.warning(f" Failed to load Upstage tokenizer: {e}. Falling back to cl100k_base with safety margin")
            self.upstage_tokenizer = None
        
        logger.info(f"Initialized UpstageEmbedder: mode={mode}, model={self.model}")
    
    def get_dimension(self) -> int:
        """Return embedding dimension"""
        return self.dimension
    
    def _count_tokens_upstage(self, text: str) -> int:
            """Count tokens using Upstage's actual tokenizer."""
            if self.upstage_tokenizer:
                return len(self.upstage_tokenizer.encode(text))
            # Fallback: cl100k_base with conservative multiplier for Korean
            encoder = tiktoken.get_encoding('cl100k_base')
            return int(len(encoder.encode(text)) * 1.4)

    def validate_text(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Validate and truncate text using Upstage's actual tokenizer.
        Overrides base class to ensure accurate token counting.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        text = text.strip()
        
        if max_length:
            if self.upstage_tokenizer:
                tokens = self.upstage_tokenizer.encode(text)
                if len(tokens) > max_length:
                    text = self.upstage_tokenizer.decode(tokens[:max_length], skip_special_tokens=True)
                    logger.debug(f"Truncated text from {len(tokens)} to ~{max_length} Upstage tokens")
            else:
                import tiktoken
                encoder = tiktoken.get_encoding('cl100k_base')
                safe_limit = int(max_length / 1.4)
                tokens = encoder.encode(text)
                if len(tokens) > safe_limit:
                    text = encoder.decode(tokens[:safe_limit])
        
        return text
    
    def embed(self, text: str) -> EmbeddingResult:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with 4096-dimensional vector
        """
        # Validate text
        text = self.validate_text(text, max_length=self.MAX_TOKENS)
        
        # Use batch endpoint with single text
        batch_result = self.embed_batch([text])
        
        return batch_result.results[0]
    
    def embed_batch(self, texts: List[str]) -> BatchEmbeddingResult:
        """
        Embed multiple texts in batch.
        
        Automatically splits into multiple requests if batch size exceeds limit.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            BatchEmbeddingResult with all embeddings
        """
        if not texts:
            raise ValueError("texts cannot be empty")
        
        texts = [self.validate_text(t, max_length=self.MAX_TOKENS) for t in texts]
        
        all_results = []
        total_tokens = 0
        
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch_texts = texts[i:i + self.MAX_BATCH_SIZE]
            batch_embeddings, tokens_used = self._call_api_with_retry(batch_texts)
            
            for text, embedding in zip(batch_texts, batch_embeddings):
                all_results.append(EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    metadata={'model': self.model, 'dimension': self.dimension}
                ))
            total_tokens += tokens_used
        
        return BatchEmbeddingResult(results=all_results, total_tokens=total_tokens)
        
    def _split_into_token_safe_batches(self, texts: List[str]) -> List[List[str]]:
        """Split texts into batches that respect both MAX_BATCH_SIZE and API token limit."""
        encoder = tiktoken.get_encoding('cl100k_base')
        
        API_TOKEN_LIMIT = 3800  # Safe margin below 4000
        batches = []
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            text_tokens = len(encoder.encode(text))
            
            # If single text fills the limit, send it alone
            if text_tokens >= API_TOKEN_LIMIT:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                batches.append([text])
                continue
            
            # Would exceed token limit or batch size?
            if (current_tokens + text_tokens > API_TOKEN_LIMIT or 
                    len(current_batch) >= self.MAX_BATCH_SIZE):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(text)
            current_tokens += text_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

    
    def _call_api_with_retry(self, texts: List[str]) -> tuple[List[np.ndarray], int]:
        """
        Call Upstage API with retry logic.
        
        Returns:
            (embeddings, tokens_used) tuple
        """
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                return self._call_api(texts)
            
            except requests.exceptions.HTTPError as e:
                last_error = e
                
                # Check if rate limit error (429)
                if hasattr(e, 'response') and e.response and e.response.status_code == 429:
                    if attempt < self.retry_attempts - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"️ Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{self.retry_attempts}...")
                        time.sleep(wait_time)
                        continue
                
                # Other HTTP errors, don't retry
                raise
            
            except requests.exceptions.RequestException as e:
                last_error = e
                
                # Network errors, retry
                if attempt < self.retry_attempts - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"️ Network error. Retrying in {wait_time}s... ({attempt + 1}/{self.retry_attempts})")
                    time.sleep(wait_time)
                    continue
                
                raise
        
        # All retries exhausted
        raise Exception(f"Failed after {self.retry_attempts} attempts: {last_error}")
    
    def _call_api(self, texts: List[str]) -> tuple[List[np.ndarray], int]:
        """
        Make actual API call to Upstage.
        
        Returns:
            (embeddings, tokens_used) tuple
        """
        # Prepare request
        payload = {
            "model": self.model,
            "input": texts
        }
        
        logger.debug(f"Sending to Upstage Embeddings API:")
        logger.debug(f"  Model: {self.model}")
        logger.debug(f"  Number of texts: {len(texts)}")
        logger.debug(f"  First text preview: {texts[0][:100] if texts else 'N/A'}...")
    
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Make request
        response = requests.post(
            self.API_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=self.timeout
        )
        
        # Handle errors
        if response.status_code != 200:
            error_msg = f"Upstage API error: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f" - {error_detail}"  # ← This will show the actual error
                logger.error(f"API Error Details: {error_detail}")  # ← ADD THIS
            except:
                error_msg += f" - {response.text[:200]}"
                logger.error(f"API Error Text: {response.text[:500]}")  # ← ADD THIS
            
            response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Extract embeddings
        embeddings = []
        for item in result['data']:
            embedding = np.array(item['embedding'], dtype=np.float32)
            
            # Validate dimension
            if len(embedding) != self.dimension:
                raise ValueError(
                    f"Unexpected embedding dimension: {len(embedding)} (expected {self.dimension})"
                )
            
            embeddings.append(embedding)
        
        # Extract token usage
        tokens_used = result.get('usage', {}).get('total_tokens', 0)
        
        # Update statistics
        self.total_tokens_used += tokens_used
        self.total_requests += 1
        
        return embeddings, tokens_used
    
    def get_stats(self) -> dict:
        """Get usage statistics"""
        return {
            'total_requests': self.total_requests,
            'total_tokens_used': self.total_tokens_used,
            'model': self.model,
            'dimension': self.dimension
        }
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.total_tokens_used = 0
        self.total_requests = 0