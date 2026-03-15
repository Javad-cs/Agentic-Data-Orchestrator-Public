import logging
from typing import Optional

from .base import (
    BaseReranker,
    CandidateDocument,
    RerankResult,
    RerankResponse
)
from .cohere import CohereReranker

logger = logging.getLogger(__name__)


def create_reranker(config) -> Optional[BaseReranker]:
    """
    Factory function to create reranker from config.
    
    Args:
        config: RerankerConfig
        
    Returns:
        BaseReranker instance or None if disabled
    """
    if not config.enabled:
        logger.info("Reranker disabled")
        return None
    
    if config.provider == "cohere":
        logger.info("Creating Cohere reranker")
        return CohereReranker(
            api_key=config.cohere_api_key,
            base_url=config.cohere_base_url,
            model=config.cohere_model,
            top_n=config.top_n
        )
    
    logger.warning(f"Unknown reranker provider: {config.provider}")
    return None


__all__ = [
    "BaseReranker",
    "CandidateDocument",
    "RerankResult",
    "RerankResponse",
    "CohereReranker",
    "create_reranker"
]