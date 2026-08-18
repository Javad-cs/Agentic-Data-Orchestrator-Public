import logging
from typing import List, Optional
import asyncio

from .base import BaseReranker, CandidateDocument, RerankResult, RerankResponse

logger = logging.getLogger(__name__)


class CohereReranker(BaseReranker):
    """
    Cohere reranker via Azure AI Foundry.
    
    Uses Azure's /providers/cohere/v2/rerank endpoint directly.
    Based on working example from colleague.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "Cohere-rerank-v4.0-fast",
        top_n: int = 5
    ):
        """
        Initialize Cohere reranker.
        
        Args:
            api_key: Azure API key
            base_url: Azure endpoint (e.g., https://xxx.services.ai.azure.com)
            model: Model name (Cohere-rerank-v4.0-fast)
            top_n: Default number of results to return
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.top_n = top_n
        
        logger.info(f"CohereReranker initialized (model={model}, top_n={top_n})")
    
    async def rerank(
        self,
        query: str,
        documents: List[CandidateDocument],
        top_k: Optional[int] = None
    ) -> RerankResponse:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of CandidateDocument objects
            top_k: Number of top results to return (None = use default top_n)
            
        Returns:
            RerankResponse with reranked results
        """
        if not documents:
            logger.warning("No documents to rerank")
            return RerankResponse(
                results=[],
                query=query,
                total_candidates=0,
                total_reranked=0,
                metadata={"reranker": self.model}
            )
        
        top_k = top_k or self.top_n
        
        # Extract text from CandidateDocument objects
        doc_texts = [doc.text for doc in documents]
        
        logger.debug(f"Reranking {len(doc_texts)} documents for query: {query[:50]}...")
        
        try:
            # Azure AI Foundry format
            import requests
            
            url = f"{self.base_url}/providers/cohere/v2/rerank"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            data = {
                "model": self.model,
                "query": query,
                "documents": doc_texts,
                "top_n": min(top_k, len(doc_texts))
            }
            
            # Use get_running_loop() instead of get_event_loop()
            loop = asyncio.get_running_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, headers=headers, json=data, timeout=30)
            )
            
            response.raise_for_status()
            result_data = response.json()
            
            # Parse results
            # Response format: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
            results = []
            
            for reranked_position, item in enumerate(result_data.get("results", [])):
                original_index = item.get("index")
                score = item.get("relevance_score")
                
                # Validate required fields
                if original_index is None or score is None:
                    logger.warning(f"Invalid result item: {item}")
                    continue
                
                if original_index < len(documents):
                    original_doc = documents[original_index]
                    
                    # Rich metadata propagation
                    results.append(RerankResult(
                        document_id=original_doc.id,
                        text=original_doc.text,
                        score=score,
                        original_rank=original_index,
                        reranked_rank=reranked_position,
                        metadata={
                            **original_doc.metadata, 
                            "original_score": original_doc.score, 
                            "original_source": original_doc.source  
                        }
                    ))
            
            logger.info(
                f"Reranked {len(results)} documents "
                f"(top score: {results[0].score:.3f})" if results else "Reranked 0 documents"
            )
            
            # Full response metadata
            return RerankResponse(
                results=results,
                query=query,
                total_candidates=len(documents),
                total_reranked=len(results),
                metadata={
                    "reranker": self.model,
                    "provider": "cohere", 
                    "base_url": self.base_url 
                }
            )
        
        except Exception as e:
            logger.error(f"Reranking error: {e}", exc_info=True)
            logger.warning("Falling back to original order")
            
            # Fallback: return original order with synthetic scores
            results = []
            for i, doc in enumerate(documents[:top_k]):
                # Rich metadata in fallback too
                results.append(RerankResult(
                    document_id=doc.id,
                    text=doc.text,
                    score=1.0 - (i * 0.1), 
                    original_rank=i,
                    reranked_rank=i,
                    metadata={
                        **doc.metadata,
                        "original_score": doc.score, 
                        "fallback": True
                    }
                ))
            
            # Full error metadata
            return RerankResponse(
                results=results,
                query=query,
                total_candidates=len(documents),
                total_reranked=len(results),
                metadata={
                    "reranker": self.model,
                    "provider": "cohere",
                    "error": str(e),
                    "fallback": True
                }
            )
    
    def get_reranker_name(self) -> str:
        """
        Return the name/model of this reranker.
        """
        return f"cohere/{self.model}"