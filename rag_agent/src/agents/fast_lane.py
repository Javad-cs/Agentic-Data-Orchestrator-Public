# src/agents/fast_lane.py

import logging
import time
import asyncio
import hashlib
from typing import AsyncIterator, Dict, Any, Optional, List

from src.config.models import SystemConfig
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_expansion import create_query_expander
from src.retrieval.rerankers import create_reranker, CandidateDocument
from src.generation import (
    create_llm_client,
    create_safety_checker,
    create_streaming_generator,
    EventType
)

logger = logging.getLogger(__name__)


class FastLane:
    """
    Fast Lane orchestrator for RAG system.
    
    Pipeline:
    1. [Optional] Query expansion (generate semantic variants)
    2. Retrieve context (hybrid search: dense + sparse + RRF)
    3. [Optional] Reranking (semantic reranking)
    4. Generate answer with citations (streaming)
    5. Safety check (heuristics + optional NLI)
    6. Stream SSE events to frontend
    
    Design goals:
    - Low latency (<4s target)
    - Streaming responses
    - Citation-backed answers
    - Safety guarantees
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize Fast Lane.
        
        Args:
            config: SystemConfig with all settings
        """
        self.config = config
        
        # Components (initialized in initialize())
        self.retriever: Optional[HybridRetriever] = None
        self.llm_client = None
        self.streaming_generator = None
        self.query_expander = None 
        self.reranker = None
        
        # State
        self.initialized = False
        
        logger.info("FastLane created (not initialized)")
    
    async def initialize(self):
        """Initialize all components asynchronously"""
        if self.initialized:
            logger.warning("FastLane already initialized")
            return
        
        logger.info("Initializing Fast Lane...")
        start_time = time.time()
        
        # 1. Initialize retriever
        self.retriever = HybridRetriever(self.config)
        await self.retriever.initialize()
        
        # 2. Initialize LLM client
        self.llm_client = create_llm_client(self.config.llm)
        
        # 3. Initialize generation components
        # create_safety_checker wraps NLI safety checker when configured
        safety_checker = None
        
        if self.config.fast_lane.safety_check.enabled:
            safety_checker = create_safety_checker(self.config.fast_lane.safety_check)
            logger.info(f"Safety checker initialized (NLI={self.config.fast_lane.safety_check.use_nli})")
        
        # Save safety_checker reference (needed for conditional use in tool mode)
        self.safety_checker = safety_checker
        
        self.streaming_generator = create_streaming_generator(
            llm_client=self.llm_client,
            safety_checker=safety_checker
        )
        
        # 4.Initialize query expander (optional)
        if self.config.fast_lane.query_expansion.enabled:
            self.query_expander = create_query_expander(
                llm_client=self.llm_client,
                config=self.config.fast_lane.query_expansion
            )
            logger.info(" Query expander enabled")
        else:
            logger.info("Query expander disabled")
        
        # 5. Initialize reranker (optional)
        if self.config.fast_lane.reranker.enabled:
            self.reranker = create_reranker(self.config.fast_lane.reranker)
            logger.info(" Reranker enabled")
        else:
            logger.info("Reranker disabled")
        
        self.initialized = True
        
        elapsed = time.time() - start_time
        logger.info(f" Fast Lane initialized in {elapsed:.2f}s")
    
    async def query(
        self,
        query: str,
        top_k: int = 5,
        language: str = "ko",
        streaming: bool = True
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process query through Fast Lane pipeline.
        
        Args:
            query: User query
            top_k: Number of final results to use for generation
            language: Response language ("ko" or "en")
            streaming: Enable streaming (always True for Fast Lane)
            
        Yields:
            SSE event dicts
        """
        if not self.initialized:
            raise RuntimeError("FastLane not initialized. Call initialize() first.")
        
        logger.info(f"Fast Lane query: {query[:50]}...")
        pipeline_start = time.time()
        
        try:
            # Step 1: Query Expansion (optional)
            queries = await self._expand_query(query, language)
            
            # Step 2: Retrieval
            if len(queries) > 1:
                yield self._create_event(
                    EventType.STATUS,
                    f"문서 검색 중... ({len(queries)}개 쿼리)" if language == "ko"
                    else f"Searching documents... ({len(queries)} queries)"
                )
            else:
                yield self._create_event(EventType.STATUS, "문서 검색 중...")
            
            retrieval_start = time.time()
            
            # Retrieve with all queries
            retrieval_results = await self._retrieve_all_queries(
                queries=queries,
                top_k=top_k
            )
            
            retrieval_time = time.time() - retrieval_start
            logger.info(f"Retrieved {len(retrieval_results)} results in {retrieval_time:.2f}s")
            
            if not retrieval_results:
                yield self._create_event(
                    EventType.ERROR,
                    data={
                        "message": "관련 문서를 찾을 수 없습니다." if language == "ko" 
                                   else "No relevant documents found.",
                        "type": "no_results"
                    }
                )
                return
            
            # Step 3: Reranking (optional)
            final_results = await self._rerank_documents(
                query=query,
                documents=retrieval_results,
                top_k=top_k,
                language=language
            )
            
            # Step 4: Generate with streaming
            # Note: StreamingGenerator emits its own status events, no need for duplicate here
            generation_start = time.time()
            
            async for event in self.streaming_generator.generate_with_citations(
                query=query,
                context_chunks=final_results,
                language=language
            ):
                yield event
            
            generation_time = time.time() - generation_start
            
            # Step 5: Log final metrics
            total_time = time.time() - pipeline_start
            logger.info(
                f"Fast Lane complete: retrieval={retrieval_time:.2f}s, "
                f"generation={generation_time:.2f}s, total={total_time:.2f}s"
            )
        
        except Exception as e:
            logger.error(f"Fast Lane error: {e}", exc_info=True)
            yield self._create_event(
                EventType.ERROR,
                data={
                    "message": str(e),
                    "type": type(e).__name__
                }
            )
    
    async def _expand_query(
        self,
        query: str,
        language: str
    ) -> List[str]:
        """
        Expand query into variants (NEW).
        
        Returns:
            List of queries [original, variant1, variant2, ...]
        """
        if not self.query_expander:
            return [query]
        
        logger.debug("Expanding query...")
        result = await self.query_expander.expand(query, language)
        
        logger.info(
            f"Query expansion: {result.total_queries} queries "
            f"({result.latency_ms}ms, success={result.success})"
        )
        
        return result.all_queries
    
    async def _retrieve_all_queries(
        self,
        queries: List[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve for all queries and merge with RRF (NEW).
        
        Args:
            queries: List of queries
            top_k: Final number of results needed
            
        Returns:
            Merged results
        """
        if len(queries) == 1:
            # Single query: use original path
            return await self.retriever.retrieve(
                query=queries[0],
                top_k=top_k,
                dense_top_k=self.config.fast_lane.retrieval.dense_top_k,
                sparse_top_k=self.config.fast_lane.retrieval.bm25_top_k,
                rrf_k=self.config.fast_lane.retrieval.rrf_k,
                rerank=False,
                include_parent=True  # preserve parent context
            )
        
        # Multiple queries: retrieve more per query for better coverage
        num_queries = len(queries)
        per_query_k = top_k * num_queries  # Scale with number of queries
        
        # Keep more docs for reranking if enabled
        merge_keep = (top_k * 4) if self.reranker else (top_k * 2)
        
        logger.debug(
            f"Multi-query retrieval: {num_queries} queries, "
            f"per_query_k={per_query_k}, merge_keep={merge_keep}, final={top_k}"
        )
        
        tasks = [
            self.retriever.retrieve(
                query=q,
                top_k=per_query_k,  # retrieve more per query
                dense_top_k=self.config.fast_lane.retrieval.dense_top_k,
                sparse_top_k=self.config.fast_lane.retrieval.bm25_top_k,
                rrf_k=self.config.fast_lane.retrieval.rrf_k,
                rerank=False,
                include_parent=True
            )
            for q in queries
        ]
        
        # Handle exceptions properly
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                logger.warning(f"Query {i+1} retrieval failed: {result}")
            else:
                valid_results.append(result)
        
        if not valid_results:
            logger.error("All retrieval queries failed")
            return []
        
        # Merge with RRF
        merged = self._merge_with_rrf(valid_results)
        
        logger.info(
            f"Retrieved {sum(len(r) for r in valid_results)} docs total, "
            f"merged to {len(merged)} unique docs, keeping top {merge_keep}"
        )
        
        return merged[:merge_keep]
    
    def _merge_with_rrf(
        self,
        all_results: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Merge multiple result lists using RRF (NEW).
        
        Args:
            all_results: List of result lists
            
        Returns:
            Merged and sorted results
        """
        k = self.config.fast_lane.retrieval.rrf_k  # use config
        
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        
        for results in all_results:
            for rank, doc in enumerate(results, 1):
                # Stable fallback for missing IDs
                doc_id = doc.get('child_id') or doc.get('id')
                
                if not doc_id:
                    # Generate stable hash from text
                    text = (doc.get('child_text') or '').encode('utf-8')
                    doc_id = 'fallback_' + hashlib.sha1(text[:5000]).hexdigest()
                
                # RRF score
                rrf_score = 1.0 / (k + rank)
                
                if doc_id in doc_scores:
                    doc_scores[doc_id] += rrf_score
                else:
                    doc_scores[doc_id] = rrf_score
                    doc_map[doc_id] = doc
        
        # Sort by RRF score
        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        return [doc_map[doc_id] for doc_id in sorted_ids]
    
    async def _rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int,
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents (optional, NEW).
        
        Args:
            query: Original query
            documents: Documents to rerank
            top_k: Number to return
            language: Query language
            
        Returns:
            Reranked or original documents
        """
        if not self.reranker:
            return documents[:top_k]
        
        logger.debug(f"Reranking {len(documents)} documents...")
        
        # Convert to CandidateDocument
        candidates = []
        for i, doc in enumerate(documents):
            text = doc.get('parent_text') or doc.get('child_text', '')
            candidates.append(CandidateDocument(
                id=doc.get('child_id') or f'doc_{i}',
                text=text,
                metadata=doc,
                score=doc.get('score'),
                source=doc.get('source')
            ))
        
        # Rerank
        try:
            rerank_response = await self.reranker.rerank(
                query=query,
                documents=candidates,
                top_k=top_k
            )
            
            reranked_docs = [result.metadata for result in rerank_response.results]
            
            logger.info(
                f"Reranked {rerank_response.total_reranked}/{rerank_response.total_candidates} docs "
                f"(top score: {rerank_response.top_result.score:.3f})"
            )
            
            return reranked_docs
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}, using original order")
            return documents[:top_k]
    
    def _create_event(
        self,
        event_type: EventType,
        content: Any = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create SSE event dict"""
        event = {"type": event_type.value}
        
        if content is not None:
            event["content"] = content
        
        if data is not None:
            event["data"] = data
        
        return event
    
    async def invoke_tool(
        self,
        query: str,
        top_k: int = 5,
        language: str = "ko"
    ) -> Dict[str, Any]:
        """
        Non-streaming tool mode for Slow Lane agent.
        
        Runs retrieval + generation without streaming events.
        Skips safety check (Slow Lane has its own validation).
        
        Args:
            query: Sub-query to answer
            top_k: Number of results to retrieve
            language: Query language
            
        Returns:
            {
                "answer": str,
                "context_chunks": List[Dict],
                "citations_used": List[int],
                "success": bool,
                "error": Optional[str]
            }
        """
        try:
            # Step 1: Expand query (if enabled)
            logger.info(" TOOL: Step 1 - Expanding query")
            queries = await self._expand_query(query, language)
            
            # Step 2: Retrieve documents
            logger.info(" TOOL: Step 2 - Retrieving documents")
            retrieval_results = await self._retrieve_all_queries(queries, top_k)
            
            if not retrieval_results:
                return {
                    "answer": "관련 정보를 찾을 수 없습니다." if language == "ko" else "No relevant information found.",
                    "context_chunks": [],
                    "citations_used": [],
                    "success": False,
                    "error": "No documents retrieved"
                }
            
            # Step 3: Rerank (if enabled)
            final_results = await self._rerank_documents(
                query=query,
                documents=retrieval_results,
                top_k=top_k,
                language=language
            )
            
            # Step 4: Generate answer (non-streaming, NO safety check)
            answer = await self._generate_answer_sync(
                query=query,
                context_chunks=final_results,
                language=language,
                skip_safety_check=True  # Explicitly skip safety check
            )
            
            if not answer or len(answer.strip()) < 10:
                logger.warning("Tool mode: Generated answer is empty or too short")
                return {
                    "answer": answer or "",
                    "context_chunks": final_results,
                    "citations_used": [],
                    "success": False,
                    "error": "Generated answer is empty or too short"
                }
            
            # Step 5: Extract citations used (deduplicated)
            import re
            citation_pattern = r'\[(\d+)\]'
            citations_used = sorted(list(set(int(c) for c in re.findall(citation_pattern, answer))))  # Deduplicate
            
            return {
                "answer": answer,
                "context_chunks": final_results,
                "citations_used": citations_used,
                "success": True,
                "error": None
            }
        
        except Exception as e:
            logger.error(f"Tool invocation error: {e}", exc_info=True)
            return {
                "answer": "",
                "context_chunks": [],
                "citations_used": [],
                "success": False,
                "error": str(e)
            }
    
    async def _generate_answer_sync(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        language: str,
        skip_safety_check: bool = False
    ) -> str:
        """
        Generate answer synchronously (non-streaming) for tool mode.
        
        Args:
            query: User query
            context_chunks: Retrieved context
            language: Response language
            skip_safety_check: If True, skip safety validation
            
        Returns:
            Complete answer string
        """
        
        # Create a temporary streaming generator WITHOUT safety checker if requested
        if skip_safety_check:
            from src.generation import create_streaming_generator
            temp_generator = create_streaming_generator(
                llm_client=self.llm_client,
                safety_checker=None  # No safety check for tool mode
            )
        else:
            temp_generator = self.streaming_generator
        
        # Collect all chunks
        full_answer = ""
        
        async for event in temp_generator.generate_with_citations(
            query=query,
            context_chunks=context_chunks,
            language=language,
            temperature=None,
            max_tokens=self.config.llm.max_tokens  # Use config value, not hardcoded
        ):
            event_type = event.get("type")
            
            # Citations like [1], [2] are already in chunk content
            # StreamingGenerator doesn't emit separate citation_marker events
            if event_type == "chunk":
                full_answer += event.get("content", "")
            
            # Stop on error
            elif event_type == "error":
                logger.error(f"Generation error: {event.get('data', {}).get('message')}")
                break
        
        logger.info(f"Tool mode generated: {len(full_answer)} chars")
        return full_answer
    
    async def close(self):
        """Cleanup resources"""
        logger.info("Closing Fast Lane...")
        
        if self.retriever:
            await self.retriever.close()
        
        if self.llm_client:
            await self.llm_client.close()
        
        self.initialized = False
        logger.info(" Fast Lane closed")


# Convenience function
async def create_fast_lane(config: SystemConfig) -> FastLane:
    """
    Create and initialize Fast Lane.
    
    Args:
        config: SystemConfig
        
    Returns:
        Initialized FastLane instance
    """
    fast_lane = FastLane(config)
    await fast_lane.initialize()
    return fast_lane