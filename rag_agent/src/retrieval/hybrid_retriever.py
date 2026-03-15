# src/retrieval/hybrid_retriever.py

import asyncpg
from pymilvus import Collection, connections
from typing import List, Dict, Any, Optional
import logging

from src.config.models import SystemConfig
from src.ingestion.embedders.upstage import UpstageEmbedder
from src.ingestion.indexers.bm25_indexer import BM25Indexer
from src.retrieval.merge.rrf import reciprocal_rank_fusion, RetrievalCandidate
from src.retrieval.rerankers.cohere import CohereReranker
from src.retrieval.rerankers.base import CandidateDocument

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval combining dense (Milvus) and sparse (BM25) search.
    
    Flow:
    1. Encode query with Upstage (query model)
    2. Search Milvus (dense/semantic search)
    3. Search BM25 (sparse/keyword search)
    4. Merge with RRF (Reciprocal Rank Fusion)
    5. Rerank with Upstage (optional, cross-encoder)
    6. Fetch parent context for top results
    
    Usage:
        retriever = HybridRetriever(config)
        await retriever.initialize()
        
        # With reranking (better quality, slower)
        results = await retriever.retrieve(
            query="스테인레스강 가공 방법",
            top_k=10,
            rerank=True
        )
        
        # Without reranking (faster)
        results = await retriever.retrieve(
            query="스테인레스강 가공 방법",
            top_k=10,
            rerank=False
        )
        
        await retriever.close()
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize hybrid retriever.
        
        Args:
            config: SystemConfig with all component configurations
        """
        self.config = config
        
        # Components (initialized in initialize())
        self.query_embedder: Optional[UpstageEmbedder] = None
        self.bm25_indexer: Optional[BM25Indexer] = None
        self.reranker: Optional[CohereReranker] = None
        
        # Database connections
        self.db_pool: Optional[asyncpg.Pool] = None
        self.milvus_collection: Optional[Collection] = None
        
        # State
        self.initialized = False
    
    async def initialize(self):
        """Initialize all retriever components and database connections."""
        if self.initialized:
            logger.warning("Retriever already initialized")
            return
        
        logger.info("Initializing hybrid retriever...")
        
        # 1. Initialize query embedder (QUERY mode!)
        self.query_embedder = UpstageEmbedder(
            config=self.config.upstage,
            mode="query"
        )
        logger.debug(" Query embedder initialized")
        
        # 2. Initialize database connections
        await self._initialize_databases()
        logger.debug(" Databases connected")
        
        # 3. Initialize BM25 indexer (for searching)
        self.bm25_indexer = BM25Indexer(db_pool=self.db_pool)
        logger.debug(" BM25 indexer initialized")
        
        # 4. Initialize reranker (ONLY if enabled in config)
        if self.config.upstage.reranking_enabled:
            self.reranker = CohereReranker(config=self.config.cohere)
            logger.debug(f" Reranker initialized (model={self.reranker.model})")
        else:
            self.reranker = None
            logger.debug(" Reranker disabled (skipping initialization)")
        
        self.initialized = True
        logger.info(" Hybrid retriever ready")
    
    async def _initialize_databases(self):
        """Initialize PostgreSQL and Milvus connections"""
        # PostgreSQL
        self.db_pool = await asyncpg.create_pool(
            self.config.database.postgres_dsn,
            min_size=self.config.database.postgres_pool_min_size,
            max_size=self.config.database.postgres_pool_max_size
        )
        logger.debug(f"PostgreSQL pool created")
        
        # Milvus
        host, port = self._parse_milvus_uri(self.config.database.milvus_uri)
        connections.connect(host=host, port=port)
        
        self.milvus_collection = Collection(self.config.database.milvus_collection_name)
        self.milvus_collection.load()
        logger.debug(f"Milvus collection loaded: {self.config.database.milvus_collection_name}")
    
    def _parse_milvus_uri(self, uri: str) -> tuple[str, str]:
        """Parse Milvus URI into host and port"""
        from urllib.parse import urlparse
        
        if not uri.startswith(('http://', 'https://')):
            uri = f'http://{uri}'
        
        parsed = urlparse(uri)
        host = parsed.hostname or 'localhost'
        port = str(parsed.port) if parsed.port else '19530'
        
        return host, port
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        rrf_k: int = 60,
        weights: Optional[Dict[str, float]] = None,
        rerank: bool = True,
        rerank_top_n: int = 20,
        include_parent: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            dense_top_k: Number of results from dense search
            sparse_top_k: Number of results from sparse search
            rrf_k: RRF constant (default 60)
            weights: Retriever weights {'dense': 0.7, 'sparse': 0.3}
            rerank: Whether to rerank results with cross-encoder
            rerank_top_n: Number of top RRF candidates to rerank
            include_parent: Whether to fetch parent context
        
        Returns:
            List of results with child text, parent context, and scores
        """
        if not self.initialized:
            raise RuntimeError("Retriever not initialized. Call initialize() first.")
        
        logger.info(f" Retrieving for query: {query[:50]}...")
        
        # 1. Dense retrieval (Milvus)
        logger.debug("Step 1/5: Dense retrieval (Milvus)...")
        dense_results = await self._dense_retrieve(query, dense_top_k)
        logger.debug(f" Dense: {len(dense_results)} results")
        
        # 2. Sparse retrieval (BM25)
        logger.debug("Step 2/5: Sparse retrieval (BM25)...")
        sparse_results = await self._sparse_retrieve(query, sparse_top_k)
        logger.debug(f" Sparse: {len(sparse_results)} results")
        
        # 3. Merge with RRF
        logger.debug("Step 3/5: Merging with RRF...")
        merged = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=rrf_k,
            weights=weights
        )
        logger.debug(f" Merged: {len(merged)} unique candidates")
        
        # 4. Rerank (if enabled and requested)
        if rerank and self.reranker is not None:
            logger.debug(f"Step 4/5: Reranking top {min(rerank_top_n, len(merged))} candidates...")
            merged = await self._rerank_candidates(query, merged, rerank_top_n)
            logger.debug(f"   Reranked")
        else:
            # Clear logging about why reranking was skipped
            if rerank and self.reranker is None:
                logger.debug("Step 4/5: Reranking skipped (reranker not initialized)")
            elif not rerank:
                logger.debug("Step 4/5: Reranking skipped (rerank=False)")
        
        # 5. Get top K
        top_candidates = merged[:top_k]
        
        # 6. Fetch parent context if requested
        if include_parent:
            logger.debug("Step 5/5: Fetching parent context...")
            results = await self._enrich_with_parents(top_candidates)
        else:
            results = [self._candidate_to_dict(c) for c in top_candidates]
        
        logger.info(f" Retrieved {len(results)} results")
        
        return results
    
    async def _dense_retrieve(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Dense retrieval using Milvus vector search"""
        # Embed query
        query_result = self.query_embedder.embed(query)
        query_vector = query_result.embedding.tolist()
        
        # Search Milvus
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 100}
        }
        
        results = self.milvus_collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            output_fields=["child_id", "parent_id", "chunk_type"]
        )
        
        # Fetch text from PostgreSQL
        child_ids = [hit.id for hit in results[0]]
        
        if not child_ids:
            return []
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT child_id, child_text FROM children WHERE child_id = ANY($1)",
                child_ids
            )
            
            text_lookup = {row['child_id']: row['child_text'] for row in rows}
        
        # Format results (filter out empty texts)
        formatted_results = []
        empty_count = 0
        
        for hit in results[0]:
            text = text_lookup.get(hit.id, '').strip()
            
            # Skip documents with empty text
            if not text:
                empty_count += 1
                logger.debug(f"Skipping dense result {hit.id}: empty text")
                continue
            
            formatted_results.append({
                'child_id': hit.id,
                'score': float(hit.distance),
                'text': text,
                'source': 'dense'
            })
        
        if empty_count > 0:
            logger.warning(
                f"Dense retrieval: Skipped {empty_count}/{len(child_ids)} results with empty text"
            )
        
        return formatted_results
    
    async def _sparse_retrieve(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Sparse retrieval using BM25"""
        results = await self.bm25_indexer.search(
            query=query,
            top_k=top_k
        )
        
        if not results:
            return []
        
        child_ids = [r['child_id'] for r in results]
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT child_id, child_text FROM children WHERE child_id = ANY($1)",
                child_ids
            )
            
            text_lookup = {row['child_id']: row['child_text'] for row in rows}
        
        # Format results (filter out empty texts)
        formatted_results = []
        empty_count = 0
        
        for r in results:
            text = text_lookup.get(r['child_id'], '').strip()
            
            # Skip documents with empty text
            if not text:
                empty_count += 1
                logger.debug(f"Skipping sparse result {r['child_id']}: empty text")
                continue
            
            formatted_results.append({
                'child_id': r['child_id'],
                'score': r['score'],
                'text': text,
                'source': 'sparse'
            })
        
        if empty_count > 0:
            logger.warning(
                f"Sparse retrieval: Skipped {empty_count}/{len(child_ids)} results with empty text"
            )
        
        return formatted_results
    
    async def _rerank_candidates(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_n: int
    ) -> List[RetrievalCandidate]:
        """
        Rerank top candidates using cross-encoder.
        
        Args:
            query: Search query
            candidates: List of candidates from RRF
            top_n: Number of top candidates to rerank
        
        Returns:
            Reranked candidates (top_n reranked + rest unchanged)
        """
        if not candidates:
            return candidates
        
        # Take top N for reranking
        candidates_to_rerank = candidates[:top_n]
        rest = candidates[top_n:]
        
        # Filter out candidates with empty text BEFORE reranking
        valid_candidates = []
        empty_candidates = []
        
        for c in candidates_to_rerank:
            if c.text and c.text.strip():
                valid_candidates.append(c)
            else:
                empty_candidates.append(c)
                logger.debug(f"Skipping reranking for {c.document_id}: empty text")
        
        if not valid_candidates:
            logger.warning("No valid candidates for reranking (all have empty text)")
            return candidates
        
        if empty_candidates:
            logger.warning(
                f"Reranking: Skipped {len(empty_candidates)}/{len(candidates_to_rerank)} "
                f"candidates with empty text"
            )
        
        # Convert to CandidateDocument format
        docs = [
            CandidateDocument(
                id=c.document_id,
                text=c.text.strip(),
                score=c.rrf_score,
                metadata=c.metadata
            )
            for c in valid_candidates
        ]
        
        # Rerank (ASYNC with must await)
        try:
            rerank_response = await self.reranker.rerank(  # ← AWAIT HERE
                query=query,
                documents=docs,
                top_k=None
            )
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return candidates
        
        # Update candidates with reranking scores
        reranked_candidates = []
        for rerank_result in rerank_response.results:
            # Find original candidate
            original = next(
                c for c in valid_candidates
                if c.document_id == rerank_result.document_id
            )
            
            # Create new candidate with updated metadata
            reranked = RetrievalCandidate(
                document_id=original.document_id,
                text=original.text,
                dense_score=original.dense_score,
                sparse_score=original.sparse_score,
                rrf_score=original.rrf_score,
                dense_rank=original.dense_rank,
                sparse_rank=original.sparse_rank,
                metadata={
                    **original.metadata,
                    'rerank_score': rerank_result.score,
                    'rerank_rank': rerank_result.reranked_rank,
                    'original_rrf_rank': rerank_result.original_rank
                }
            )
            
            reranked_candidates.append(reranked)
        
        # Combine: reranked + empty candidates + rest
        return reranked_candidates + empty_candidates + rest
    
    async def _enrich_with_parents(
        self,
        candidates: List[RetrievalCandidate]
    ) -> List[Dict[str, Any]]:
        """Fetch parent context for candidates"""
        child_ids = [c.document_id for c in candidates]
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT 
                    c.child_id,
                    c.parent_id,
                    c.child_text,
                    p.parent_text,
                    p.parent_type,
                    p.source_file,
                    p.page_number
                FROM children c
                JOIN parents p ON c.parent_id = p.parent_id
                WHERE c.child_id = ANY($1)
                """,
                child_ids
            )
            
            context_lookup = {row['child_id']: row for row in rows}
        
        # Enrich candidates
        results = []
        for candidate in candidates:
            context = context_lookup.get(candidate.document_id)
            
            result = {
                'child_id': candidate.document_id,
                'child_text': candidate.text,
                'rrf_score': candidate.rrf_score,
                'dense_score': candidate.dense_score,
                'dense_rank': candidate.dense_rank,
                'sparse_score': candidate.sparse_score,
                'sparse_rank': candidate.sparse_rank,
            }
            
            # Add reranking scores if available
            if 'rerank_score' in candidate.metadata:
                result['rerank_score'] = candidate.metadata['rerank_score']
                result['rerank_rank'] = candidate.metadata['rerank_rank']
                result['original_rrf_rank'] = candidate.metadata['original_rrf_rank']
            
            if context:
                result.update({
                    'parent_id': context['parent_id'],
                    'parent_text': context['parent_text'],
                    'parent_type': context['parent_type'],
                    'source_file': context['source_file'],
                    'page_number': context['page_number']
                })
            
            results.append(result)
        
        return results
    
    def _candidate_to_dict(self, candidate: RetrievalCandidate) -> Dict[str, Any]:
        """Convert candidate to dictionary"""
        result = {
            'child_id': candidate.document_id,
            'child_text': candidate.text,
            'rrf_score': candidate.rrf_score,
            'dense_score': candidate.dense_score,
            'dense_rank': candidate.dense_rank,
            'sparse_score': candidate.sparse_score,
            'sparse_rank': candidate.sparse_rank,
        }
        
        # Add reranking scores if available
        if 'rerank_score' in candidate.metadata:
            result['rerank_score'] = candidate.metadata['rerank_score']
            result['rerank_rank'] = candidate.metadata['rerank_rank']
            result['original_rrf_rank'] = candidate.metadata['original_rrf_rank']
        
        return result
    
    async def close(self):
        """Close all connections and cleanup resources"""
        logger.info("Closing hybrid retriever...")
        
        if self.db_pool:
            await self.db_pool.close()
            logger.debug(" PostgreSQL pool closed")
        
        if self.milvus_collection:
            connections.disconnect("default")
            logger.debug(" Milvus connection closed")
        
        # Close reranker's HTTP client
        if self.reranker:
            await self.reranker.close()
            logger.debug(" Reranker closed")
        
        self.initialized = False
        logger.info(" Retriever closed")