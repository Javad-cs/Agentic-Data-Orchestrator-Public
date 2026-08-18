import asyncpg
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from .base import BaseIndexer, IndexResult, BatchIndexResult, IngestionDocument, IndexerType
from .bm25_tokenizer import BM25Tokenizer

logger = logging.getLogger(__name__)


class BM25Indexer(BaseIndexer):
    """
    BM25 indexer that stores term frequencies in PostgreSQL.
    
    Optimizations:
    - Bulk insert with executemany() for atomic batches
    - Pre-tokenization for best-effort batches
    - Structured logging instead of print()
    - Proper error handling
    """
    
    def __init__(
        self,
        db_pool: asyncpg.Pool,
        tokenizer: Optional[BM25Tokenizer] = None
    ):
        """
        Initialize BM25 indexer.
        
        Args:
            db_pool: PostgreSQL connection pool
            tokenizer: BM25Tokenizer (creates default if None)
        """
        self.db_pool = db_pool
        self.tokenizer = tokenizer or BM25Tokenizer()
    
    def get_indexer_type(self) -> IndexerType:
        """Return indexer type"""
        return IndexerType.SPARSE
    
    async def index(self, document: IngestionDocument) -> IndexResult:
        """
        Index a single document.
        
        Args:
            document: IngestionDocument with text content
            
        Returns:
            IndexResult
        """
        try:
            # Tokenize and get term frequencies
            term_frequencies = self.tokenizer.tokenize_with_frequencies(document.text)
            
            if not term_frequencies:
                logger.warning(
                    f"Document {document.child_id}: No tokens extracted (stopwords only or empty)"
                )
                return IndexResult(
                    document_id=document.child_id,
                    success=False,
                    indexer_type=self.get_indexer_type(),
                    error="No tokens extracted (empty content or all stopwords)"
                )
            
            # Calculate document length
            doc_length = sum(term_frequencies.values())
            
            # Insert into database
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO bm25_index (child_id, term_frequencies, doc_length, parent_id)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (child_id)
                    DO UPDATE SET
                        term_frequencies = EXCLUDED.term_frequencies,
                        doc_length = EXCLUDED.doc_length,
                        updated_at = NOW()
                    """,
                    document.child_id,
                    json.dumps(term_frequencies),
                    doc_length,
                    document.parent_id
                )
            
            logger.debug(
                f"Indexed {document.child_id}: {len(term_frequencies)} unique terms, "
                f"{doc_length} total tokens"
            )
            
            return IndexResult(
                document_id=document.child_id,
                success=True,
                indexer_type=self.get_indexer_type(),
                metadata={
                    'unique_terms': len(term_frequencies),
                    'doc_length': doc_length
                }
            )
        
        except Exception as e:
            logger.error(f"Failed to index {document.child_id}: {e}")
            return IndexResult(
                document_id=document.child_id,
                success=False,
                indexer_type=self.get_indexer_type(),
                error=str(e)
            )
    
    async def index_batch(
        self,
        documents: List[IngestionDocument],
        transaction: bool = True
    ) -> BatchIndexResult:
        """
        Index multiple documents in batch.
        
        Args:
            documents: List of IngestionDocument objects
            transaction: If True, rollback all on any failure
        
        Returns:
            BatchIndexResult
        """
        if transaction:
            return await self._index_batch_atomic(documents)
        else:
            return await self._index_batch_best_effort(documents)
    
    async def _index_batch_atomic(
        self,
        documents: List[IngestionDocument]
    ) -> BatchIndexResult:
        """
        Index batch atomically with bulk insert (OPTIMIZED).
        
        Uses executemany() for better performance.
        """
        results = []
        
        try:
            # Pre-tokenize all documents
            batch_data = []
            for doc in documents:
                term_frequencies = self.tokenizer.tokenize_with_frequencies(doc.text)
                
                if not term_frequencies:
                    raise ValueError(f"Document {doc.child_id}: No tokens extracted")
                
                doc_length = sum(term_frequencies.values())
                
                batch_data.append((
                    doc.child_id,
                    json.dumps(term_frequencies),
                    doc_length,
                    doc.parent_id
                ))
                
                results.append(IndexResult(
                    document_id=doc.child_id,
                    success=True,
                    indexer_type=self.get_indexer_type(),
                    metadata={
                        'unique_terms': len(term_frequencies),
                        'doc_length': doc_length
                    }
                ))
            
            # Bulk insert using executemany (MUCH faster)
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(
                        """
                        INSERT INTO bm25_index (child_id, term_frequencies, doc_length, parent_id)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (child_id)
                        DO UPDATE SET
                            term_frequencies = EXCLUDED.term_frequencies,
                            doc_length = EXCLUDED.doc_length,
                            updated_at = NOW()
                        """,
                        batch_data
                    )
            
            logger.info(f"Bulk indexed {len(batch_data)} documents atomically")
            
            return BatchIndexResult(
                results=results,
                total_success=len(results),
                total_failed=0,
                indexer_type=self.get_indexer_type()
            )
        
        except Exception as e:
            logger.error(f"Atomic batch indexing failed: {e}")
            
            # Transaction rolled back - mark all as failed
            failed_results = [
                IndexResult(
                    document_id=doc.child_id,
                    success=False,
                    indexer_type=self.get_indexer_type(),
                    error=f"Batch transaction failed: {str(e)}"
                )
                for doc in documents
            ]
            
            return BatchIndexResult(
                results=failed_results,
                total_success=0,
                total_failed=len(documents),
                indexer_type=self.get_indexer_type()
            )
    
    async def _index_batch_best_effort(
        self,
        documents: List[IngestionDocument]
    ) -> BatchIndexResult:
        """
        Index batch with best effort (OPTIMIZED).
        
        Pre-tokenizes all documents, then bulk inserts successful ones.
        """
        results = []
        batch_data = []
        
        # Pre-tokenize all documents
        for doc in documents:
            try:
                term_frequencies = self.tokenizer.tokenize_with_frequencies(doc.text)
                
                if not term_frequencies:
                    logger.warning(f"Document {doc.child_id}: No tokens extracted")
                    results.append(IndexResult(
                        document_id=doc.child_id,
                        success=False,
                        indexer_type=self.get_indexer_type(),
                        error="No tokens extracted"
                    ))
                    continue
                
                doc_length = sum(term_frequencies.values())
                
                batch_data.append((
                    doc.child_id,
                    json.dumps(term_frequencies),
                    doc_length,
                    doc.parent_id
                ))
                
                results.append(IndexResult(
                    document_id=doc.child_id,
                    success=True,
                    indexer_type=self.get_indexer_type(),
                    metadata={
                        'unique_terms': len(term_frequencies),
                        'doc_length': doc_length
                    }
                ))
            
            except Exception as e:
                logger.error(f"Failed to tokenize {doc.child_id}: {e}")
                results.append(IndexResult(
                    document_id=doc.child_id,
                    success=False,
                    indexer_type=self.get_indexer_type(),
                    error=str(e)
                ))
        
        # Bulk insert successful documents
        if batch_data:
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.executemany(
                        """
                        INSERT INTO bm25_index (child_id, term_frequencies, doc_length, parent_id)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (child_id)
                        DO UPDATE SET
                            term_frequencies = EXCLUDED.term_frequencies,
                            doc_length = EXCLUDED.doc_length,
                            updated_at = NOW()
                        """,
                        batch_data
                    )
                
                logger.info(f"Best-effort indexed {len(batch_data)}/{len(documents)} documents")
            
            except Exception as e:
                logger.error(f"Bulk insert failed: {e}")
                # Mark all successful tokenizations as failed
                for result in results:
                    if result.success:
                        result.success = False
                        result.error = f"Database insert failed: {str(e)}"
        
        total_success = sum(1 for r in results if r.success)
        total_failed = len(results) - total_success
        
        return BatchIndexResult(
            results=results,
            total_success=total_success,
            total_failed=total_failed,
            indexer_type=self.get_indexer_type()
        )
    
    async def delete(self, document_id: str) -> bool:
        """
        Delete a document from BM25 index.
        
        This automatically triggers DF decrements via PostgreSQL trigger.
        
        Args:
            document_id: Child ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM bm25_index WHERE child_id = $1",
                    document_id
                )
                
                # Extract deleted count from "DELETE N"
                deleted_count = int(result.split()[-1])
                
                if deleted_count > 0:
                    logger.debug(f"Deleted {document_id} from BM25 index")
                    return True
                else:
                    logger.debug(f"Document {document_id} not found in BM25 index")
                    return False
        
        except Exception as e:
            logger.error(f"Error deleting from BM25 index: {e}")
            return False
    
    async def delete_batch(self, document_ids: List[str]) -> int:
        """
        Delete multiple documents.
        
        Args:
            document_ids: List of child IDs to delete
            
        Returns:
            Number of documents actually deleted
        """
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM bm25_index WHERE child_id = ANY($1)",
                    document_ids
                )
                
                # Extract deleted count
                deleted_count = int(result.split()[-1])
                logger.info(f"Batch deleted {deleted_count}/{len(document_ids)} documents")
                return deleted_count
        
        except Exception as e:
            logger.error(f"Error batch deleting from BM25 index: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get BM25 index statistics"""
        try:
            async with self.db_pool.acquire() as conn:
                stats = await conn.fetch(
                    "SELECT stat_key, stat_value FROM bm25_stats"
                )
                
                result = {row['stat_key']: float(row['stat_value']) for row in stats}
                
                unique_terms = await conn.fetchval(
                    "SELECT COUNT(*) FROM bm25_df"
                )
                result['unique_terms'] = unique_terms
                
                return result
        
        except Exception as e:
            logger.error(f"Error getting BM25 stats: {e}")
            return {}
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        k1: float = 1.5,
        b: float = 0.75
    ) -> List[Dict[str, Any]]:
        """
        Search using BM25 scoring.
        
        Pre-filters using GIN index on term_frequencies for performance.
        Only calculates BM25 scores for documents containing query terms.
        
        Args:
            query: Search query (will be tokenized using same normalizations as indexing)
            top_k: Number of results to return
            k1: BM25 k1 parameter
            b: BM25 b parameter
            
        Returns:
            List of results with child_id and score
        """
        # Tokenize query with SAME normalization as indexing
        query_terms = self.tokenizer.tokenize(query)
        
        if not query_terms:
            logger.debug(f"Query '{query}' produced no tokens after normalization")
            return []
        
        try:
            async with self.db_pool.acquire() as conn:
                results = await conn.fetch(
                    """
                    SELECT 
                        child_id,
                        calculate_bm25_score(child_id, $1, $2, $3) as score
                    FROM bm25_index
                    WHERE term_frequencies ?| $1  -- Pre-filter using GIN index (fast!)
                    ORDER BY score DESC
                    LIMIT $4
                    """,
                    query_terms,
                    k1,
                    b,
                    top_k
                )
                
                logger.debug(
                    f"BM25 search for {len(query_terms)} terms returned {len(results)} results"
                )
                
                return [
                    {
                        'child_id': row['child_id'],
                        'score': float(row['score'])
                    }
                    for row in results
                ]
        
        except Exception as e:
            logger.error(f"Error searching BM25 index: {e}")
            return []