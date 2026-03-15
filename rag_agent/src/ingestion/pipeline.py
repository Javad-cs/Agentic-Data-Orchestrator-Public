import asyncio
import asyncpg
from pymilvus import Collection
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from src.config.models import SystemConfig
from src.ingestion.parsers.upstage import UpstageParser
from src.ingestion.parsers.upstage_async import UpstageAsyncParser
from src.ingestion.chunkers.text_chunker import TextChunker, TextChunk
from src.ingestion.chunkers.table_chunker import TableChunker, TableChunk
from src.ingestion.embedders.upstage import UpstageEmbedder
from src.ingestion.indexers.bm25_indexer import BM25Indexer
from src.ingestion.indexers.base import IngestionDocument

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result from ingestion pipeline"""
    job_id: str
    source_file: str
    status: str  # "completed" | "failed" | "partial"
    
    # Statistics
    total_parents: int = 0
    total_children: int = 0
    total_embedded: int = 0
    total_indexed: int = 0
    
    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    
    def mark_completed(self):
        """Mark job as completed and calculate duration"""
        self.completed_at = datetime.now()
        self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        
        if not self.errors:
            self.status = "completed"
        elif self.total_children > 0:
            self.status = "partial"
        else:
            self.status = "failed"


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.
    
    Flow:
    1. Parse PDF with Upstage → Elements
    2. Chunk elements → Parents + Children
    3. Embed children with Upstage → Dense vectors
    4. Index children with BM25 → Term frequencies
    5. Store in PostgreSQL + Milvus
    
    Usage:
        config = SystemConfig()
        pipeline = IngestionPipeline(config)
        await pipeline.initialize()
        
        result = await pipeline.ingest_document("document.pdf")
        
        await pipeline.close()
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize ingestion pipeline.
        
        Args:
            config: SystemConfig with all component configurations
        """
        self.config = config
        
        # Components (initialized in initialize())
        self.parser: Optional[UpstageParser] = None
        self.text_chunker: Optional[TextChunker] = None
        self.table_chunker: Optional[TableChunker] = None
        self.embedder: Optional[UpstageEmbedder] = None
        self.bm25_indexer: Optional[BM25Indexer] = None
        
        # Database connections
        self.db_pool: Optional[asyncpg.Pool] = None
        self.milvus_collection: Optional[Collection] = None
        
        # State
        self.initialized = False
    
    async def initialize(self):
        """
        Initialize all pipeline components and database connections.
        
        Must be called before ingesting documents.
        """
        if self.initialized:
            logger.warning("Pipeline already initialized")
            return
        
        logger.info("Initializing ingestion pipeline...")
        
        # 1. Initialize parser (modified)
        # Parser will be selected per-file based on size
        self.parser = None
        logger.debug(" Parser selection deferred to ingestion time")
        
        # 2. Initialize chunkers
        self.text_chunker = TextChunker(config=self.config.ingestion.chunking)
        self.table_chunker = TableChunker(config=self.config.ingestion.chunking)
        logger.debug(" Chunkers initialized")
        
        # 3. Initialize embedder
        self.embedder = UpstageEmbedder(
            config=self.config.upstage,
            mode="passage"  # we're embedding documents
        )
        logger.debug(" Embedder initialized (passage mode)")
        
        # 4. Initialize database connections
        await self._initialize_databases()
        logger.debug(" Databases connected")
        
        # 5. Initialize BM25 indexer
        self.bm25_indexer = BM25Indexer(db_pool=self.db_pool)
        logger.debug(" BM25 indexer initialized")
        
        self.initialized = True
        logger.info(" Ingestion pipeline ready")
    
    async def _initialize_databases(self):
        """Initialize PostgreSQL and Milvus connections"""
        # PostgreSQL
        self.db_pool = await asyncpg.create_pool(
            self.config.database.postgres_dsn,
            min_size=self.config.database.postgres_pool_min_size,
            max_size=self.config.database.postgres_pool_max_size
        )
        logger.debug(f"PostgreSQL pool created: {self.config.database.postgres_dsn}")
        
        # Milvus
        from pymilvus import connections
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
    
    async def ingest_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionResult:
        """
        Ingest a single document through the full pipeline.
        
        Args:
            file_path: Path to document file
            metadata: Optional metadata (ACLs, tags, etc.)
            
        Returns:
            IngestionResult with statistics and status
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        # Create job
        job_id = str(uuid.uuid4())
        result = IngestionResult(
            job_id=job_id,
            source_file=file_path,
            status="processing"
        )
        
        logger.info(f" Starting ingestion: {file_path} (job_id={job_id})")
        
        try:
            # Log job start to database
            await self._log_job_start(job_id, file_path)
            
            # NEW: Select parser based on file size
            file_size = Path(file_path).stat().st_size
            size_mb = file_size / (1024 * 1024)
            threshold_mb = 20  # Use async for files >20MB
            
            if size_mb > threshold_mb:
                logger.info(f" Large file ({size_mb:.1f}MB) - using async parser")
                self.parser = UpstageAsyncParser(
                    api_key=self.config.upstage.api_key,
                    async_endpoint=self.config.upstage.async_endpoint,
                    model=self.config.upstage.parse_model,
                    poll_interval=self.config.upstage.async_poll_interval,
                    max_wait_time=self.config.upstage.async_max_wait_time
                )
            else:
                logger.info(f" Standard file ({size_mb:.1f}MB) - using sync parser")
                self.parser = UpstageParser(
                    api_key=self.config.upstage.api_key,
                    timeout=self.config.upstage.upstage_timeout,
                    endpoint=self.config.upstage.parse_endpoint,
                    model=self.config.upstage.parse_model
                )
            
            # Step 1: Parse document
            logger.info("Step 1/5: Parsing document...")
            parsed_doc = self.parser.parse(file_path)
            logger.info(f" Parsed {len(parsed_doc.elements)} elements")
            
            # Step 2: Chunk elements
            logger.info("Step 2/5: Chunking elements...")
            all_parents, all_children = await self._chunk_elements(parsed_doc, file_path)
            result.total_parents = len(all_parents)
            result.total_children = len(all_children)
            logger.info(f" Created {len(all_parents)} parents, {len(all_children)} children")
            
            if not all_children:
                raise ValueError("No children created (document may be empty or all stopwords)")
            
            # Step 3: Store parents in PostgreSQL
            logger.info("Step 3/5: Storing parents...")
            await self._store_parents(all_parents, file_path, metadata)
            logger.info(f" Stored {len(all_parents)} parents")
            
            # Step 4: Embed children + Store in Milvus
            logger.info("Step 4/5: Embedding and storing children...")
            await self._embed_and_store_children(all_children)
            result.total_embedded = len(all_children)
            logger.info(f" Embedded and stored {len(all_children)} children")
            
            # Step 5: Index with BM25
            logger.info("Step 5/5: Building BM25 index...")
            await self._index_with_bm25(all_children)
            result.total_indexed = len(all_children)
            logger.info(f" Indexed {len(all_children)} children with BM25")
            
            # Mark completed
            result.mark_completed()
            
            # Log job completion
            await self._log_job_complete(job_id, result)
            
            logger.info(
                f" Ingestion complete: {file_path} "
                f"({result.total_parents} parents, {result.total_children} children, "
                f"{result.duration_seconds:.2f}s)"
            )
            
            return result
        
        except Exception as e:
            logger.error(f" Ingestion failed: {e}", exc_info=True)
            result.errors.append(str(e))
            result.mark_completed()
            
            # Log job failure
            await self._log_job_failed(job_id, str(e))
            
            return result
    
    async def _chunk_elements(
        self,
        parsed_doc,
        source_file: str
    ) -> tuple[List[Any], List[Any]]:
        """
        Chunk all elements into parents and children.
        
        Returns:
            (all_parents, all_children) tuple
        """
        all_parents = []
        all_children = []
        
        for elem in parsed_doc.elements:
            element_id = f"{Path(source_file).stem}_elem_{elem.metadata.get('id', 0)}"
            
            if elem.element_type == "table":
                # Use table chunker
                parents, children = self.table_chunker.chunk_table(
                    table_markdown=elem.content,
                    source_id=element_id
                )
            else:
                # Use text chunker (for text, headings, lists)
                parents, children = self.text_chunker.chunk_text(
                    text=elem.content,
                    source_id=element_id
                )
            
            all_parents.extend(parents)
            all_children.extend(children)
        
        return all_parents, all_children
    
    async def _store_parents(
        self,
        parents: List[Any],
        source_file: str,
        metadata: Optional[Dict[str, Any]]
    ):
        """Store parent chunks in PostgreSQL"""
        async with self.db_pool.acquire() as conn:
            for parent in parents:
                # Determine parent type
                if hasattr(parent, 'has_header') and parent.has_header:
                    parent_type = 'table'
                else:
                    parent_type = 'text'
                
                # Extract metadata
                acl_users = metadata.get('acl_users', []) if metadata else []
                acl_groups = metadata.get('acl_groups', []) if metadata else []
                
                await conn.execute(
                    """
                    INSERT INTO parents (
                        parent_id, parent_text, parent_type, token_count,
                        table_header, is_table_split, split_part, total_parts,
                        source_file, file_name,
                        start_row_idx, end_row_idx,
                        acl_users, acl_groups
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (parent_id) DO UPDATE SET
                        parent_text = EXCLUDED.parent_text,
                        updated_at = NOW()
                    """,
                    parent.chunk_id,
                    parent.text,
                    parent_type,
                    parent.token_count,
                    getattr(parent, 'header', None),
                    getattr(parent, 'is_split', False),
                    getattr(parent, 'split_part', 1),
                    getattr(parent, 'total_parts', 1),
                    source_file,
                    Path(source_file).name,
                    getattr(parent, 'start_row_idx', None),
                    getattr(parent, 'end_row_idx', None),
                    acl_users,
                    acl_groups
                )
    
    async def _embed_and_store_children(self, children: List[Any]):
        """Embed children and store in both PostgreSQL and Milvus"""
        # First, store children in PostgreSQL
        async with self.db_pool.acquire() as conn:
            for child in children:
                # Determine child type
                if hasattr(child, 'row_indices') and child.row_indices:
                    child_type = 'table_row_group'
                else:
                    child_type = 'text_chunk'
                
                await conn.execute(
                    """
                    INSERT INTO children (
                        child_id, parent_id, child_text, child_type,
                        token_count, chunk_index, row_indices
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (child_id) DO UPDATE SET
                        child_text = EXCLUDED.child_text
                    """,
                    child.chunk_id,
                    child.parent_id,
                    child.text,
                    child_type,
                    child.token_count,
                    getattr(child, 'chunk_index', 0),
                    getattr(child, 'row_indices', None)
                )
        
        # Embed in batches
        batch_size = self.config.upstage.embedding_batch_size
        for i in range(0, len(children), batch_size):
            batch = children[i:i + batch_size]
            
            # Embed batch
            texts = [child.text for child in batch]
            embed_result = self.embedder.embed_batch(texts)
            
            # Prepare data for Milvus
            milvus_data = []
            for child, embedding_result in zip(batch, embed_result.results):
                milvus_data.append({
                    'child_id': child.chunk_id,
                    'dense_vector': embedding_result.embedding.tolist(),
                    'parent_id': child.parent_id,
                    'source_file': '',  # Will be populated from parent lookup if needed
                    'page_number': 0,
                    'chunk_type': 'table_row_group' if hasattr(child, 'row_indices') else 'text_chunk'
                })
            
            # Insert into Milvus
            self.milvus_collection.insert(milvus_data)
        
        # Flush to ensure data is persisted
        self.milvus_collection.flush()
    
    async def _index_with_bm25(self, children: List[Any]):
        """Index children with BM25"""
        # Prepare documents for BM25 indexer
        documents = [
            IngestionDocument(
                child_id=child.chunk_id,
                parent_id=child.parent_id,
                text=child.text
            )
            for child in children
        ]
        
        # Batch index with BEST-EFFORT (not atomic)
        result = await self.bm25_indexer.index_batch(
            documents=documents,
            transaction=False  # Changed: Continue on failures
        )
        
        if result.total_failed > 0:
            logger.warning(
                f"BM25 indexing: {result.total_success}/{len(documents)} succeeded, "
                f"{result.total_failed} failed (stopwords-only or empty)"
            )
            
            # Log failed IDs (but don't crash the whole ingestion)
            for failed_result in result.results:
                if not failed_result.success:
                    logger.debug(f"Skipped BM25 indexing for {failed_result.document_id}: {failed_result.error}")
        else:
            logger.info(f" All {result.total_success} documents indexed with BM25")
    
    async def _log_job_start(self, job_id: str, source_file: str):
        """Log job start to ingestion_log table"""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingestion_log (job_id, source_file, status)
                VALUES ($1, $2, 'processing')
                """,
                uuid.UUID(job_id),
                source_file
            )
    
    async def _log_job_complete(self, job_id: str, result: IngestionResult):
        """Log job completion to ingestion_log table"""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_log
                SET status = 'completed',
                    total_parents = $2,
                    total_children = $3,
                    completed_at = $4
                WHERE job_id = $1
                """,
                uuid.UUID(job_id),
                result.total_parents,
                result.total_children,
                result.completed_at
            )
    
    async def _log_job_failed(self, job_id: str, error_message: str):
        """Log job failure to ingestion_log table"""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_log
                SET status = 'failed',
                    error_message = $2,
                    completed_at = NOW()
                WHERE job_id = $1
                """,
                uuid.UUID(job_id),
                error_message
            )
    
    async def close(self):
        """Close all connections and cleanup resources"""
        logger.info("Closing ingestion pipeline...")
        
        if self.db_pool:
            await self.db_pool.close()
            logger.debug(" PostgreSQL pool closed")
        
        if self.milvus_collection:
            from pymilvus import connections
            connections.disconnect("default")
            logger.debug(" Milvus connection closed")
        
        self.initialized = False
        logger.info(" Pipeline closed")