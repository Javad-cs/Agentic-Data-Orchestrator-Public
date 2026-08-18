"""Wrapper for Text-to-SQL agent"""
import sys
import asyncio
import logging
from pathlib import Path
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)

# Add text_to_sql_agent to path
_current_file = Path(__file__).resolve()
_project_root = _current_file.parents[3]
_sql_agent_path = str(_project_root / "text_to_sql_agent")

def _import_sql_agent_classes():
    """Import text_to_sql_agent classes safely"""
    for key in list(sys.modules.keys()):
        if key == 'src' or key.startswith('src.'):
            del sys.modules[key]
    
    sys.path.insert(0, _sql_agent_path)
    
    from final_sql_w_cand_voting.orchestrator import VotingOrchestrator
    from final_sql_w_cand_voting.few_shot_store import FewShotStore
    from final_sql_w_cand_voting.candidate_generator import CandidateGenerator
    from sql_generation import Algorithm1Runner
    from schema_linking import FocusedSchemaBuilder, FocusedSchemaConfig, SchemaVariantGenerator
    from indexing import FieldIndex, SchemaLiteralMatcher
    from profiling import ColumnProfiler, ProfileSummarizer, TableProfiler, TableSummarizer
    from profiling.metadata_enricher import MetadataEnricher
    from oracle_adapter import connect_database as oracle_connect
    from core import connect_database, create_llm_client
    from config import settings
    
    return (
        VotingOrchestrator, FewShotStore, CandidateGenerator,
        Algorithm1Runner, FocusedSchemaBuilder, FocusedSchemaConfig,
        SchemaVariantGenerator, FieldIndex, SchemaLiteralMatcher,
        ColumnProfiler, ProfileSummarizer, MetadataEnricher,
        connect_database, create_llm_client, settings,
        TableProfiler, TableSummarizer, oracle_connect 
    )

(
    VotingOrchestrator, FewShotStore, CandidateGenerator,
    Algorithm1Runner, FocusedSchemaBuilder, FocusedSchemaConfig,
    SchemaVariantGenerator, FieldIndex, SchemaLiteralMatcher,
    ColumnProfiler, ProfileSummarizer, MetadataEnricher,
    connect_database, create_llm_client, settings,
    TableProfiler, TableSummarizer, oracle_connect
) = _import_sql_agent_classes()

from .base import BaseTool, ToolResponse



class SQLTool(BaseTool):
    """Database-agnostic Text-to-SQL tool for production databases"""
    
    def __init__(self):
        """
        Initialize SQL tool.
        Reads database config from global settings (supports any SQL database).
        """
        # Load config from global settings
        from router_agent.src.config.settings import get_settings
        settings = get_settings()
        
        self.db_config = {
            "primary_dsn": settings.PRIMARY_DSN,
            "primary_user": settings.PRIMARY_USER,
            "primary_password": settings.PRIMARY_PASSWORD,
            "primary_db_name": settings.PRIMARY_DB_NAME,
            "linked_dsn": settings.LINKED_DSN,
            "linked_user": settings.LINKED_USER,
            "linked_password": settings.LINKED_PASSWORD,
            "linked_db_name": settings.LINKED_DB_NAME,
            "linked_suffix": settings.LINKED_SUFFIX,
            "db_type": settings.DB_TYPE,  # "oracle", "postgres", etc.
        }
        
        # Pipeline components (lazy init)
        self.reasoning_client = None
        self.masking_client = None
        self.algo1_runner = None
        self.fs_builder = None
        self.variant_gen = None
        self.store = None
        self.generator = None
        
        # Table metadata (for Slow Lane)
        self.table_metadata_map = None  # Built during initialization
        
        self._initialized = False
        logger.info(f"SQL tool created (db={self.db_config['primary_db_name']})")
        
    def _get_db_connection(self, is_linked: bool = False):
        """
        Get database connection (database-agnostic).
        
        Args:
            is_linked: If True, connect to linked database
            
        Returns:
            Database connection context manager
        """
        if is_linked and self.db_config.get("linked_dsn"):
            return oracle_connect(
                self.db_config["linked_dsn"],
                self.db_config["linked_user"],
                self.db_config["linked_password"]
            )
        else:
            return oracle_connect(
                self.db_config["primary_dsn"],
                self.db_config["primary_user"],
                self.db_config["primary_password"]
            )
    
    async def _ensure_initialized(self):
        """Lazy initialization."""
        if self._initialized:
            return
        
        logger.info(f"Initializing SQL pipeline...")
        await asyncio.to_thread(self._setup_pipeline)
        self._initialized = True
        logger.info("SQL pipeline initialized")
    
    
    def _setup_pipeline(self):
        """
        Setup pipeline for any database.
        Profiles primary + linked databases, builds indices, initializes runners.
        """
        import sys
        import os
        import pickle
        from pathlib import Path
        
        # Add text_to_sql_agent to path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "text_to_sql_agent"))
        
        from text_to_sql_agent.oracle_adapter import connect_database
        from text_to_sql_agent.core import create_llm_client
        from text_to_sql_agent.profiling import ColumnProfiler, ProfileSummarizer
        from text_to_sql_agent.indexing import FieldIndex, SchemaLiteralMatcher
        from text_to_sql_agent.schema_linking import (
            FocusedSchemaBuilder, 
            FocusedSchemaConfig, 
            SchemaVariantGenerator
        )
        from text_to_sql_agent.sql_generation import Algorithm1Runner
        from text_to_sql_agent.final_sql_w_cand_voting.few_shot_store import FewShotStore
        from text_to_sql_agent.final_sql_w_cand_voting.candidate_generator import CandidateGenerator
        
        logger.info(f"Setting up SQL pipeline for {self.db_config['primary_db_name']}")
        
        # Create LLM clients
        self.reasoning_client = create_llm_client(model="gpt-4.1")
        self.masking_client = create_llm_client(model="gpt-4.1")
        
        # Load or create checkpoint
        checkpoint_file = Path("./data/profile_checkpoint.pkl")
        profiles = []
        profiled_columns = set()
        
        if checkpoint_file.exists():
            try:
                logger.info("Loading profiles from checkpoint...")
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    profiles = checkpoint_data.get('profiles', [])
                    profiled_columns = checkpoint_data.get('profiled_columns', set())
                logger.info(f"Loaded {len(profiles)} cached profiles")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, will re-profile")
                profiles = []
                profiled_columns = set()
        
        def save_checkpoint():
            """Save profiling checkpoint."""
            try:
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({'profiles': profiles, 'profiled_columns': profiled_columns}, f)
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
        
        profiler = ColumnProfiler()
        
        # 1. Profile primary database
        primary_dsn = self.db_config["primary_dsn"]
        primary_user = self.db_config["primary_user"]
        primary_password = self.db_config["primary_password"]
        primary_db_name = self.db_config["primary_db_name"]
        
        logger.info(f"Profiling {primary_db_name} tables...")
        
        with connect_database(primary_dsn, primary_user, primary_password) as db:
            primary_tables = ['EQP_MODE_GEN_HIST', 'CIM_EQP_MST']
            logger.info(f"Found {len(primary_tables)} tables in {primary_db_name}")
            
            for table in primary_tables:
                try:
                    table_info = db.get_table_info(table)
                    for col in table_info.columns:
                        col_key = f"{primary_db_name}.{table}.{col.name}"
                        
                        if col_key in profiled_columns:
                            logger.debug(f"Skipping {col_key} (cached)")
                            continue
                        
                        try:
                            profile = profiler.profile_column(db, table, col.name, col.type)
                            profiles.append(profile)
                            profiled_columns.add(col_key)
                            save_checkpoint()
                            logger.debug(f"Profiled {col_key}")
                        except Exception as e:
                            logger.warning(f"Failed to profile {col_key}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to get info for table {table}: {e}")
        
        logger.info(f"Profiled {len([p for p in profiles if self.db_config.get('linked_suffix', '@') not in p.table_name])} columns from {primary_db_name}")
        
        # 2. Profile linked database (if exists)
        if self.db_config.get("linked_dsn"):
            linked_dsn = self.db_config["linked_dsn"]
            linked_user = self.db_config["linked_user"]
            linked_password = self.db_config["linked_password"]
            linked_db_name = self.db_config["linked_db_name"]
            linked_suffix = self.db_config["linked_suffix"]
            
            logger.info(f"Profiling {linked_db_name} tables...")
            
            try:
                with connect_database(linked_dsn, linked_user, linked_password) as db:
                    linked_tables = ['EQP_MST', 'DEPT_MST', 'WORK_CENTER']
                    logger.info(f"Found {len(linked_tables)} tables in {linked_db_name}")
                    
                    for table in linked_tables:
                        try:
                            table_info = db.get_table_info(table)
                            for col in table_info.columns:
                                col_key = f"{linked_db_name}.{table}.{col.name}"
                                
                                if col_key in profiled_columns:
                                    logger.debug(f"Skipping {col_key} (cached)")
                                    continue
                                
                                try:
                                    profile = profiler.profile_column(db, table, col.name, col.type)
                                    # Add suffix to table name for linked DB
                                    profile.table_name = f"{table}{linked_suffix}"
                                    profiles.append(profile)
                                    profiled_columns.add(col_key)
                                    save_checkpoint()
                                    logger.debug(f"Profiled {col_key}")
                                except Exception as e:
                                    logger.warning(f"Failed to profile {col_key}: {e}")
                        except Exception as e:
                            logger.warning(f"Failed to get info for table {table}: {e}")
                
                logger.info(f"Profiled {len([p for p in profiles if linked_suffix in p.table_name])} columns from {linked_db_name}")
            
            except Exception as e:
                logger.error(f"Could not connect to {linked_db_name}: {e}")
                logger.warning(f"Continuing with only {primary_db_name} tables")
        
        if not profiles:
            raise RuntimeError("No tables profiled! Check database connections.")
        
        logger.info(f"Total profiled: {len(profiles)} columns")
        
        # 3. Summarize profiles
        logger.info("Summarizing column profiles...")
        summarizer = ProfileSummarizer(use_cache=True)
        metadata_list = [summarizer.summarize(p) for p in profiles]
        logger.info(f"Summarized {len(metadata_list)} columns")
        
        # 4. Build FAISS index
        logger.info("Building FAISS index...")
        field_index = FieldIndex()
        field_index.build_from_metadata(metadata_list, use_full_description=True, show_progress=False)
        logger.info("FAISS index built")
        
        # 5. Build LSH index
        logger.info("Building LSH index...")
        literal_matcher = SchemaLiteralMatcher(threshold=0.3, skip_constants=True)
        for m in metadata_list:
            literal_matcher.index_column_from_profile(m.profile)
        logger.info("LSH index built")
        
        # 6. Create metadata map
        metadata_map = {(m.profile.table_name, m.profile.column_name): m for m in metadata_list}
        
        # 7. Initialize Algorithm1Runner
        logger.info("Initializing SQL generation components...")
        self.algo1_runner = Algorithm1Runner(
            llm_client=self.reasoning_client,
            literal_matcher=literal_matcher,
            metadata_map=metadata_map,
            max_literal_refinements=1,
            max_syntax_fixes=1
        )
        
        # 8. Initialize FocusedSchemaBuilder
        fs_config = FocusedSchemaConfig(faiss_threshold=0.2, lsh_threshold=0.3)
        self.fs_builder = FocusedSchemaBuilder(
            field_index=field_index,
            literal_matcher=literal_matcher,
            config=fs_config
        )
        
        # 9. Initialize SchemaVariantGenerator
        self.variant_gen = SchemaVariantGenerator(metadata_map)
        
        # 10. Initialize Few-shot store
        logger.info("Loading few-shot examples...")
        store_dir = Path("./data/few_shot_store")
        self.store = FewShotStore(
            llm_client=self.masking_client, 
            store_dir=str(store_dir)
        )
        if store_dir.exists():
            self.store.load()
            logger.info(f"Loaded {len(self.store.examples) if hasattr(self.store, 'examples') else 0} few-shot examples")
        else:
            logger.info("No few-shot store found (will use zero-shot)")
        
        # 11. Initialize CandidateGenerator
        self.generator = CandidateGenerator(
            llm_client=self.reasoning_client, 
            rng_seed=42
        )
        
        # 12. Build table metadata map for Slow Lane
        self._build_table_metadata_map(metadata_list)
        
        logger.info("SQL pipeline setup complete")
        
    def _build_table_metadata_map(self, field_metadata_list):
        """
        Build table metadata map for Slow Lane planner.
        
        This is called once during initialization and cached.
        """
        logger.info("Building table metadata map for Slow Lane...")
        
        # Group field metadata by table
        from collections import defaultdict
        fields_by_table = defaultdict(list)
        for fm in field_metadata_list:
            fields_by_table[fm.profile.table_name].append(fm)
        
        # Build table profiles and summaries
        table_profiler = TableProfiler()
        table_summarizer = TableSummarizer(use_cache=True)
        
        self.table_metadata_map = {}
        
        # Process primary database tables
        with self._get_db_connection() as db:  # ← CHANGED
            for table_name, field_metas in fields_by_table.items():
                # Skip linked DB tables (they have @LINKED_DATABASE suffix)
                if '@' in table_name:
                    logger.info(f"Skipping linked table {table_name} for table metadata")
                    continue
                
                # Get column profiles
                col_profiles = [fm.profile for fm in field_metas]
                
                # Profile table
                table_profile = table_profiler.profile_table(db, table_name, col_profiles)
                
                # Summarize table
                table_metadata = table_summarizer.summarize(table_profile, field_metas)
                
                self.table_metadata_map[table_name] = table_metadata
        
        # Process linked database tables if they exist
        if self.db_config.get("linked_dsn"):
            with self._get_db_connection(is_linked=True) as db:  # ← CHANGED
                for table_name, field_metas in fields_by_table.items():
                    # Only process linked tables (with @LINKED_DATABASE suffix)
                    if '@' not in table_name:
                        continue
                    
                    # Remove suffix for querying
                    base_table_name = table_name.split('@')[0]
                    
                    col_profiles = [fm.profile for fm in field_metas]
                    table_profile = table_profiler.profile_table(db, base_table_name, col_profiles)
                    table_metadata = table_summarizer.summarize(table_profile, field_metas)
                    
                    self.table_metadata_map[table_name] = table_metadata
        
        logger.info(f"Built metadata for {len(self.table_metadata_map)} tables")
            
    async def get_table_context(self, query: str, max_tables: int = 5) -> str:
        """
        Get formatted table context for Slow Lane planner.
        
        Args:
            query: User query
            max_tables: Maximum tables to include
            
        Returns:
            Formatted string with table summaries
        """
        await self._ensure_initialized()
        
        try:
            # Get table summaries
            table_summaries = await asyncio.to_thread(
                self._get_table_summaries_sync,
                query,
                max_tables
            )
            
            if not table_summaries:
                return "No SQL tables available."
            
            # Format for planner
            lines = ["AVAILABLE SQL TABLES:", ""]
            
            for summary in table_summaries:
                lines.append(summary.format_for_planner())
                lines.append("")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Failed to get table context: {e}", exc_info=True)
            return "SQL table information unavailable."
    
    def _get_table_summaries_sync(self, query: str, max_tables: int):
        """Get table summaries synchronously (thread-safe)."""
        return self.fs_builder.get_relevant_tables(
            question=query,
            table_metadata_map=self.table_metadata_map,
            max_tables=max_tables
        )
    
    def name(self) -> str:
        return "text_to_sql_agent"
    
    def description(self) -> str:
        return """Query structured database for factual data.
        Use when query needs: numbers, counts, specific facts, calculations.
        Input: Natural language question.
        Output: Answer from database with SQL query used."""
    
    async def run(self, query: str) -> ToolResponse:
        """Run text-to-SQL pipeline with lazy initialization"""
        logger.debug(f"SQL tool processing query: {query[:100]}...")
        
        await self._ensure_initialized()
        
        try:
            # Execute entire pipeline in single thread (SQLite thread-safe)
            sql, results = await asyncio.to_thread(
                self._execute_query_sync,
                query
            )
            
            if not sql or not sql.strip():
                return ToolResponse(
                    answer="",
                    success=False,
                    error="Failed to generate valid SQL"
                )
            
            formatted_answer = self._format_results(results, sql)
            
            logger.info(f"SQL tool succeeded: {len(results)} rows")
            
            return ToolResponse(
                answer=formatted_answer,
                success=True,
                metadata={
                    "sql_query": sql,
                    "rows_returned": len(results),
                    # Oracle returns tuples without column names in results
                    "columns": [] # Column names not easily accessible from tuple results
                }
            )
            
        except Exception as e:
            logger.error(f"SQL tool exception: {e}", exc_info=True)
            return ToolResponse(
                answer="",
                success=False,
                error=f"SQL tool error: {str(e)}"
            )
    
    def _execute_query_sync(self, query: str) -> Tuple[str, list]:
        """Execute complete pipeline in single thread (SQLite thread-safe)."""
        # Build schema and variants
        focused = self.fs_builder.build(query)
        variants = self.variant_gen.generate_all(focused, include_scores=False)
        
        # Connect, solve, execute in same thread
        with self._get_db_connection() as db:
            orchestrator = VotingOrchestrator(
                runner=self.algo1_runner,
                few_shot_store=self.store,
                candidate_generator=self.generator,
                database=db,
                num_candidates=3
            )
            
            final_sql = orchestrator.solve(
                question=query,
                schema_variants=variants,
                db_id=None
            )
            
            if not final_sql or not final_sql.strip():
                return ("", [])
            
            results = db.execute_query(final_sql)
            return (final_sql, results)
    
    def _format_results(self, results: list, sql: str) -> str:
        """Format SQL results as natural language"""
        if not results:
            return "No results found."
        
        # Handle single value result
        if len(results) == 1 and len(results[0]) == 1:
            # Oracle returns tuples, not dicts
            value = results[0][0] if isinstance(results[0], tuple) else list(results[0].values())[0]
            return f"Result: {value}"
        
        row_count = len(results)
        sample_size = min(3, row_count)
        
        formatted = f"Found {row_count} results:\n"
        for i, row in enumerate(results[:sample_size], 1):
            # Handle both tuple and dict formats
            if isinstance(row, tuple):
                formatted += f"{i}. {row}\n"
            else:
                formatted += f"{i}. {dict(row)}\n"
        
        if row_count > sample_size:
            formatted += f"... and {row_count - sample_size} more rows"
        
        return formatted