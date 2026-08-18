import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Integration Example: Adapting test_phase4_pipeline.py for Oracle

This file shows the exact modifications needed to make existing
test_phase4_pipeline.py work with the Oracle database.

CHANGES NEEDED:
1. Replace SQLite connection with Oracle connection
2. Replace BIRD questions with Korean questions
3. Handle Oracle-specific SQL syntax
4. Adjust for Oracle metadata
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
import os
import pickle
from datetime import datetime

# Setup logging (SAME AS ORIGINAL)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("OracleBenchmark")

# === CHANGE 1: Import Oracle adapter instead of SQLite ===
# OLD: from core import connect_database  # SQLite version
# NEW: 
from oracle_adapter import connect_database  # Oracle version

# === ORIGINAL IMPORTS (KEEP THESE) ===
from core import create_llm_client
from profiling import ColumnProfiler, ProfileSummarizer
from profiling.metadata_enricher import MetadataEnricher
from indexing import FieldIndex, SchemaLiteralMatcher
from schema_linking import FocusedSchemaBuilder, FocusedSchemaConfig, SchemaVariantGenerator
from sql_generation import Algorithm1Runner
from config import settings

# Phase 4 Imports (KEEP THESE)
from final_sql_w_cand_voting.few_shot_store import FewShotStore
from final_sql_w_cand_voting.candidate_generator import CandidateGenerator
from final_sql_w_cand_voting.orchestrator import VotingOrchestrator


# === CHANGE 2: Replace load_stratified_data with Korean questions ===
# OLD: def load_stratified_data(target_db: str, target_per_difficulty: int = 15):
#      # Loads from BIRD dev.json
# NEW:
def load_korean_questions() -> List[Dict]:
    """
    Load Korean manufacturing questions for testing.
    
    These questions are from internal documentationument and focus on:
    - 가동률 (utilization rate)
    - 비가동사유 (downtime reasons)
    - 매출/이익 (sales/profit)
    """
    
    questions = [
        {
            "question_id": "K1",
            "question": "12월 연삭1반 가동률 자료 보여줘",
            "difficulty": "simple",
            "category": "가동률",
            "expected_tables": ["EQP_MODE_GEN_HIST", "EQP_MST@LINKED_DATABASE", "CIM_EQP_MST"],
            "description": "December utilization rate for grinding shift 1",
            # For compatibility with original pipeline
            "db_id": "manufacturing",
            "SQL": None  # No gold SQL for Korean questions yet
        },
        {
            "question_id": "K2",
            "question": "형압반 지난주 대비 가동률 감소한 설비 알려줘",
            "difficulty": "moderate",
            "category": "가동률",
            "expected_tables": ["EQP_MODE_GEN_HIST", "EQP_MST@LINKED_DATABASE", "CIM_EQP_MST"],
            "description": "Equipment with decreased utilization vs last week",
            "db_id": "manufacturing",
            "SQL": None
        },
        {
            "question_id": "K3",
            "question": "지난주 연삭1반 레오페리 6호기 가동률 알려줘",
            "difficulty": "simple",
            "category": "가동률",
            "expected_tables": ["EQP_MODE_GEN_HIST", "EQP_MST@LINKED_DATABASE", "CIM_EQP_MST"],
            "description": "Specific equipment utilization last week",
            "db_id": "manufacturing",
            "SQL": None
        },
        {
            "question_id": "K4",
            "question": "11월 한달간 ED반 설비 가동률 평균 얼마야?",
            "difficulty": "simple",
            "category": "가동률",
            "expected_tables": ["EQP_MODE_GEN_HIST", "EQP_MST@LINKED_DATABASE", "CIM_EQP_MST"],
            "description": "Average utilization for ED shift in November",
            "db_id": "manufacturing",
            "SQL": None
        },
        {
            "question_id": "K5",
            "question": "12월 소결반 설비중 5일이상 가동률 0%인 설비 리스트 정리해줘",
            "difficulty": "moderate",
            "category": "가동률",
            "expected_tables": ["EQP_MODE_GEN_HIST", "EQP_MST@LINKED_DATABASE", "CIM_EQP_MST"],
            "description": "Equipment with 0% utilization for 5+ days",
            "db_id": "manufacturing",
            "SQL": None
        }
    ]
    
    logger.info(f"Loaded {len(questions)} Korean manufacturing questions")
    return questions


# === CHANGE 3: Modify setup_pipeline for Oracle ===
# Key differences:
# 1. Use Oracle DSN instead of SQLite path
# 2. Different metadata enrichment (no BIRD dataset)
# 3. May need Oracle-specific profiling
def setup_pipeline_oracle():
    """
    Setup pipeline for Oracle database.
    
    Returns:
        Tuple of (runner, fs_builder, variant_gen, store, generator, db_connection_info)
    """
    
    # Get database connections from config (uses settings.use_docker automatically)
    primary_database_dsn, primary_database_user, primary_database_password = settings.primary_connection
    linked_database_dsn, linked_database_user, linked_database_password = settings.linked_connection
    
    logger.info(f"Using {settings.db_type.upper()} database")
    logger.info(f"Docker mode: {settings.use_docker}")
    logger.info(f"Primary DB: {settings.primary_db_name}, Linked DB: {settings.linked_db_name}{settings.linked_suffix}")
    
    # Create LLM clients
    reasoning_client = create_llm_client(model="gpt-4.1")
    masking_client = create_llm_client(model="gpt-4.1")
    
    # Load or create checkpoint
    checkpoint_file = "./data/profile_checkpoint.pkl"
    profiles = []
    profiled_columns = set()
    
    if os.path.exists(checkpoint_file):
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
    
    # Profile both databases
    profiler = ColumnProfiler()
    
    def save_checkpoint():
        try:
            os.makedirs("./data", exist_ok=True)
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'profiles': profiles, 'profiled_columns': profiled_columns}, f)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    # 1. Profile PRIMARY_DATABASE tables
    logger.info(f"Profiling {settings.primary_db_name} tables...")
    with connect_database(primary_database_dsn, primary_database_user, primary_database_password) as primary_database_db:
        primary_database_tables = ['EQP_MODE_GEN_HIST', 'CIM_EQP_MST']
        
        for table in primary_database_tables:
            try:
                table_info = primary_database_db.get_table_info(table)
                for col in table_info.columns:
                    col_key = f"{settings.primary_db_name}.{table}.{col.name}"
                    if col_key in profiled_columns:
                        logger.info(f"Skipping {col_key} (already profiled)")
                        continue
                    
                    try:
                        profile = profiler.profile_column(primary_database_db, table, col.name, col.type)
                        profiles.append(profile)
                        profiled_columns.add(col_key)
                        save_checkpoint()
                        logger.info(f"Profiled {col_key}")
                    except Exception as e:
                        logger.warning(f"Failed to profile {col_key}: {e}")
            except Exception as e:
                logger.warning(f"Failed to get info for {table}: {e}")
        
        logger.info(f"Profiled {len([p for p in profiles if settings.linked_suffix not in p.table_name])} columns from {settings.primary_db_name}")
    
    # 2. Profile LINKED_DATABASE tables
    logger.info(f"Profiling {settings.linked_db_name} tables...")
    try:
        with connect_database(linked_database_dsn, linked_database_user, linked_database_password) as linked_database_db:
            linked_database_tables = ['EQP_MST', 'DEPT_MST', 'WORK_CENTER']
            
            for table in linked_database_tables:
                try:
                    table_info = linked_database_db.get_table_info(table)
                    for col in table_info.columns:
                        col_key = f"{settings.linked_db_name}.{table}.{col.name}"
                        if col_key in profiled_columns:
                            logger.info(f"Skipping {col_key} (already profiled)")
                            continue
                        
                        try:
                            profile = profiler.profile_column(linked_database_db, table, col.name, col.type)
                            profile.table_name = f"{table}{settings.linked_suffix}"
                            profiles.append(profile)
                            profiled_columns.add(col_key)
                            save_checkpoint()
                            logger.info(f"Profiled {col_key}")
                        except Exception as e:
                            logger.warning(f"Failed to profile {col_key}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to get info for {table}: {e}")
            
            logger.info(f"Profiled {len([p for p in profiles if settings.linked_suffix in p.table_name])} columns from {settings.linked_db_name}")
    
    except Exception as e:
        logger.error(f"Could not connect to {settings.linked_db_name} database: {e}")
        logger.warning(f"Continuing with only {settings.primary_db_name} tables. Generated SQL may not work properly!")
        logger.warning(f"Make sure {settings.linked_db_name} is accessible")
    
    if not profiles:
        raise RuntimeError("No tables profiled! Check database connections.")
    
    logger.info(f"Total profiled: {len(profiles)} columns from both databases")
    
    # Summarize profiles (SAME AS ORIGINAL)
    summarizer = ProfileSummarizer(use_cache=True)
    metadata_list = [summarizer.summarize(p) for p in profiles]
    
    # === MODIFIED: Skip BIRD enrichment for production database ===
    logger.info(f"Skipping BIRD metadata enrichment for {settings.db_type} database")
    
    # Build indices (SAME AS ORIGINAL)
    field_index = FieldIndex()
    field_index.build_from_metadata(metadata_list, use_full_description=True, show_progress=False)
    
    literal_matcher = SchemaLiteralMatcher(threshold=0.3, skip_constants=True)
    for m in metadata_list:
        literal_matcher.index_column_from_profile(m.profile)
    
    metadata_map = {(m.profile.table_name, m.profile.column_name): m for m in metadata_list}
    
    # Create components (SAME AS ORIGINAL)
    algo1_runner = Algorithm1Runner(
        llm_client=reasoning_client,
        literal_matcher=literal_matcher,
        metadata_map=metadata_map,
        max_literal_refinements=1,
        max_syntax_fixes=1
    )
    
    fs_config = FocusedSchemaConfig(faiss_threshold=0.2, lsh_threshold=0.3)
    fs_builder = FocusedSchemaBuilder(
        field_index=field_index,
        literal_matcher=literal_matcher,
        config=fs_config
    )
    
    variant_gen = SchemaVariantGenerator(metadata_map)
    
    # Use empty few-shot store (disabled)
    from final_sql_w_cand_voting.few_shot_store import FewShotStore
    store = FewShotStore(llm_client=masking_client, store_dir="./data/korean_few_shot_store")
    logger.info("Using empty few-shot store (disabled)")
    
    generator = CandidateGenerator(llm_client=reasoning_client, rng_seed=42)
    
    return (
        algo1_runner, 
        fs_builder, 
        variant_gen, 
        store, 
        generator, 
        (primary_database_dsn, primary_database_user, primary_database_password)  # Return primary connection for query execution
    )

# === IMPORTANT: Handling Database Links (@LINKED_DATABASE) ===
"""
CRITICAL ISSUE: EQP_MST Table Location

EQP_MST is NOT in PRIMARY_DATABASE but in LINKED_DATABASE database.
This creates several challenges:

1. PROFILING CHALLENGE:
   - Oracle's user_tables only shows tables in the current database (PRIMARY_DATABASE)
   - You cannot profile EQP_MST@LINKED_DATABASE using standard metadata queries
   - Options:
     a) Connect directly to LINKED_DATABASE database to profile it
     b) Use Oracle's ALL_TAB_COLUMNS view to query remote tables
     c) Manually create metadata for known LINKED_DATABASE tables

2. SQL GENERATION CHALLENGE:
   - LLM must learn to add @LINKED_DATABASE suffix to table names
   - The example SQL in internal documentation shows: "FROM EQP_MST@LINKED_DATABASE MST"
   - Few-shot examples MUST include this pattern
   
3. SCHEMA LINKING CHALLENGE:
   - Field index needs to know which tables require @LINKED_DATABASE
   - Either:
     a) Profile from LINKED_DATABASE directly and add to index
     b) Add metadata manually for LINKED_DATABASE tables
     c) Post-process SQL to add @LINKED_DATABASE suffix

RECOMMENDED APPROACH:
1. Connect to LINKED_DATABASE database separately to profile EQP_MST
2. Add these profiles to metadata alongside PRIMARY_DATABASE profiles
3. Mark which tables need @LINKED_DATABASE suffix (add metadata field)
4. Create few-shot examples showing @LINKED_DATABASE usage
5. Post-process generated SQL to add @LINKED_DATABASE where needed

ALTERNATIVE SIMPLER APPROACH:
1. Don't profile EQP_MST at all
2. Use only CIM_EQP_MST (in PRIMARY_DATABASE) 
3. Notion SQL shows both tables are used together:
   - CIM_EQP_MST for joins
   - EQP_MST@LINKED_DATABASE for master data
4. Let the LLM learn the pattern from few-shot examples
"""


def profile_linked_database_tables():
    """
    Optional standalone function to profile tables from linked database.
    
    This is already integrated into setup_pipeline_oracle() above,
    but keeping this as reference if you need to profile linked tables separately.
    """
    linked_database_dsn, linked_database_user, linked_database_password = settings.linked_connection
    
    from profiling import ColumnProfiler
    
    profiler = ColumnProfiler()
    linked_database_profiles = []
    
    with connect_database(linked_database_dsn, linked_database_user, linked_database_password) as db:
        linked_database_tables = ['EQP_MST', 'DEPT_MST', 'WORK_CENTER', 'EMP_MST', 'ITEM_MST']
        
        for table in linked_database_tables:
            table_info = db.get_table_info(table)
            for col in table_info.columns:
                profile = profiler.profile_column(db, table, col.name, col.type)
                # Add database link suffix
                profile.table_name = f"{table}{settings.linked_suffix}"
                linked_database_profiles.append(profile)
    
    return linked_database_profiles


# === CHANGE 4: Modify run_scale_test ===
def run_korean_test():
    """
    Run test on Korean manufacturing questions with Oracle database.
    """
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        RESULTS_DIR = "results/sql"
        LOGS_DIR = "logs/sql"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        questions = load_korean_questions()
        
        runner, fs_builder, variant_gen, store, generator, (dsn, user, password) = setup_pipeline_oracle()
        
        log_file_path = os.path.join(RESULTS_DIR, f"korean_oracle_results_{timestamp}.jsonl")
        
        with connect_database(dsn, user, password) as db_conn, open(log_file_path, "w", encoding="utf-8") as log_file:
            
            orchestrator = VotingOrchestrator(runner, store, generator, db_conn, num_candidates=3)
            
            stats = {k: {"total": 0, "correct": 0} for k in ["simple", "moderate", "challenging"]}
            error_count = 0
            
            print(f"\nSTARTING KOREAN ORACLE TEST: {len(questions)} Questions")
            print("="*60)
            
            for i, case in enumerate(tqdm(questions)):
                q = case["question"]
                difficulty = case.get("difficulty", "unknown")
                q_id = case.get("question_id", f"Q{i}")
                
                stats[difficulty]["total"] += 1
                result_record = {
                    "question_id": q_id,
                    "difficulty": difficulty,
                    "category": case.get("category", "unknown"),
                    "question": q,
                    "pred_sql": "",
                    "status": "FAIL",
                    "error": "",
                    "description": case.get("description", "")
                }
                
                # Run Pipeline
                try:
                    # Build focused schema
                    focused = fs_builder.build(q)
                    
                    # Generate variants
                    variants = variant_gen.generate_all(focused, include_scores=False)
                    
                    # Generate SQL
                    pred_sql = orchestrator.solve(q, variants, db_id="manufacturing")
                    result_record["pred_sql"] = pred_sql
                    
                    # === MODIFIED: Handle Oracle SQL execution ===
                    # Oracle may use different syntax than SQLite
                    try:
                        pred_res = db_conn.execute_query(pred_sql)
                        result_record["status"] = "SUCCESS"
                        result_record["row_count"] = len(pred_res)
                        
                        # Since we don't have gold SQL, we can't check correctness
                        # Just mark as success if query executed
                        stats[difficulty]["correct"] += 1
                        
                    except Exception as exec_error:
                        result_record["status"] = "SQL_ERROR"
                        result_record["error"] = f"Execution failed: {exec_error}"
                        logger.error(f"SQL execution error [ID: {q_id}]: {exec_error}")
                        
                except Exception as e:
                    error_count += 1
                    result_record["status"] = "CRASH"
                    result_record["error"] = str(e)
                    logger.error(f"\nCRASH [ID: {q_id}]: {e}")
                
                # Write result
                log_file.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                log_file.flush()
            
            # Report
            print("\n" + "="*60)
            print(f"FINAL REPORT FOR KOREAN ORACLE TEST")
            print("="*60)
            
            valid_total = sum(s['total'] for s in stats.values())
            print(f"Questions Evaluated: {valid_total}")
            
            total_success = 0
            for level in ["simple", "moderate", "challenging"]:
                s = stats[level]
                if s["total"] > 0:
                    acc = (s["correct"] / s["total"]) * 100
                    print(f"  > {level.capitalize().ljust(12)}: {s['correct']}/{s['total']} ({acc:.1f}%)")
                    total_success += s["correct"]
            
            print("-" * 30)
            final_rate = (total_success / valid_total) * 100 if valid_total > 0 else 0
            print(f"SUCCESS RATE: {final_rate:.2f}%")
            print(f"System Crashes: {error_count}")
            print(f"Log: {log_file_path}")
            print("="*60)
    
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    """
    USAGE:
    
    1. Make sure oracle_adapter.py is in the same directory
    2. Install oracledb: pip install oracledb
    3. Run: python test_oracle_korean_pipeline.py
    
    EXPECTED OUTPUT:
    - Connects to Oracle database
    - Profiles key tables
    - Runs 5 Korean questions through pipeline
    - Saves results to korean_oracle_results.jsonl
    """
    run_korean_test()


# === SUMMARY OF CHANGES ===
"""
CHANGES FROM ORIGINAL test_phase4_pipeline.py:

1. Database Connection:
   - OLD: connect_database(str(sqlite_path))
   - NEW: connect_database(dsn, user, password) using Oracle adapter

2. Question Loading:
   - OLD: load_stratified_data(TARGET_DB, target_per_difficulty=15)
   - NEW: load_korean_questions()

3. Database Path:
   - OLD: settings.bird_root_path / "dev_databases" / db_name / f"{db_name}.sqlite"
   - NEW: "host.docker.internal:1521/PRIMARY_DATABASE"

4. Metadata Enrichment:
   - OLD: enricher = MetadataEnricher(settings.bird_root_path)
   - NEW: Skipped or use Oracle-specific enrichment

5. Few-Shot Store:
   - OLD: "./data/few_shot_store" (BIRD examples)
   - NEW: "./data/korean_few_shot_store" (Korean examples)

6. Gold SQL Comparison:
   - OLD: Compare pred_sql results with gold_sql results
   - NEW: No gold SQL available, just check if query executes

7. Output File:
   - OLD: benchmark_results.jsonl
   - NEW: korean_oracle_results.jsonl

8. DATABASE LINKS (@LINKED_DATABASE): ️ CRITICAL NEW ISSUE
   - EQP_MST is in LINKED_DATABASE database, not PRIMARY_DATABASE
   - Must use: FROM EQP_MST@LINKED_DATABASE (with @LINKED_DATABASE suffix)
   - Cannot profile linked tables with standard queries
   - Few-shot examples MUST show @LINKED_DATABASE usage
   - May need to connect to LINKED_DATABASE directly for profiling

WHAT STAYS THE SAME:
- Schema linking logic
- Variant generation
- SQL generation (Algorithm 1)
- Voting orchestrator
- Error handling structure
- Logging and progress tracking
"""