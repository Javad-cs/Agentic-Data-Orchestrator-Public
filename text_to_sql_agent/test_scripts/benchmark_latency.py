import json
import logging
import sys
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm

# Setup logging
# We keep this to see progress logs, but the Table will be printed separately at the end
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LatencyTest")

# Imports
from core import connect_database, create_llm_client
from profiling import ColumnProfiler, ProfileSummarizer
from profiling.metadata_enricher import MetadataEnricher
from indexing import FieldIndex, SchemaLiteralMatcher
from schema_linking import FocusedSchemaBuilder, FocusedSchemaConfig, SchemaVariantGenerator
from sql_generation import Algorithm1Runner
from config import settings
from final_sql_w_cand_voting.few_shot_store import FewShotStore
from final_sql_w_cand_voting.candidate_generator import CandidateGenerator
from final_sql_w_cand_voting.orchestrator import VotingOrchestrator

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def map_difficulty(raw_diff: str) -> str:
    d = raw_diff.lower().strip()
    if d in ['simple', 'easy']: return 'simple'
    if d in ['moderate', 'medium']: return 'moderate'
    if d in ['challenging', 'hard', 'difficult']: return 'challenging'
    return 'unknown'

def load_stratified_subset(target_db: str, count_per_diff: int = 5) -> List[Dict]:
    """Loads a strict subset of questions: 5 simple, 5 moderate, 5 challenging."""
    dev_json_path = settings.bird_root_path / "dev" / "dev.json"
    if not dev_json_path.exists():
        dev_json_path = settings.bird_root_path / "dev.json"
    
    if not dev_json_path.exists():
        raise FileNotFoundError(f"Could not find dev.json at {settings.bird_root_path}")

    with open(dev_json_path, 'r') as f:
        data = json.load(f)
    
    target_questions = [item for item in data if item['db_id'] == target_db]
    by_difficulty = defaultdict(list)
    
    for q in target_questions:
        diff = map_difficulty(q.get('difficulty', 'unknown'))
        by_difficulty[diff].append(q)
    
    final_selection = []
    random.seed(42)
    
    for cat in ['simple', 'moderate', 'challenging']:
        available = by_difficulty.get(cat, [])
        random.shuffle(available)
        selected = available[:count_per_diff]
        final_selection.extend(selected)
        
    return final_selection

def setup_pipeline(db_name: str):
    db_path = settings.bird_root_path / "dev_databases" / db_name / f"{db_name}.sqlite"
    
    reasoning_client = create_llm_client() 
    masking_client = create_llm_client(model="gpt-4.1") 
    
    with connect_database(str(db_path)) as db:
        profiler = ColumnProfiler()
        profiles = []
        for table in db.get_tables():
            for col in db.get_table_info(table).columns:
                profiles.append(profiler.profile_column(db, table, col.name, col.type))
    
    summarizer = ProfileSummarizer(use_cache=True)
    metadata_list = [summarizer.summarize(p) for p in profiles]
    
    enricher = MetadataEnricher(settings.bird_root_path)
    metadata_list = enricher.enrich_batch(db_name, metadata_list)
    
    field_index = FieldIndex()
    field_index.build_from_metadata(metadata_list, use_full_description=True, show_progress=False)
    
    literal_matcher = SchemaLiteralMatcher(threshold=0.3, skip_constants=True)
    for m in metadata_list:
        literal_matcher.index_column_from_profile(m.profile)
        
    metadata_map = {(m.profile.table_name, m.profile.column_name): m for m in metadata_list}
    
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
    store = FewShotStore(llm_client=masking_client, store_dir= Path(__file__).parent.parent / "data" / "few_shot_store")
    store.load()
    generator = CandidateGenerator(llm_client=reasoning_client, rng_seed=42)
    
    return fs_builder, variant_gen, store, generator, algo1_runner, db_path

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------

def run_latency_test():
    TARGET_DB = "superhero"
    
    try:
        questions = load_stratified_subset(TARGET_DB, count_per_diff=5)
        if not questions:
            logger.error("No questions found.")
            return

        logger.info("Initializing Pipeline components...")
        fs_builder, variant_gen, store, generator, runner, db_path = setup_pipeline(TARGET_DB)
        
        with connect_database(str(db_path)) as db_conn:
            orchestrator = VotingOrchestrator(runner, store, generator, db_conn, num_candidates=3)
            
            # We will store results here instead of printing them immediately
            results_buffer = []
            
            print(f"\n STARTING BENCHMARK ({len(questions)} items)... check logs for progress.")

            for i, case in enumerate(tqdm(questions)):
                q = case["question"]
                q_id = case.get("question_id", i)
                diff = map_difficulty(case.get("difficulty", "unknown"))
                
                # Record
                row = {
                    "id": q_id,
                    "diff": diff,
                    "link": 0.0,
                    "var": 0.0,
                    "orch": 0.0,
                    "total": 0.0,
                    "status": "ERR"
                }

                try:
                    # 1. Schema Linking
                    t0 = time.perf_counter()
                    focused = fs_builder.build(q)
                    t1 = time.perf_counter()
                    
                    # 2. Variant Gen
                    variants = variant_gen.generate_all(focused, include_scores=False)
                    t2 = time.perf_counter()
                    
                    # 3. Orchestrator
                    pred_sql = orchestrator.solve(q, variants, db_id=None)
                    t3 = time.perf_counter()
                    
                    row["link"] = t1 - t0
                    row["var"] = t2 - t1
                    row["orch"] = t3 - t2
                    row["total"] = t3 - t0
                    row["status"] = "OK" if pred_sql else "EMPTY"

                except Exception as e:
                    row["status"] = "CRASH"
                    logger.error(f"Error on {q_id}: {e}")
                
                results_buffer.append(row)

            # -------------------------------------------------------------------------
            # FINAL REPORT (Printed once at the very end)
            # -------------------------------------------------------------------------
            print("\n\n")
            print(f"{'='*90}")
            print(f" DETAILED LATENCY REPORT: {TARGET_DB}")
            print(f"{'='*90}")
            print(f"{'ID':<6} | {'Difficulty':<12} | {'Link(s)':<8} | {'Var(s)':<8} | {'Orch(s)':<8} | {'Total(s)':<8} | {'Status'}")
            print(f"{'-'*90}")

            stats = {"simple": [], "moderate": [], "challenging": []}

            for r in results_buffer:
                # Add to stats if valid
                if r["status"] == "OK":
                    if r["diff"] in stats:
                        stats[r["diff"]].append(r)

                # Print Row
                if r["status"] == "CRASH":
                     print(f"{r['id']:<6} | {r['diff']:<12} | {'-':>8} | {'-':>8} | {'-':>8} | {'-':>8} | CRASH")
                else:
                    print(f"{r['id']:<6} | {r['diff']:<12} | {r['link']:8.3f} | {r['var']:8.3f} | {r['orch']:8.3f} | {r['total']:8.3f} | {r['status']}")

            print(f"{'='*90}")
            print(" STATS SUMMARY (Mean / P95)")
            print(f"{'-'*90}")
            print(f"{'Metric':<20} | {'Simple':<18} | {'Moderate':<18} | {'Challenging':<18}")
            print(f"{'-'*90}")
            
            def get_stat_str(rows, key):
                if not rows: return "0.00 / 0.00"
                vals = [x[key] for x in rows]
                avg = np.mean(vals)
                p95 = np.percentile(vals, 95)
                return f"{avg:.2f} / {p95:.2f}"

            for key, name in [("link", "Schema Linking"), ("var", "Variant Gen"), ("orch", "Orchestrator"), ("total", "Total E2E")]:
                s_stat = get_stat_str(stats["simple"], key)
                m_stat = get_stat_str(stats["moderate"], key)
                c_stat = get_stat_str(stats["challenging"], key)
                print(f"{name:<20} | {s_stat:<18} | {m_stat:<18} | {c_stat:<18}")
            
            print(f"{'='*90}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_latency_test()