import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Phase 4 Benchmark: Scale Test.
Runs the Voting Orchestrator against a stratified batch of questions.
FIXED: Corrected FocusedSchemaBuilder initialization arguments.
"""

import json
import logging
import sys
import traceback
import random
from pathlib import Path
from typing import List, Any, Set, Dict
from collections import defaultdict
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Benchmark")

from core import connect_database, create_llm_client
from profiling import ColumnProfiler, ProfileSummarizer
from profiling.metadata_enricher import MetadataEnricher
from indexing import FieldIndex, SchemaLiteralMatcher
from schema_linking import FocusedSchemaBuilder, FocusedSchemaConfig, SchemaVariantGenerator
from sql_generation import Algorithm1Runner
from config import settings

# Phase 4 Imports
from final_sql_w_cand_voting.few_shot_store import FewShotStore
from final_sql_w_cand_voting.candidate_generator import CandidateGenerator
from final_sql_w_cand_voting.orchestrator import VotingOrchestrator

def map_difficulty(raw_diff: str) -> str:
    d = raw_diff.lower().strip()
    if d in ['simple', 'easy']: return 'simple'
    if d in ['moderate', 'medium']: return 'moderate'
    if d in ['challenging', 'hard', 'difficult']: return 'challenging'
    return 'unknown'

def load_stratified_data(target_db: str, target_per_difficulty: int = 15) -> List[Dict]:
    dev_json_path = settings.bird_root_path / "dev.json"
    if not dev_json_path.exists():
        dev_json_path = settings.bird_root_path / "dev" / "dev.json"
    
    if not dev_json_path.exists():
        raise FileNotFoundError(f"Could not find dev.json at {settings.bird_root_path}")

    with open(dev_json_path, 'r') as f:
        data = json.load(f)
    
    target_questions = [item for item in data if item['db_id'] == target_db]
    if not target_questions:
        return []

    by_difficulty = defaultdict(list)
    for q in target_questions:
        diff = map_difficulty(q.get('difficulty', 'unknown'))
        by_difficulty[diff].append(q)
    
    final_selection = []
    random.seed(42)

    if 'simple' in by_difficulty:
        logger.info("Found difficulty labels! Performing stratified sampling...")
        categories = ['simple', 'moderate', 'challenging']
        for cat in categories:
            available = by_difficulty.get(cat, [])
            random.shuffle(available)
            count = min(len(available), target_per_difficulty)
            final_selection.extend(available[:count])
            logger.info(f"  > Added {len(final_selection[-count:])} '{cat}' questions.")
    else:
        logger.warning("No labels. Random shuffle.")
        random.shuffle(target_questions)
        final_selection = target_questions[:(target_per_difficulty * 3)]
        
    logger.info(f" Final Benchmark Set: {len(final_selection)} Questions.")
    return final_selection

def normalize_result(rows: List[Any], keep_columns: int = None) -> Set[str]:
    if not rows: return set()
    normalized = set()
    for row in rows:
        vals = []
        if isinstance(row, dict):
            current_vals = [row[k] for k in sorted(row.keys())]
        elif isinstance(row, (list, tuple)):
            current_vals = list(row)
        else:
            current_vals = [row]
            
        if keep_columns and len(current_vals) > keep_columns:
            current_vals = current_vals[:keep_columns]
            
        for v in current_vals:
            if v is None: vals.append("__NULL__")
            elif isinstance(v, float): vals.append(f"{v:.6f}".rstrip('0').rstrip('.'))
            else: vals.append(str(v).strip())
        normalized.add(str(tuple(vals)))
    return normalized

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
    
    # --- Pass config as keyword argument to avoid mismatch ---
    fs_builder = FocusedSchemaBuilder(
        field_index=field_index, 
        literal_matcher=literal_matcher, 
        config=fs_config
    )
    
    variant_gen = SchemaVariantGenerator(metadata_map)
    
    store = FewShotStore(llm_client=masking_client, store_dir="./data/few_shot_store")
    store.load()
    
    generator = CandidateGenerator(llm_client=reasoning_client, rng_seed=42)
    
    return algo1_runner, fs_builder, variant_gen, store, generator, db_path

def run_scale_test():
    TARGET_DB = "superhero"
    
    try:
        questions = load_stratified_data(TARGET_DB, target_per_difficulty=15)
        runner, fs_builder, variant_gen, store, generator, db_path = setup_pipeline(TARGET_DB)
        log_file_path = "benchmark_results.jsonl"
        
        with connect_database(str(db_path)) as db_conn, open(log_file_path, "w") as log_file:
            orchestrator = VotingOrchestrator(runner, store, generator, db_conn, num_candidates=3)
            
            stats = {k: {"total": 0, "correct": 0} for k in ["simple", "moderate", "challenging", "unknown"]}
            error_count = 0
            gold_error_count = 0
            
            print(f"\n STARTING STRATIFIED TEST: {len(questions)} Questions")
            print("="*60)
            
            for i, case in enumerate(tqdm(questions)):
                q = case["question"]
                gold_sql = case["SQL"]
                difficulty = map_difficulty(case.get("difficulty", "unknown"))
                q_id = case.get("question_id", i)
                
                # 1. Execute Gold
                gold_width = None
                try:
                    gold_res = db_conn.execute_query(gold_sql)
                    if gold_res:
                        first_row = gold_res[0]
                        if isinstance(first_row, (list, tuple, dict)): gold_width = len(first_row)
                        else: gold_width = 1
                    gold_set = normalize_result(gold_res)
                except Exception as e:
                    logger.error(f" Gold SQL Error [ID: {q_id}]: {e}")
                    gold_error_count += 1
                    continue 

                stats[difficulty]["total"] += 1
                result_record = {
                    "question_id": q_id,
                    "difficulty": difficulty,
                    "question": q,
                    "gold_sql": gold_sql,
                    "pred_sql": "",
                    "status": "FAIL",
                    "error": ""
                }

                # 2. Run Pipeline
                try:
                    focused = fs_builder.build(q)
                    variants = variant_gen.generate_all(focused, include_scores=False)
                    pred_sql = orchestrator.solve(q, variants, db_id=None)
                    result_record["pred_sql"] = pred_sql
                    
                    pred_res = db_conn.execute_query(pred_sql)
                    pred_set = normalize_result(pred_res, keep_columns=gold_width)
                    
                    if gold_set == pred_set:
                        stats[difficulty]["correct"] += 1
                        result_record["status"] = "PASS"
                    else:
                        pass
                        
                except Exception as e:
                    error_count += 1
                    result_record["status"] = "CRASH"
                    result_record["error"] = str(e)
                    logger.error(f"\n CRASH [ID: {q_id}]: {e}")

                log_file.write(json.dumps(result_record) + "\n")
                log_file.flush()

            # --- REPORT ---
            print("\n" + "="*60)
            print(f" FINAL REPORT FOR '{TARGET_DB}'")
            valid_total = sum(s['total'] for s in stats.values())
            print(f"Questions Evaluated: {valid_total}")
            
            total_correct = 0
            for level in ["simple", "moderate", "challenging", "unknown"]:
                s = stats[level]
                if s["total"] > 0:
                    acc = (s["correct"] / s["total"]) * 100
                    print(f"  > {level.capitalize().ljust(12)}: {s['correct']}/{s['total']} ({acc:.1f}%)")
                    total_correct += s["correct"]
            
            print("-" * 30)
            final_acc = (total_correct / valid_total) * 100 if valid_total > 0 else 0
            print(f" OVERALL ACCURACY: {final_acc:.2f}%")
            print(f" System Crashes:   {error_count}")
            print(f" Gold Skipped:     {gold_error_count}")
            print(f" Log:              {log_file_path}")
            print("="*60)

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_scale_test()