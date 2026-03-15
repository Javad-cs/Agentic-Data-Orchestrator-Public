"""
Phase 5: Populate Few-Shot Store.
Strategy: "Difficulty-Stratified Leave-One-Out".
Reads 'dev.json', EXCLUDES 'superhero'.
Features:
- GPT-4.1 Masking (Generative)
- Deduplication (No repeated questions)
- Difficulty Normalization (Maps 'medium'->'moderate', etc.)
- Deterministic Schema Sorting (Tables & Columns)
"""

import json
import logging
import sys
import random
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Populate")

from config import settings
from core import create_llm_client
from final_sql_w_cand_voting.few_shot_store import FewShotStore

def map_difficulty(raw_diff: str) -> str:
    """Standardize difficulty labels."""
    d = raw_diff.lower().strip()
    if d in ['simple', 'easy']: return 'simple'
    if d in ['moderate', 'medium']: return 'moderate'
    if d in ['challenging', 'hard', 'difficult']: return 'challenging'
    return 'unknown'

def build_schema_cache(tables_json_path: Path) -> dict:
    """
    Pre-loads schema strings for ALL databases into a dictionary.
    Returns: {db_id: formatted_schema_string}
    """
    try:
        with open(tables_json_path, 'r') as f:
            all_dbs = json.load(f)
    except FileNotFoundError:
        logger.error(f"Schema metadata file not found at {tables_json_path}")
        return {}

    schema_cache = {}
    
    for db_meta in all_dbs:
        db_id = db_meta['db_id']
        
        table_names = db_meta['table_names_original']
        columns = db_meta['column_names_original'] 
        
        # Map table_idx -> list of column names
        table_cols = defaultdict(list)
        for col_idx, (table_idx, col_name) in enumerate(columns):
            if table_idx == -1: continue 
            table_cols[table_idx].append(col_name)
            
        lines = []
        # FIX: Sort table names alphabetically for strict determinism
        # This ensures the schema string matches what the Orchestrator generates later.
        sorted_indices = sorted(range(len(table_names)), key=lambda k: table_names[k])
        
        for table_idx in sorted_indices:
            name = table_names[table_idx]
            # FIX: Sort columns too
            cols = sorted(table_cols[table_idx])
            cols_str = ", ".join(cols)
            lines.append(f"# {name} ({cols_str})")
            
        schema_cache[db_id] = "\n".join(lines)
        
    return schema_cache

def main():
    # --- CONFIGURATION ---
    bird_root = settings.bird_root_path
    dev_json_path = bird_root / "dev.json"
    tables_json_path = bird_root / "dev_tables.json"
    
    if not tables_json_path.exists():
        tables_json_path = bird_root / "tables.json"
    
    if not dev_json_path.exists():
        logger.error(f" Data file not found: {dev_json_path}")
        return

    # Safety: Reset store
    store_dir = Path(__file__).parent.parent / "data" / "few_shot_store"
    if store_dir.exists():
        logger.warning(f"Store directory {store_dir} exists. Deleting for fresh population...")
        shutil.rmtree(store_dir)

    # --- ARCHITECTURE: Use GPT-4.1 for Masking ---
    MASKING_MODEL = "gpt-4.1" 
    logger.info(f"Initializing Masking Client using model: {MASKING_MODEL}...")
    llm_client = create_llm_client(model=MASKING_MODEL)
    store = FewShotStore(llm_client=llm_client, store_dir=str(store_dir))
    
    # 1. Pre-load Schema
    logger.info(f"Pre-loading schema metadata...")
    schema_cache = build_schema_cache(tables_json_path)
    
    # 2. Load Data
    logger.info(f"Loading data from {dev_json_path}...")
    with open(dev_json_path, 'r') as f:
        data = json.load(f)
    
    # --- STRATIFIED SAMPLING (Robust) ---
    random.seed(42)

    # Group questions by DB -> Difficulty
    db_buckets = defaultdict(lambda: defaultdict(list))
    
    for i, item in enumerate(data):
        # Attach ID if missing for deduping
        if 'question_id' not in item:
            item['question_id'] = i
            
        if item['db_id'] == 'superhero': 
            continue # Leakage Protection
            
        diff = map_difficulty(item.get('difficulty', 'unknown'))
        db_buckets[item['db_id']][diff].append(item)
        
    data_to_process = []
    # Track seen IDs to prevent duplicates during fallback
    seen_ids = set()
    
    SAMPLES_PER_DIFF = 1  # 1 Simple, 1 Mod, 1 Hard per DB
    
    logger.info(f"Sampling strategy: {SAMPLES_PER_DIFF} per difficulty per DB.")
    
    for db_id, buckets in sorted(db_buckets.items()): # Sort DBs for determinism
        if db_id not in schema_cache: continue
        
        for difficulty in ['simple', 'moderate', 'challenging']:
            candidates = buckets.get(difficulty, [])
            
            # Filter candidates that were already picked (by fallback)
            valid_candidates = [c for c in candidates if c['question_id'] not in seen_ids]
            
            needed = SAMPLES_PER_DIFF
            
            # 1. Try to fill from current difficulty
            if valid_candidates:
                picked = random.sample(valid_candidates, min(len(valid_candidates), needed))
                for p in picked:
                    seen_ids.add(p['question_id'])
                    data_to_process.append(p)
                needed -= len(picked)
            
            # 2. Fallback if still needed (Pick from ANY difficulty in this DB)
            if needed > 0:
                # Gather all other available questions for this DB
                all_others = []
                for other_diff in ['simple', 'moderate', 'challenging', 'unknown']:
                    for q in buckets.get(other_diff, []):
                        if q['question_id'] not in seen_ids:
                            all_others.append(q)
                
                if all_others:
                    fallback_picks = random.sample(all_others, min(len(all_others), needed))
                    for p in fallback_picks:
                        seen_ids.add(p['question_id'])
                        data_to_process.append(p)

    logger.info(f" Populating store with {len(data_to_process)} unique, stratified examples.")

    examples_buffer = []
    
    for item in tqdm(data_to_process):
        question = item['question']
        sql = item['SQL']
        db_id = item['db_id']
        
        schema_context = schema_cache.get(db_id, "")
        if not schema_context: continue

        masked_q = store.masker.mask(question, schema_context)
        
        if not masked_q: continue

        examples_buffer.append({
            "question": question,
            "masked_question": masked_q, 
            "sql": sql,
            "db_id": db_id
        })

    # Save once at the end (Efficiency)
    if examples_buffer:
        store.add_examples(examples_buffer)
        store.save()
        
    logger.info(f" Done! Store populated with {len(store)} examples.")

if __name__ == "__main__":
    main()