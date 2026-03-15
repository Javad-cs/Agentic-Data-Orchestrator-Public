"""
Inspection Script.
Verifies the contents of the Few-Shot Store and tests retrieval
to ensure the 'Dark Horse' question finds relevant examples.
"""

import logging
import sys
import random
from core import create_llm_client
from final_sql_w_cand_voting.few_shot_store import FewShotStore

# Setup logging to see the masking in action
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Inspector")

def main():
    print(" Initializing Inspector...")
    
    # Use the same model you used for population (gpt-4.1) to ensure consistent masking
    llm_client = create_llm_client(model="gpt-4.1")
    store = FewShotStore(llm_client=llm_client, store_dir= Path(__file__).parent.parent / "data" / "few_shot_store")
    
    # 1. Load Data
    store.load()
    if len(store) == 0:
        print(" Store is empty! Run populate_store.py first.")
        return

    print(f" Loaded {len(store)} examples from disk.\n")

    # 2. Inspect Quality (Random Samples)
    print("---  Data Quality Check (3 Random Samples) ---")
    examples = list(store.examples.values())
    sample = random.sample(examples, min(3, len(examples)))
    
    for i, ex in enumerate(sample):
        print(f"\n[Sample #{i+1}]")
        print(f"Original: {ex.original_question}")
        print(f"Masked:   {ex.masked_question}")
        print(f"DB ID:    {ex.db_id}")
        # Verify masking happened
        if "[VALUE]" in ex.masked_question or "[TABLE]" in ex.masked_question or "[COLUMN]" in ex.masked_question:
            print("Status:    Masked Correctly")
        else:
            print("Status:    Potentially Unmasked (Check this)")

    # 3. Test Retrieval (The "Dark Horse" Simulation)
    print("\n---  Retrieval Simulation ---")
    question = "Which superhero has the most durability published by Dark Horse Comics?"
    print(f"Query: {question}")
    
    # Minimal schema context for the test (just names)
    # The mask needs to know 'superhero' is a table and 'Dark Horse Comics' is a value
    context = """
# superhero (superhero_name, publisher_id)
# publisher (publisher_name)
# hero_attribute (attribute_value)
# attribute (attribute_name)
"""
    
    print("Searching...")
    hits = store.retrieve(question, schema_context=context, k=3)
    
    for i, (ex, score) in enumerate(hits):
        print(f"\n[Hit #{i+1}] Score: {score:.4f}")
        print(f"Found Q:  {ex.original_question}")
        print(f"Masked:   {ex.masked_question}")
        print(f"SQL:      {ex.sql}")
        
        # Check if the retrieved SQL has the "Tie-Breaking" logic (Nested Subquery)
        if "SELECT" in ex.sql and "SELECT" in ex.sql[ex.sql.find("SELECT")+1:]:
            print("Logic:     Contains Subquery (Good for ties)")
        else:
            print("Logic:    Simple Query")

if __name__ == "__main__":
    main()