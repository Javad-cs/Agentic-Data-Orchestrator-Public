"""
Few-Shot Store for retrieving structurally similar SQL examples.
Implements the BIRD masking + vector search strategy with persistence, scoring, and safety checks.
"""

import json
import os
import logging
import hashlib
import numpy as np
import faiss
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer

from .query_masker import QueryMasker
from core import BaseLLMClient

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class FewShotExample:
    original_question: str
    masked_question: str
    sql: str
    db_id: str = "unknown"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict) -> 'FewShotExample':
        return FewShotExample(**data)

class FewShotStore:
    """
    Vector store for masked few-shot examples.
    Supports:
    - Masking (Question -> Structure) via LLM
    - Vector Search (FAISS IndexFlatIP)
    - Persistence
    """
    
    def __init__(self, llm_client: BaseLLMClient, model_name: str = "all-MiniLM-L6-v2", store_dir: str = "./data/few_shot_store"):
        self.model_name = model_name
        self.store_dir = store_dir
        self.llm_client = llm_client 
        
        self._embedding_model = None 
        # Initialize Masker with the robust client
        self.masker = QueryMasker(llm_client)
        
        # FAISS index (ID-mapped for safety)
        self.index = None
        self.examples: Dict[int, FewShotExample] = {} 
        self.next_id = 0
        self.dimension = 0
        
        self._content_hashes = set()
        
        os.makedirs(store_dir, exist_ok=True)

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.model_name}...")
            self._embedding_model = SentenceTransformer(self.model_name)
        return self._embedding_model
    
    def __len__(self) -> int:
        return len(self.examples)

    def _hash_example(self, masked_q: str, sql: str, db_id: str) -> str:
        content = json.dumps({
            "q": masked_q, 
            "sql": sql, 
            "db": db_id
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def add_examples(self, data: List[Dict[str, str]]):
        """
        Add new examples to the store (in-memory). 
        """
        if not data:
            return

        logger.info(f"Processing {len(data)} incoming examples...")
        
        masked_texts = []
        temp_examples = []
        skipped = 0
        
        for item in data:
            if 'masked_question' in item:
                masked_text = item['masked_question']
            else:
                # Require schema context if masking on the fly
                context = item.get('schema_context', '')
                masked_text = self.masker.mask(item['question'], context)

            sql = item['sql']
            db_id = item.get('db_id', 'unknown')
            
            ex_hash = self._hash_example(masked_text, sql, db_id)
            if ex_hash in self._content_hashes:
                skipped += 1
                continue
            
            self._content_hashes.add(ex_hash)
            
            ex = FewShotExample(
                original_question=item['question'],
                masked_question=masked_text,
                sql=sql,
                db_id=db_id
            )
            
            masked_texts.append(masked_text)
            temp_examples.append(ex)

        if skipped > 0:
            logger.info(f"Skipped {skipped} duplicate examples.")

        if not temp_examples:
            return

        embeddings = self.embedding_model.encode(masked_texts, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        
        if self.index is None:
            self.dimension = embeddings.shape[1]
            base_index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap2(base_index)
            
        start_id = self.next_id
        ids = np.arange(start_id, start_id + len(embeddings)).astype('int64')
        
        self.index.add_with_ids(embeddings, ids)
        
        for i, ex in enumerate(temp_examples):
            self.examples[int(ids[i])] = ex
            
        self.next_id += len(embeddings)
        logger.info(f"Added {len(embeddings)} new examples. Total count: {len(self.examples)}")

    def retrieve(self, question: str, schema_context: str, k: int = 5, threshold: float = 0.4, filter_db_id: Optional[str] = None) -> List[Tuple[FewShotExample, float]]:
        """Retrieve similar examples using masked similarity."""
        if self.index is None or len(self.examples) == 0:
            logger.warning("Store is empty.")
            return []
        
        # 1. Mask
        masked_q_text = self.masker.mask(question, schema_context)
        logger.debug(f"Retrieved using mask: {masked_q_text}")
        
        # 2. Embed
        query_vec = self.embedding_model.encode([masked_q_text], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        
        # 3. Search
        pool_size = len(self.examples)
        if filter_db_id:
            search_k = min(max(k * 3, 50), pool_size)
        else:
            search_k = min(k, pool_size)
        
        scores, ids = self.index.search(query_vec, search_k)
        
        # 4. Filter
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1: continue
            if score < threshold: continue
            
            ex = self.examples[idx]
            if filter_db_id and ex.db_id != filter_db_id:
                continue
                
            results.append((ex, float(score)))
            if len(results) >= k:
                break
            
        return results

    def save(self):
        """Save index and metadata to disk."""
        if self.index is None:
            logger.warning("Nothing to save.")
            return
            
        index_path = os.path.join(self.store_dir, "index.faiss")
        meta_path = os.path.join(self.store_dir, "metadata.json")
        
        faiss.write_index(self.index, index_path)
        
        data = {
            "model_name": self.model_name,
            "next_id": self.next_id,
            "dimension": self.dimension,
            "examples": {str(k): v.to_dict() for k, v in self.examples.items()}
        }
        with open(meta_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved store to {self.store_dir}")

    def load(self):
        """Load index and metadata from disk."""
        index_path = os.path.join(self.store_dir, "index.faiss")
        meta_path = os.path.join(self.store_dir, "metadata.json")
        
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            logger.warning(f"No existing store found at {self.store_dir}")
            return
            
        try:
            with open(meta_path, 'r') as f:
                data = json.load(f)
            
            stored_model = data.get("model_name", "unknown")
            if stored_model != self.model_name:
                raise ValueError(f"Model mismatch! Store uses '{stored_model}' but currently using '{self.model_name}'.")

            self.index = faiss.read_index(index_path)
            
            if self.index.d != data["dimension"]:
                raise ValueError(f"Dimension mismatch! Index: {self.index.d}, Metadata: {data['dimension']}")

            self.next_id = data["next_id"]
            self.dimension = data["dimension"]
            self.examples = {int(k): FewShotExample.from_dict(v) for k, v in data["examples"].items()}
            
            self._content_hashes = set()
            for ex in self.examples.values():
                h = self._hash_example(ex.masked_question, ex.sql, ex.db_id)
                self._content_hashes.add(h)
            
            logger.info(f"Loaded store with {len(self.examples)} examples.")
            
        except Exception as e:
            logger.error(f"Failed to load store: {e}. Resetting state.")
            self.index = None
            self.examples = {}
            self.next_id = 0
            self.dimension = 0
            self._content_hashes = set()