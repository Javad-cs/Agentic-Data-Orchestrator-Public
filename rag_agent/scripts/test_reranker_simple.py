#!/usr/bin/env python3
"""
Test Cohere reranker with Azure AI Foundry.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.models import SystemConfig
from src.retrieval.rerankers import create_reranker
import asyncio


async def test_reranker():
    print("\n" + "="*70)
    print("COHERE RERANKER TEST")
    print("="*70)
    
    # Load config
    try:
        config = SystemConfig()
        reranker_config = config.fast_lane.reranker
        
        print(f"\n Config:")
        print(f"   Enabled: {reranker_config.enabled}")
        print(f"   Provider: {reranker_config.provider}")
        print(f"   Base URL: {reranker_config.cohere_base_url}")
        print(f"   Model: {reranker_config.cohere_model}")
        print(f"   API Key: {reranker_config.cohere_api_key[:5]}...{reranker_config.cohere_api_key[-5:]}")
    except Exception as e:
        print(f"\n Config error: {e}")
        return
    
    # Create reranker
    print("\n Creating reranker...")
    reranker = create_reranker(reranker_config)
    
    if not reranker:
        print(" Reranker is disabled or failed to create")
        return
    
    print(" Reranker created")
    
    # Test documents using CandidateDocument type
    from src.retrieval.rerankers import CandidateDocument
    
    query = "스테인레스강 고속 가공"
    
    documents = [
        CandidateDocument(
            id='doc1',
            text='CrN 코팅은 낮은 마찰 계수를 가집니다.',
            metadata={'source_file': 'test.pdf', 'page': 1},
            score=0.75,
            source='dense'
        ),
        CandidateDocument(
            id='doc2',
            text='PC8110 PVD 코팅은 스테인레스강 고속 가공에 적합합니다.',
            metadata={'source_file': 'test.pdf', 'page': 2},
            score=0.82,
            source='dense'
        ),
        CandidateDocument(
            id='doc3',
            text='AlTiN 코팅은 고온에서 내열성이 우수합니다.',
            metadata={'source_file': 'test.pdf', 'page': 3},
            score=0.68,
            source='sparse'
        )
    ]
    
    print(f"\n Query: {query}")
    print(f" Documents: {len(documents)}")
    
    # Rerank
    print("\n Reranking...")
    try:
        response = await reranker.rerank(query, documents, top_k=3)
        
        print(f"\n Reranked {response.total_reranked}/{response.total_candidates} documents:\n")
        
        for result in response.results:
            print(f"   {result.reranked_rank + 1}. Score: {result.score:.4f} (was rank {result.original_rank})")
            print(f"      ID: {result.document_id}")
            print(f"      Text: {result.text[:100]}")
            print()
        
        # Check if reranking worked (doc2 should be top)
        if response.top_result and response.top_result.document_id == 'doc2':
            print(" SUCCESS! Most relevant document ranked first!")
        else:
            print("️  Unexpected ranking - check if reranker is working correctly")
        
        print(f"\nMetadata: {response.metadata}")
        
    except Exception as e:
        print(f"\n Reranking failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(test_reranker())