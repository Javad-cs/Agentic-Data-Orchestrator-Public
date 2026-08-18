#!/usr/bin/env python3
"""
Test query expansion module.
"""

import asyncio
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.config.models import SystemConfig
from src.generation import create_llm_client
from src.retrieval.query_expansion import create_query_expander


async def test_query_expansion():
    print("\n" + "="*70)
    print("QUERY EXPANSION TEST")
    print("="*70)
    
    # Load config
    config = SystemConfig()
    
    # Create LLM client
    print("\n Creating LLM client...")
    llm_client = create_llm_client(config.llm)
    
    # Create query expander
    print(" Creating query expander...")
    expander = create_query_expander(
        llm_client=llm_client,
        config=config.fast_lane.query_expansion
    )
    
    if not expander:
        print(" Query expansion is disabled")
        return
    
    print(f" Query expander created")
    print(f"   Variants: {expander.num_variants}")
    print(f"   Temperature: {expander.temperature}")
    print(f"   Parallel: {expander.parallel}")
    
    # Test queries
    test_queries = [
        ("스테인레스강 고속 가공", "ko"),
        ("PVD 코팅 장점", "ko"),
        ("CNC machining tips", "en")
    ]
    
    for query, lang in test_queries:
        print("\n" + "-"*70)
        print(f" Query: {query} (language={lang})")
        print("-"*70)
        
        # Expand query
        result = await expander.expand(query, language=lang)
        
        print(f"\n Expansion {'SUCCESS' if result.success else 'FAILED'}")
        print(f"   Latency: {result.latency_ms}ms")
        print(f"   Total queries: {result.total_queries}")
        
        print(f"\n All queries:")
        for i, q in enumerate(result.all_queries, 1):
            marker = "" if i == 1 else ""
            label = "Original" if i == 1 else f"Variant {i-1}"
            print(f"   {marker} {label}: {q}")
        
        if result.metadata:
            print(f"\n Metadata:")
            for key, value in result.metadata.items():
                print(f"   - {key}: {value}")
    
    # Test parallel vs sequential comparison
    print("\n" + "="*70)
    print("PARALLEL vs SEQUENTIAL COMPARISON")
    print("="*70)
    
    query = "고속 절삭 조건"
    
    # Parallel
    expander.parallel = True
    result_parallel = await expander.expand(query)
    print(f"\n Parallel: {result_parallel.latency_ms}ms, {len(result_parallel.expanded_queries)} variants")
    
    # Sequential
    expander.parallel = False
    result_sequential = await expander.expand(query)
    print(f" Sequential: {result_sequential.latency_ms}ms, {len(result_sequential.expanded_queries)} variants")
    
    speedup = result_sequential.latency_ms / max(result_parallel.latency_ms, 1)
    print(f"\n Speedup: {speedup:.2f}x faster with parallel")
    
    # Cleanup
    await llm_client.close()
    
    print("\n" + "="*70)
    print(" ALL TESTS COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_query_expansion())