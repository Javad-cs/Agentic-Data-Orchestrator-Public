#!/usr/bin/env python3
"""
Test concurrent LLM calls to identify empty response issue.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.models import SystemConfig
from src.generation import create_llm_client


async def test_single_generation(client, query_id: int, query: str):
    """Test a single generation"""
    logger = logging.getLogger(f"test_{query_id}")
    
    try:
        logger.info(f"Starting query {query_id}: {query[:50]}...")
        
        chunks = []
        async for chunk in client.stream_generate(
            prompt=query,
            system_prompt="답변을 간결하게 작성하세요."
        ):
            if chunk.content:
                chunks.append(chunk.content)
        
        full_answer = ''.join(chunks)
        logger.info(f"Query {query_id} complete: {len(chunks)} chunks, {len(full_answer)} chars")
        
        return query_id, len(chunks), len(full_answer), None
    
    except Exception as e:
        logger.error(f"Query {query_id} failed: {e}", exc_info=True)
        return query_id, 0, 0, str(e)


async def main():
    """Test concurrent LLM calls"""
    print("\n" + "="*70)
    print("CONCURRENT LLM TEST")
    print("="*70)
    
    # Load config
    config = SystemConfig()
    
    # Create LLM client
    client = create_llm_client(config.llm)
    
    try:
        # Test queries
        queries = [
            "스테인레스강 가공 방법은?",
            "PVD 코팅 특징은?",
            "고속 가공 조건은?",
            "AlTiN 코팅 장점은?",
            "CrN 코팅 용도는?"
        ]
        
        print(f"\nTesting {len(queries)} concurrent queries...\n")
        
        # Run concurrently
        tasks = [test_single_generation(client, i+1, q) for i, q in enumerate(queries)]
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        failures = 0
        for query_id, chunks, chars, error in results:
            if error:
                print(f" Query {query_id}: FAILED - {error}")
                failures += 1
            elif chunks == 0:
                print(f"️  Query {query_id}: 0 chunks (empty response)")
                failures += 1
            else:
                print(f" Query {query_id}: {chunks} chunks, {chars} chars")
        
        print(f"\nSuccess rate: {len(queries) - failures}/{len(queries)}")
        
        if failures > 0:
            print(f"\n️  {failures} queries failed or returned empty")
            print("Check logs above for details")
        else:
            print("\n All queries succeeded!")
    
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())