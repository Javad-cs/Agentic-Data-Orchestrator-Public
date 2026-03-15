#!/usr/bin/env python3
"""
Test LLM Router for Fast/Slow lane decision.

Usage:
    python scripts/test_router.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.models import SystemConfig
from src.generation import create_llm_client
from src.router import create_router


async def main():
    """Test router with various queries"""
    
    print("=" * 70)
    print("LLM ROUTER TEST")
    print("=" * 70)
    print()
    
    # Initialize
    print(" Initializing...")
    config = SystemConfig()
    llm_client = create_llm_client(config.llm)
    
    router = create_router(
        llm_client=llm_client,
        default_lane="fast",
        temperature=0.3
    )
    print(" Router initialized")
    print()
    
    # Test queries
    test_cases = [
        # Fast Lane examples (simple factual)
        ("스테인레스강 가공 조건은?", "ko"),
        ("PVD 코팅이란?", "ko"),
        ("What is CNC machining?", "en"),
        ("AlTiN 코팅의 특징은?", "ko"),
        
        # Slow Lane examples (complex reasoning)
        ("PVD와 CVD 코팅 중 어떤 것이 스테인레스강 가공에 더 적합한가?", "ko"),
        ("고속 절삭 조건을 최적화하려면 어떻게 해야 하는가?", "ko"),
        ("Why does tool wear increase with cutting speed?", "en"),
        ("세라믹 공구와 초경 공구의 장단점을 비교 분석해줘", "ko"),
    ]
    
    for i, (query, language) in enumerate(test_cases, 1):
        print("-" * 70)
        print(f" Query {i}: {query}")
        print(f"   Language: {language}")
        print()
        
        # Route
        decision = await router.route(query, language)
        
        # Display result
        emoji = "" if decision.is_fast_lane else ""
        print(f"{emoji} Decision: {decision.lane.upper()} LANE")
        print(f"   Confidence: {decision.confidence:.2f}")
        print(f"   Reasoning: {decision.reasoning}")
        print()
    
    print("=" * 70)
    print(" ALL TESTS COMPLETE")
    print("=" * 70)
    
    # Cleanup
    await llm_client.close()


if __name__ == "__main__":
    asyncio.run(main())