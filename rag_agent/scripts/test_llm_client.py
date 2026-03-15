#!/usr/bin/env python3
"""
Test script for Azure OpenAI LLM client.

Usage:
    python scripts/test_llm_client.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.models import SystemConfig
from src.generation.llm_client import create_llm_client


async def test_non_streaming():
    """Test non-streaming generation"""
    print("\n" + "="*70)
    print("TEST 1: Non-Streaming Generation")
    print("="*70)
    
    # Load config
    config = SystemConfig()
    
    # Create LLM client
    client = create_llm_client(config.llm)
    
    try:
        # Test prompt
        prompt = "스테인레스강 고속 가공에 적합한 코팅을 간단히 설명해주세요."
        
        print(f"\n Prompt: {prompt}")
        print(f" Model: {client.get_model_name()}")
        print(f"\n Generating...")
        
        # Generate
        result = await client.generate(
            prompt=prompt,
            system_prompt="당신은 금속 가공 전문가입니다. 간결하고 정확하게 답변하세요."
        )
        
        print(f"\n Generated {result.usage.get('completion_tokens', 0)} tokens")
        print(f"\n Response:\n{result.content}")
        print(f"\n Usage: {result.usage}")
    
    finally:
        await client.close()


async def test_streaming():
    """Test streaming generation"""
    print("\n" + "="*70)
    print("TEST 2: Streaming Generation")
    print("="*70)
    
    # Load config
    config = SystemConfig()
    
    # Create LLM client
    client = create_llm_client(config.llm)
    
    try:
        # Test prompt
        prompt = "스테인레스강 가공에서 PVD 코팅의 장점 3가지를 나열해주세요."
        
        print(f"\n Prompt: {prompt}")
        print(f" Model: {client.get_model_name()}")
        print(f"\n Streaming...\n")
        print(" Response (streaming):")
        print("-" * 70)
        
        # Stream generate
        full_response = ""
        async for chunk in client.stream_generate(
            prompt=prompt,
            system_prompt="당신은 금속 가공 전문가입니다."
        ):
            print(chunk.content, end='', flush=True)
            full_response += chunk.content
            
            if chunk.finish_reason:
                print(f"\n\n Finished (reason: {chunk.finish_reason})")
        
        print("-" * 70)
        print(f"\n Total characters: {len(full_response)}")
    
    finally:
        await client.close()


async def main():
    """Run all tests"""
    try:
        # Test 1: Non-streaming
        await test_non_streaming()
        
        # Test 2: Streaming
        await test_streaming()
        
        print("\n" + "="*70)
        print(" All tests passed!")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())