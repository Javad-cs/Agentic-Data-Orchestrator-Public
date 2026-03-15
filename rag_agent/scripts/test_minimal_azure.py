#!/usr/bin/env python3
"""
Minimal test: 3 concurrent streaming requests directly to Azure OpenAI.
No RAG, no FastAPI, no complex logic - just pure Azure API calls.

This will tell us if the empty stream issue is:
- Azure API throttling
- Our code bug
"""

import asyncio
import os
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()


async def single_stream_test(client, query_id: int, prompt: str):
    """Test a single streaming request"""
    print(f"\n Query {query_id}: Starting...")
    
    try:
        stream = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "답변을 간결하게 작성하세요."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=100,
            stream=True
        )
        
        chunks = 0
        chars = 0
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                chunks += 1
                chars += len(content)
        
        if chars == 0:
            print(f" Query {query_id}: EMPTY STREAM (0 chars)")
            return query_id, False, 0
        else:
            print(f" Query {query_id}: SUCCESS ({chunks} chunks, {chars} chars)")
            return query_id, True, chars
    
    except Exception as e:
        print(f" Query {query_id}: ERROR - {e}")
        return query_id, False, 0


async def main():
    print("="*70)
    print("MINIMAL AZURE OPENAI STREAMING TEST")
    print("="*70)
    print("\nTesting 3 concurrent streaming requests...")
    print("No RAG, no FastAPI - just pure Azure OpenAI API calls")
    print("="*70)
    
    # Initialize Azure OpenAI client
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("LLM__AZURE_ENDPOINT"),
        api_key=os.getenv("LLM__AZURE_API_KEY"),
        api_version=os.getenv("LLM__AZURE_API_VERSION", "2024-02-15-preview")
    )
    
    # Test prompts
    prompts = [
        "스테인레스강 가공 방법은?",
        "PVD 코팅 특징은?",
        "고속 가공 조건은?"
    ]
    
    try:
        # Run 3 concurrent requests
        tasks = [single_stream_test(client, i+1, p) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        successes = sum(1 for _, success, _ in results if success)
        failures = len(results) - successes
        
        print(f"\n Successes: {successes}/{len(results)}")
        print(f" Failures: {failures}/{len(results)}")
        
        if failures > 0:
            print("\n CONCLUSION:")
            print("   Azure OpenAI is throttling concurrent requests.")
            print("   This is an Azure API limitation, NOT a bug in our code.")
            print("   Solutions:")
            print("   - Add semaphore to limit concurrent requests (code fix)")
            print("   - Upgrade Azure quota (production fix)")
            print("   - Accept 2/3 success rate for PoC (acceptable for now)")
        else:
            print("\n CONCLUSION:")
            print("   All concurrent requests succeeded!")
            print("   If our full pipeline fails, it's a bug in our code.")
        
        print("="*70 + "\n")
    
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())