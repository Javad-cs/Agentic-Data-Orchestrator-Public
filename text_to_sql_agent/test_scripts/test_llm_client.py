import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Test LLM client functionality."""

import asyncio
from core import create_llm_client, LLMMessage


def test_sync():
    """Test synchronous LLM call."""
    print(" Testing synchronous LLM client...")
    
    client = create_llm_client()
    print(f" Client created: {client}")
    
    messages = [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="Say 'Hello, World!' and nothing else."),
    ]
    
    response = client.generate(messages, temperature=0.0)
    
    print(f" Response: {response.content}")
    print(f" Model: {response.model}")
    print(f" Tokens: {response.tokens_used}")
    print(f" Finish reason: {response.finish_reason}")


async def test_async():
    """Test asynchronous LLM call."""
    print("\n Testing asynchronous LLM client...")
    
    client = create_llm_client()
    
    messages = [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="Count from 1 to 3, separated by commas."),
    ]
    
    response = await client.generate_async(messages, temperature=0.0)
    
    print(f" Response: {response.content}")
    print(f" Tokens: {response.tokens_used}")


async def test_concurrent():
    """Test multiple concurrent requests."""
    print("\n Testing concurrent requests...")
    
    client = create_llm_client()
    
    tasks = []
    for i in range(3):
        messages = [
            LLMMessage(role="user", content=f"What is {i} + 1? Reply with just the number."),
        ]
        tasks.append(client.generate_async(messages, temperature=0.0))
    
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f" Request {i}: {response.content.strip()}")


if __name__ == "__main__":
    # Test sync
    test_sync()
    
    # Test async
    asyncio.run(test_async())
    
    # Test concurrent
    asyncio.run(test_concurrent())
    
    print("\n All LLM client tests passed!\n")