#!/usr/bin/env python3
"""
Test Cohere Rerank API via Azure to verify correct endpoint format.
"""

import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()


async def test_cohere_rerank():
    """Test different API formats to find the correct one"""
    
    endpoint = "https://your-resource.cognitiveservices.azure.com"
    api_key = os.getenv("RERANKER__COHERE_API_KEY", "")
    deployment = "Cohere-rerank-v4.0-fast"
    api_version = "2024-05-01-preview"
    
    if not api_key:
        print(" Set RERANKER__COHERE_API_KEY in .env")
        return
    
    print("="*70)
    print("TESTING COHERE RERANK API FORMATS")
    print("="*70)
    
    query = "스테인레스강 가공"
    documents = [
        "PC8110 PVD 코팅은 스테인레스강 고속 가공에 적합합니다.",
        "AlTiN 코팅은 내열성이 우수합니다.",
        "CrN 코팅은 마찰 계수가 낮습니다."
    ]
    
    # Try Format 1: OpenAI Chat Completions (what Azure gives us)
    print("\n Test 1: OpenAI Chat Completions Format")
    url1 = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    
    payload1 = {
        "messages": [
            {
                "role": "user",
                "content": f"Rerank these documents for query: '{query}'\n\nDocuments:\n" +
                           "\n".join([f"{i}. {doc}" for i, doc in enumerate(documents)])
            }
        ]
    }
    
    await test_request(url1, api_key, payload1, "Chat Completions")
    
    # Try Format 2: Direct Cohere Rerank Format
    print("\n Test 2: Cohere Native Rerank Format")
    url2 = f"{endpoint}/cohere/deployments/{deployment}/rerank?api-version={api_version}"
    
    payload2 = {
        "query": query,
        "documents": documents,
        "top_n": 3
    }
    
    await test_request(url2, api_key, payload2, "Native Rerank")
    
    # Try Format 3: Extensions Format
    print("\n Test 3: Azure Extensions Format")
    url3 = f"{endpoint}/openai/deployments/{deployment}/extensions/chat/completions?api-version={api_version}"
    
    payload3 = {
        "messages": [{"role": "user", "content": query}],
        "dataSources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "documents": documents
                }
            }
        ]
    }
    
    await test_request(url3, api_key, payload3, "Extensions")
    
    print("\n" + "="*70)
    print("Check which format works and we'll use that one!")
    print("="*70)


async def test_request(url, api_key, payload, format_name):
    """Test a specific API request format"""
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"    SUCCESS!")
                print(f"   Response keys: {list(data.keys())}")
                print(f"   Response preview: {json.dumps(data, indent=2)[:500]}...")
            else:
                print(f"    Error: {response.text[:200]}")
    
    except Exception as e:
        print(f"    Exception: {e}")


if __name__ == "__main__":
    asyncio.run(test_cohere_rerank())