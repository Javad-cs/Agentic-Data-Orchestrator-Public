#!/usr/bin/env python3
"""
Test FastAPI endpoints.

Tests both streaming and non-streaming modes.
"""

import asyncio
import httpx
import json
import sys


async def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*70)
    print("TEST: Health Check")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'
        print(" Health check passed")


async def test_streaming_query():
    """Test streaming query endpoint"""
    print("\n" + "="*70)
    print("TEST: Streaming Query (SSE)")
    print("="*70)
    
    query_data = {
        "query": "스테인레스강 고속 가공에 적합한 코팅은?",
        "top_k": 5,
        "language": "ko",
        "streaming": True
    }
    
    print(f"\nQuery: {query_data['query']}")
    print("\nStreaming events:")
    print("-" * 70)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/query",
            json=query_data
        ) as response:
            
            event_count = 0
            full_answer = ""
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                # SSE format: "data: {...}"
                if line.startswith("data: "):
                    event_json = line[6:]  # Remove "data: " prefix
                    event = json.loads(event_json)
                    event_count += 1
                    
                    event_type = event['type']
                    
                    if event_type == 'status':
                        print(f" STATUS: {event['content']}")
                    
                    elif event_type == 'citation':
                        cit = event['data']
                        print(f" CITATION: {cit['id']} → {cit['file']}, page {cit.get('page', 'N/A')}")
                    
                    elif event_type == 'chunk':
                        content = event['content']
                        full_answer += content
                        print(content, end='', flush=True)
                    
                    elif event_type == 'done':
                        metadata = event['metadata']
                        print(f"\n\n DONE:")
                        print(f"   Latency: {metadata['latency_ms']}ms")
                        print(f"   Answer length: {metadata['answer_length']} chars")
                        print(f"   Citations: {metadata['citation_count']}")
                    
                    elif event_type == 'error':
                        print(f"\n ERROR: {event['data']['message']}")
            
            print("-" * 70)
            print(f"\nTotal events: {event_count}")
            print(f"Full answer length: {len(full_answer)} chars")
            print("\n Streaming test passed")


async def test_non_streaming_query():
    """Test non-streaming query endpoint"""
    print("\n" + "="*70)
    print("TEST: Non-Streaming Query (JSON)")
    print("="*70)
    
    query_data = {
        "query": "PVD 코팅의 장점은?",
        "top_k": 3,
        "language": "ko",
        "streaming": False
    }
    
    print(f"\nQuery: {query_data['query']}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:8000/query",
            json=query_data
        )
        
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('answer', '')
            print(f"\nAnswer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print(f"Citations: {len(data.get('citations', []))}")
            print(f"Metadata: {data.get('metadata', {})}")
            print("\n Non-streaming test passed")
        else:
            print(f"\n Error Response:")
            try:
                error_data = response.json()
                print(f"   {error_data}")
            except:
                print(f"   {response.text}")
            print("\n  Non-streaming test failed - see error above")


async def test_concurrent_requests():
    """Test multiple concurrent requests"""
    print("\n" + "="*70)
    print("TEST: Concurrent Requests")
    print("="*70)
    
    queries = [
        "스테인레스강 가공 방법",
        "PVD 코팅 특징",
        "고속 가공 조건"
    ]
    
    async def single_query(query: str, idx: int):
        """Single query task"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    "http://localhost:8000/query",
                    json={
                        "query": query,
                        "top_k": 3,
                        "language": "ko",
                        "streaming": False
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return idx, response.status_code, len(data.get('answer', '')), None
                else:
                    error_msg = response.text
                    return idx, response.status_code, 0, error_msg
            except Exception as e:
                return idx, 0, 0, str(e)
    
    # Run queries concurrently
    tasks = [single_query(q, i) for i, q in enumerate(queries, 1)]
    results = await asyncio.gather(*tasks)
    
    print(f"\nCompleted {len(results)} concurrent queries:")
    all_passed = True
    for idx, status, answer_len, error in results:
        if status == 200:
            print(f"   Query {idx}: Status {status}, Answer length {answer_len} chars")
        else:
            print(f"   Query {idx}: Status {status}, Error: {error[:100] if error else 'Unknown'}")
            all_passed = False
    
    if all_passed:
        print("\n Concurrent requests passed")
    else:
        print("\n  Some concurrent requests failed (check errors above)")


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("FastAPI Endpoint Tests")
    print("="*70)
    print("\n  Make sure the API server is running:")
    print("   uvicorn api.main:app --reload")
    print("="*70)
    
    try:
        # Test 1: Health check
        await test_health_check()
        
        # Test 2: Streaming
        await test_streaming_query()
        
        # Test 3: Non-streaming
        await test_non_streaming_query()
        
        # Test 4: Concurrent
        await test_concurrent_requests()
        
        print("\n" + "="*70)
        print(" ALL API TESTS PASSED!")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())