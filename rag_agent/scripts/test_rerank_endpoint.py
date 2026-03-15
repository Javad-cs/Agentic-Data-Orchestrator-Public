import requests
import os

api_key = os.getenv("UPSTAGE_API_KEY")

endpoints_to_try = [
    "https://api.upstage.ai/v1/rerank",
    "https://api.upstage.ai/v1/solar/rerank",
    "https://api.upstage.ai/v1/reranking",
    "https://api.upstage.ai/v1/solar/reranking",
]

for endpoint in endpoints_to_try:
    print(f"\nTrying: {endpoint}")
    
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "solar-rerank-1",
            "query": "test query",
            "documents": ["test document 1", "test document 2"]
        }
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code != 404:
        print(f" FOUND! Endpoint: {endpoint}")
        print(f"Response: {response.text[:200]}")
        break
    else:
        print(f" 404 Not Found")