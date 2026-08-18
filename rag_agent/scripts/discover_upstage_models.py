import os
import requests

API_KEY = os.getenv("UPSTAGE_API_KEY")

print("="*70)
print(" UPSTAGE MODEL DISCOVERY")
print("="*70)

# 1. Try to list models
print("\n Step 1: Fetching available models...")
try:
    response = requests.get(
        "https://api.upstage.ai/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    
    if response.status_code == 200:
        data = response.json()
        print("\n SUCCESS! Available models:")
        print("-"*70)
        
        rerank_models = []
        for model in data.get('data', []):
            model_id = model.get('id', 'Unknown')
            print(f"   • {model_id}")
            
            # Flag potential rerank models
            if 'rerank' in model_id.lower():
                rerank_models.append(model_id)
        
        if rerank_models:
            print("\n Reranking models found:")
            for m in rerank_models:
                print(f"    {m}")
        else:
            print("\n️  No models with 'rerank' in name found.")
            print("   Looking for models with 'solar' prefix:")
            solar_models = [m.get('id') for m in data.get('data', []) if 'solar' in m.get('id', '').lower()]
            for m in solar_models:
                print(f"   • {m}")
    else:
        print(f" Failed: {response.status_code}")
        print(f"Response: {response.text[:500]}")

except Exception as e:
    print(f" Error listing models: {e}")


# 2. Test known rerank endpoints with a dummy request
print("\n" + "="*70)
print(" Step 2: Testing rerank endpoints...")
print("="*70)

test_endpoints = [
    ("https://api.upstage.ai/v1/rerank", "solar-rerank-1"),
    ("https://api.upstage.ai/v1/solar/rerank", "solar-rerank-1"),
    ("https://api.upstage.ai/v1/reranking", "solar-rerank-1"),
    ("https://api.upstage.ai/v1/solar/reranking", "solar-rerank-1"),
    ("https://api.upstage.ai/v1/completions/rerank", "solar-rerank-1"),
]

for endpoint, model in test_endpoints:
    print(f"\n Testing: {endpoint}")
    print(f"   Model: {model}")
    
    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "query": "test query",
                "documents": ["test document 1", "test document 2"]
            },
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"    SUCCESS! This endpoint works!")
            print(f"   Response preview: {response.text[:200]}")
            print("\n" + "="*70)
            print(" FOUND THE WORKING ENDPOINT!")
            print(f"   URL: {endpoint}")
            print(f"   Model: {model}")
            print("="*70)
            break
        elif response.status_code == 404:
            print(f"    404 Not Found")
        elif response.status_code == 401:
            print(f"    401 Unauthorized (check API key)")
        elif response.status_code == 400:
            print(f"   ️  400 Bad Request")
            print(f"   Response: {response.text[:200]}")
            print(f"   (Endpoint might exist but our format is wrong)")
        else:
            print(f"   ️  {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    
    except Exception as e:
        print(f"    Error: {e}")


print("\n" + "="*70)
print(" Discovery Complete!")
print("="*70)