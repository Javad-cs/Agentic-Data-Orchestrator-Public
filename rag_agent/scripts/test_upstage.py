# scripts/test_upstage.py
import os
import sys
import requests
import json
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
api_key = os.getenv("INGESTION__UPSTAGE_API_KEY")

if not api_key:
    print(" Error: INGESTION__UPSTAGE_API_KEY not found in .env")
    sys.exit(1)

# 2. Configuration
# document-parse' endpoint
URL = "https://api.upstage.ai/v1/document-ai/layout-analysis"
FILE_PATH = "data/inputs/sample.pdf"

def test_api():
    if not os.path.exists(FILE_PATH):
        print(f" Error: File not found at {FILE_PATH}")
        print("   Please put a small PDF file in 'data/inputs/' and name it 'sample.pdf'")
        return

    print(f" Sending {FILE_PATH} to Upstage API...")
    
    try:
        with open(FILE_PATH, "rb") as f:
            response = requests.post(
                URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"document": f},
                # We ask for HTML and Markdown to see what's best
                data={"output_formats": "['html', 'markdown']"}
            )
        
        if response.status_code == 200:
            print("\n Success! Here is the JSON structure:\n")
            data = response.json()
            
            # Inspect the 'elements' keys specifically
            print(json.dumps(data, indent=2)[:3000] + "\n... (truncated) ...")
            if "elements" in data:
                for elem in data["elements"]:
                    if elem.get("category") == "table":
                        print("\n=== FULL TABLE TEXT ===")
                        print(elem.get("text"))
                        print("=== END TABLE ===\n")
                        break
            
            # Specific check for structure
            if "content" in data:
                print("\n Structure Check: Found 'content' key.")
                if "elements" in data["content"]:
                    print(f"   Found {len(data['content']['elements'])} elements.")
                    if len(data['content']['elements']) > 0:
                        print("   First element keys:", data['content']['elements'][0].keys())
                        print("   First element category:", data['content']['elements'][0].get('category'))
            
        else:
            print(f"\n API Error {response.status_code}:")
            print(response.text)

    except Exception as e:
        print(f"\n Connection Error: {e}")

if __name__ == "__main__":
    test_api()