#!/usr/bin/env python3
"""
Debug script to check config values are loaded correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.models import SystemConfig

def main():
    print("\n" + "="*70)
    print("CONFIG DEBUG")
    print("="*70)
    
    try:
        config = SystemConfig()
        
        print("\n LLM Configuration:")
        print(f"  Provider: {config.llm.provider}")
        print(f"  Endpoint: {config.llm.azure_endpoint}")
        print(f"  API Key: {config.llm.azure_api_key[:20]}...{config.llm.azure_api_key[-10:]}")
        print(f"  API Version: {config.llm.azure_api_version}")
        print(f"  Default Model: {config.llm.default_model}")
        print(f"  Fallback Model: {config.llm.fallback_model}")
        print(f"  Temperature: {config.llm.temperature}")
        print(f"  Max Tokens: {config.llm.max_tokens}")
        
        print("\n Upstage Configuration:")
        print(f"  API Key: {config.upstage.api_key[:20]}...{config.upstage.api_key[-10:]}")
        
        print("\n Database Configuration:")
        print(f"  Postgres DSN: {config.database.postgres_dsn}")
        print(f"  Milvus URI: {config.database.milvus_uri}")
        
        print("\n" + "="*70)
        print(" Config loaded successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n Error loading config: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()