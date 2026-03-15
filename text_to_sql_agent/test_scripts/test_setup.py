import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Test that basic setup is working."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings  # This imports the settings instance

def test_config():
    """Test configuration loading."""
    print(" Testing configuration...")
    print(f" Python version: {sys.version.split()[0]}")
    print(f" Virtual env: {sys.prefix}")
    
    try:
        print(f" Endpoint loaded: {settings.azure_openai_endpoint[:40]}...")
        print(f" Model: {settings.default_model}")
        print(f" API Key present: {'Yes' if settings.azure_openai_key else 'No'}")
        print(f" Data path: {settings.bird_data_path}")
        print("\n Configuration looks good!\n")
    except Exception as e:
        print(f"\n Error loading configuration: {e}")
        print("\n Make sure you have created .env file with:")
        print("   - AZURE_OPENAI_ENDPOINT")
        print("   - AZURE_OPENAI_KEY")
        print("   - DEFAULT_MODEL")
        raise

if __name__ == "__main__":
    test_config()