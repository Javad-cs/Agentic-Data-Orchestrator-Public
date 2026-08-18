from src.config.models import SystemConfig
import os

# Set environment variables
os.environ["UPSTAGE_API_KEY"] = "test_key_123"
os.environ["DATABASE__POSTGRES_HOST"] = "db.example.com"
os.environ["DATABASE__POSTGRES_PORT"] = "5433"
os.environ["DATABASE__POSTGRES_USER"] = "test_user"
os.environ["DATABASE__POSTGRES_PASSWORD"] = "secret"

# Load config
config = SystemConfig()

# Verify nested loading works
assert config.upstage.api_key == "test_key_123"
assert config.database.postgres_host == "db.example.com"
assert config.database.postgres_port == 5433
assert config.database.postgres_user == "test_user"
assert config.database.postgres_dsn == "postgresql://test_user:secret@db.example.com:5433/rag_db"

print(" Config loading works correctly!")