import asyncpg
import asyncio
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.models import SystemConfig
from scripts.db.milvus_schema import create_milvus_collection


async def setup_postgresql(config: SystemConfig):
    """Setup PostgreSQL database"""
    print("️  Setting up PostgreSQL...")
    
    # Read schema file
    schema_path = Path(__file__).parent / "db" / "schema.sql"
    if not schema_path.exists():
        print(f" Schema file not found: {schema_path}")
        return False
    
    with open(schema_path) as f:
        schema_sql = f.read()
    
    try:
        # Connect to PostgreSQL
        conn = await asyncpg.connect(config.database.postgres_dsn)
        
        print(" Executing schema...")
        await conn.execute(schema_sql)
        
        print(" PostgreSQL schema created successfully!")
        
        # Verify tables
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        print(f"\n Created tables:")
        for row in tables:
            print(f"   - {row['table_name']}")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f" PostgreSQL setup failed: {e}")
        return False


def setup_milvus(config: SystemConfig):
    """Setup Milvus collection"""
    print("\n Setting up Milvus...")
    
    try:
        create_milvus_collection(
            collection_name=config.database.milvus_collection_name,
            dimension=config.upstage.embedding_dimension,
            milvus_uri=config.database.milvus_uri,
            drop_existing=False
        )
        return True
    except Exception as e:
        print(f" Milvus setup failed: {e}")
        return False


async def main():
    """Main setup function"""
    print(" RAG System Database Setup")
    print("=" * 50)
    
    # Load config
    try:
        config = SystemConfig()
    except Exception as e:
        print(f" Failed to load config: {e}")
        print("   Make sure .env file exists with required variables:")
        print("   - UPSTAGE_API_KEY")
        print("   - DATABASE__POSTGRES_HOST")
        print("   - DATABASE__POSTGRES_PORT")
        print("   - DATABASE__POSTGRES_DATABASE")
        print("   - DATABASE__POSTGRES_USER")
        print("   - DATABASE__POSTGRES_PASSWORD")
        return
    
    print(f" Configuration:")
    print(f"   PostgreSQL: {config.database.postgres_host}:{config.database.postgres_port}")
    print(f"   Database: {config.database.postgres_database}")
    print(f"   Milvus: {config.database.milvus_uri}")
    print(f"   Collection: {config.database.milvus_collection_name}")
    print(f"   Vector Dim: {config.upstage.embedding_dimension}")
    print()
    
    # Setup PostgreSQL
    pg_success = await setup_postgresql(config)
    
    # Setup Milvus
    milvus_success = setup_milvus(config)
    
    # Summary
    print("\n" + "=" * 50)
    if pg_success and milvus_success:
        print(" Database setup completed successfully!")
    else:
        print("️  Database setup completed with errors")
        if not pg_success:
            print("   - PostgreSQL setup failed")
        if not milvus_success:
            print("   - Milvus setup failed")


if __name__ == "__main__":
    asyncio.run(main())