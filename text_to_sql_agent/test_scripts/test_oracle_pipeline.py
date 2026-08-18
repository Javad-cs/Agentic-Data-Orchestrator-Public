import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)

"""
Oracle Database Test Pipeline for Korean Manufacturing Questions
Tests the text-to-SQL system against real Korean manufacturing database
"""

import oracledb
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("OracleTest")

# --- DATABASE CREDENTIALS ---
# PRIMARY_DATABASE (Primary)
PRIMARY_DATABASE_USER = os.getenv("PRIMARY_USER")
PRIMARY_DATABASE_PASS = os.getenv("PRIMARY_PASSWORD")

# LINKED_DATABASE (Linked database, different port)
LINKED_DATABASE_USER = os.getenv("LINKED_USER")
LINKED_DATABASE_PASS = os.getenv("LINKED_PASSWORD")

# DSN Configuration
# Use host.docker.internal when running inside Docker container
# Use 127.0.0.1 when running directly on host machine
USE_DOCKER = os.environ.get('USE_DOCKER', 'true').lower() == 'true'

if USE_DOCKER:
    # Running inside Docker container
    PRIMARY_DATABASE_DSN = os.getenv("PRIMARY_DSN")
    LINKED_DATABASE_DSN = os.getenv("LINKED_DSN")
    logger.info(" Using Docker mode (host.docker.internal)")
else:
    # Running directly on host machine
    PRIMARY_DATABASE_DSN = os.getenv("PRIMARY_DSN")
    LINKED_DATABASE_DSN = os.getenv("LINKED_DSN")
    logger.info(" Using host mode (127.0.0.1)")

# --- TEST QUESTIONS (Korean Manufacturing Domain) ---
TEST_QUESTIONS = [
    {
        "id": "Q1",
        "question": "12월 연삭1반 가동률 자료 엑셀 파일로 생성해줘",
        "category": "가동률",
        "difficulty": "simple",
        "expected_tables": ["EQP_MODE_GEN_HIST", "EQP_MST"],
        "notes": "December utilization rate for grinding shift 1"
    },
    {
        "id": "Q2",
        "question": "형압반 지난주 대비 가동률 감소한 설비 알려줘",
        "category": "가동률",
        "difficulty": "moderate",
        "expected_tables": ["EQP_MODE_GEN_HIST", "EQP_MST"],
        "notes": "Equipment with decreased utilization vs last week"
    },
    {
        "id": "Q3",
        "question": "12월 소결반 설비중 5일이상 가동률 0%인 설비 리스트 정리해줘",
        "category": "가동률",
        "difficulty": "moderate",
        "expected_tables": ["EQP_MODE_GEN_HIST", "EQP_MST"],
        "notes": "Equipment with 0% utilization for 5+ days"
    },
    {
        "id": "Q4",
        "question": "지난주 연삭1반 레오페리 6호기 가동률 알려줘",
        "category": "가동률",
        "difficulty": "simple",
        "expected_tables": ["EQP_MODE_GEN_HIST", "EQP_MST"],
        "notes": "Specific equipment utilization last week"
    },
    {
        "id": "Q5",
        "question": "11월 한달간 ED반 설비 가동률 평균 얼마야?",
        "category": "가동률",
        "difficulty": "simple",
        "expected_tables": ["EQP_MODE_GEN_HIST", "EQP_MST"],
        "notes": "Average utilization for ED shift in November"
    }
]


class OracleConnection:
    """Wrapper for Oracle database connection"""
    
    def __init__(self, dsn, user, password):
        self.dsn = dsn
        self.user = user
        self.password = password
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Connect to Oracle database"""
        try:
            oracledb.init_oracle_client()
            self.conn = oracledb.connect(
                user=self.user,
                password=self.password,
                dsn=self.dsn
            )
            self.cursor = self.conn.cursor()
            logger.info(f" Connected to {self.dsn}")
            return True
        except Exception as e:
            logger.error(f" Connection failed: {e}")
            return False
    
    def get_tables(self) -> List[str]:
        """Get list of available tables"""
        try:
            # Get tables from PRIMARY_DATABASE
            self.cursor.execute("""
                SELECT table_name 
                FROM user_tables 
                ORDER BY table_name
            """)
            tables = [row[0] for row in self.cursor.fetchall()]
            logger.info(f"Found {len(tables)} tables in {self.dsn}")
            return tables
        except Exception as e:
            logger.error(f"Error getting tables: {e}")
            return []
    
    def get_table_schema(self, table_name: str) -> List[Dict]:
        """Get schema information for a table"""
        try:
            self.cursor.execute(f"""
                SELECT 
                    column_name,
                    data_type,
                    data_length,
                    nullable
                FROM user_tab_columns
                WHERE table_name = '{table_name}'
                ORDER BY column_id
            """)
            
            columns = []
            for row in self.cursor.fetchall():
                columns.append({
                    "name": row[0],
                    "type": row[1],
                    "length": row[2],
                    "nullable": row[3]
                })
            return columns
        except Exception as e:
            logger.error(f"Error getting schema for {table_name}: {e}")
            return []
    
    def execute_query(self, sql: str) -> List[Any]:
        """Execute a SQL query and return results"""
        try:
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info(" Connection closed")


def inspect_database(db: OracleConnection):
    """Inspect database structure and key tables"""
    logger.info("\n" + "="*60)
    logger.info("DATABASE INSPECTION")
    logger.info("="*60)
    
    # Get all tables
    tables = db.get_tables()
    
    # key tables mentioned in the documentation
    key_tables = ["EQP_MODE_GEN_HIST", "CIM_EQP_MST", "EQP_MST"]
    
    for table in key_tables:
        if table in tables:
            logger.info(f"\n Table: {table}")
            schema = db.get_table_schema(table)
            logger.info(f"   Columns ({len(schema)}):")
            for col in schema[:10]:  # Show first 10 columns
                logger.info(f"      - {col['name']} ({col['type']})")
            
            # Try to get row count
            try:
                result = db.execute_query(f"SELECT COUNT(*) FROM {table}")
                count = result[0][0] if result else 0
                logger.info(f"   Row count: {count:,}")
            except Exception as e:
                logger.info(f"   Row count: Unable to retrieve ({e})")
        else:
            logger.warning(f"\n Table not found: {table}")


def test_basic_queries(db: OracleConnection):
    """Test basic queries to verify database access"""
    logger.info("\n" + "="*60)
    logger.info("BASIC QUERY TESTS")
    logger.info("="*60)
    
    test_queries = [
        {
            "name": "Check CIM_EQP_MST sample",
            "sql": "SELECT * FROM CIM_EQP_MST WHERE ROWNUM <= 5"
        },
        {
            "name": "Check EQP_MODE_GEN_HIST recent data",
            "sql": """
                SELECT * FROM EQP_MODE_GEN_HIST 
                WHERE ROWNUM <= 5 
                ORDER BY BASE_TIME DESC
            """
        }
    ]
    
    for test in test_queries:
        logger.info(f"\n {test['name']}")
        try:
            results = db.execute_query(test['sql'])
            logger.info(f"    Retrieved {len(results)} rows")
            if results:
                logger.info(f"   Sample: {results[0][:5]}...")  # Show first 5 columns
        except Exception as e:
            logger.error(f"    Failed: {e}")


def run_pipeline_on_question(question: Dict, db: OracleConnection):
    """
    Run the text-to-SQL pipeline on a single question
    
    This is where we integrate actual pipeline:
    1. Schema linking (FocusedSchemaBuilder)
    2. Variant generation (SchemaVariantGenerator)
    3. SQL generation (VotingOrchestrator)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Question [{question['id']}]: {question['question']}")
    logger.info(f"Category: {question['category']} | Difficulty: {question['difficulty']}")
    logger.info(f"{'='*60}")
    
    result = {
        "question_id": question['id'],
        "question": question['question'],
        "category": question['category'],
        "difficulty": question['difficulty'],
        "timestamp": datetime.now().isoformat(),
        "status": "NOT_IMPLEMENTED",
        "sql": None,
        "results": None,
        "error": None
    }
    
    # TODO: Will integrate my actual pipeline here
    # For now, this is a placeholder showing what needs to happen:
    
    logger.info("\n Pipeline steps (to be implemented):")
    logger.info("   1. [ ] Extract key entities from question")
    logger.info("   2. [ ] Build focused schema (FocusedSchemaBuilder)")
    logger.info("   3. [ ] Generate schema variants")
    logger.info("   4. [ ] Run VotingOrchestrator to generate SQL candidates")
    logger.info("   5. [ ] Execute best SQL candidate")
    logger.info("   6. [ ] Validate results")
    
    result['status'] = "PENDING_IMPLEMENTATION"
    return result


def main():
    """Main test runner"""
    logger.info("\n" + "#"*30)
    logger.info("ORACLE TEXT-TO-SQL PIPELINE TEST")
    logger.info("#"*30 + "\n")
    
    # Test both databases
    databases = [
        ("PRIMARY_DATABASE", PRIMARY_DATABASE_DSN, PRIMARY_DATABASE_USER, PRIMARY_DATABASE_PASS),
        ("LINKED_DATABASE", LINKED_DATABASE_DSN, LINKED_DATABASE_USER, LINKED_DATABASE_PASS)
    ]
    
    for db_name, dsn, user, password in databases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {db_name} Database")
        logger.info(f"{'='*60}")
        
        db = OracleConnection(dsn, user, password)
        
        try:
            # Step 1: Connect to database
            if not db.connect():
                logger.error(f" Cannot connect to {db_name}, skipping...")
                continue
            
            # Step 2: Inspect database structure
            inspect_database(db)
            
            # Step 3: Run basic query tests
            test_basic_queries(db)
            
        except Exception as e:
            logger.error(f"\n {db_name} test failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            db.close()
    
    # Step 4: Test Korean questions structure (PRIMARY_DATABASE only)
    logger.info("\n" + "="*60)
    logger.info("KOREAN TEST QUESTIONS PREVIEW")
    logger.info("="*60)
    logger.info(f"\nTotal questions prepared: {len(TEST_QUESTIONS)}")
    for q in TEST_QUESTIONS:
        logger.info(f"\n[{q['id']}] {q['question']}")
        logger.info(f"   Expected tables: {', '.join(q['expected_tables'])}")
        logger.info(f"   Difficulty: {q['difficulty']}")
    
    logger.info("\n" + "="*60)
    logger.info(" Database connectivity tests complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()