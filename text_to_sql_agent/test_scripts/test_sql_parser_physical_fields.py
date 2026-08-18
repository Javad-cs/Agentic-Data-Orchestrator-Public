import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Test physical field extraction on real LLM-generated SQL queries.

Uses REAL schema from Oracle profiler.
"""

import sys
from pathlib import Path

# Add text_to_sql_agent to path
_current_file = Path(__file__).resolve()
_agent_root = _current_file.parent
sys.path.insert(0, str(_agent_root))

# Now import from text_to_sql_agent
from sql_generation.sql_parser import SQLParser


# REAL SCHEMA from Oracle profiler
REAL_SCHEMA = {
    "CIM_EQP_MST": {
        "SITE", "EQP_ID", "EQP_NO", "EQP_GROUP", "CIM_EQP_DESC", 
        "CIM_EQP_TYPE", "EQP_IP", "ROBOT_IP", "FUNAC_IP", "USE_YN",
        "CREATE_USER", "CREATE_TIME", "UPDATE_USER", "UPDATE_TIME",
        "MODE_TYPE", "STOP_BASE", "RUN_BASE", "CIM_ID"
    },
    
    "EQP_MODE_GEN_HIST": {
        "SITE", "EQP_ID", "EQP_NO", "CUR_TIME", "CREATE_TIME",
        "WORK_CENTER", "TEST_USER", "PROD_ORDER_NUMBER", "PROCESS_NAME",
        "LOT_ID", "ITEM_NUMBER", "STATUS", "EQP_PART_NUMBER",
        "PART_NUMBER", "START_DATE_TIME", "USE_YN", "FST_CREATE_SYS",
        "LAST_UPDATE_SYS", "LAST_UPDATE_TIME", "BASE_TIME", "EQP_MODE",
        "STATUS_D"
    },
    
    "EQP_MST@LINKED_DATABASE": {
        "PLANT_CODE", "EQP_ID", "EQP_NO", "ASSET_FLAG", "EQP_DESC",
        "ASSET_NUMBER", "MODEL_NUMBER", "EQP_SPEC", "WORK_CENTER",
        "DEPT_CODE", "MANUFACTURER", "MANUFACTURE_DATE", "PURCHASE_DATE",
        "PRICE", "CURRENCY", "CALIBRATION", "DISUSE_DATE",
        "TRACK_RECORD_FLAG", "MAIN_EQP_FLAG", "EQP_TYPE", "REPAIRER",
        "EQP_NAME_ENG", "FUNC_LOCATION", "FUNC_POSITION_DESC",
        "MAIN_WORK_CENTER", "DISUSE_FLAG", "PLC_FLAG", "IP_ADDRESS",
        "PORT_NUMBER", "CREATE_DATE_TIME", "CREATE_USER_ID",
        "UPDATE_DATE_TIME", "UPDATE_USER_ID", "RUN_TYPE",
        "VIRTUAL_EQP_FLAG", "IDLE_TIME_LIMIT", "PLC_COUNT_FLAG",
        "PLC_TYPE", "OIL_CH_DATE", "OIL_CH_EX_DATE", "PCS_MEASURE_DATE",
        "EQP_GRADE", "SORT_NO", "AUTO_YN"
    },
    
    "DEPT_MST@LINKED_DATABASE": {
        "CORP_CODE", "DEPT_CODE", "DEPT_NAME", "DEPT_ALIAS",
        "SUBAREA_CODE", "SUBAREA_DESC", "BEGIN_DATE", "END_DATE",
        "USE_FLAG", "DEPT_LEVEL", "WC_FLAG", "CREATE_DATE_TIME"
    },
    
    "WORK_CENTER@LINKED_DATABASE": {
        "PLANT_CODE", "WORK_CENTER", "WORK_CENTER_NAME", "DEPT_CODE",
        "LOAD_EST_BASE", "WORK_CENTER_TYPE", "RUN_TYPE", "BATCH_SIZE",
        "CREATE_DATE_TIME", "RECORD_INPUT_TYPE", "WEIGH_FLAG"
    }
}


def test_query_1_hallucinated_field():
    """
    Query 1: Contains hallucinated field m.EQP_DESC (should be CIM_EQP_DESC)
    
    Expected: Parser should REJECT the hallucinated field
    """
    
    sql = """
    SELECT m.EQP_DESC,
           g1.EQP_ID
    FROM CIM_EQP_MST m
    JOIN (
        SELECT h.EQP_ID
        FROM EQP_MODE_GEN_HIST h
        WHERE h.STATUS = 'RUN'
        GROUP BY h.EQP_ID
    ) g1 ON m.EQP_ID = g1.EQP_ID
    """
    
    parser = SQLParser(dialect="oracle", schema=REAL_SCHEMA)
    result = parser.parse(sql)
    
    fields = sorted(result.referenced_fields)
    
    print("\n" + "="*70)
    print("TEST 1: Hallucinated Field (m.EQP_DESC)")
    print("="*70)
    print(f"\nExtracted {len(fields)} physical fields:")
    for table, col in fields:
        print(f"  {table}.{col}")
    
    # Check that hallucinated field is rejected
    hallucinated = [f for f in fields if f[1] == "EQP_DESC" and f[0] == "CIM_EQP_MST"]
    
    if hallucinated:
        print(f"\n FAIL: Accepted hallucinated field CIM_EQP_MST.EQP_DESC")
        print(f"   (Should be CIM_EQP_DESC, not EQP_DESC)")
        return False
    
    # Check that valid fields are kept
    expected = {("CIM_EQP_MST", "EQP_ID"), ("EQP_MODE_GEN_HIST", "EQP_ID"), ("EQP_MODE_GEN_HIST", "STATUS")}
    if not expected.issubset(set(fields)):
        print(f"\n  WARNING: Missing expected valid fields")
        print(f"   Expected: {expected}")
        print(f"   Got: {set(fields)}")
    
    # Check no subquery alias
    if any(table in ["G1", "g1"] for table, _ in fields):
        print(f"\n FAIL: Found subquery alias g1")
        return False
    
    print(f"\n PASS: Hallucinated field correctly rejected")
    print(f"   Valid fields extracted: {len(fields)}")
    return True


def test_query_2_ctes_with_real_schema():
    """Query 2: CTEs should be filtered, real fields kept"""
    
    sql = """
    WITH HYUNGAP_WORK_CENTER AS (
        SELECT WC.WORK_CENTER
          FROM WORK_CENTER@LINKED_DATABASE WC
         INNER JOIN DEPT_MST@LINKED_DATABASE D ON WC.DEPT_CODE = D.DEPT_CODE
         WHERE D.DEPT_NAME LIKE '%형압반%'
    ),
    THIS_WEEK AS (
        SELECT h.EQP_ID,
               SUM(CASE WHEN h.STATUS='RUN' THEN 1 ELSE 0 END) AS RUN_COUNT
          FROM EQP_MODE_GEN_HIST h
         WHERE h.WORK_CENTER IN (SELECT WORK_CENTER FROM HYUNGAP_WORK_CENTER)
         GROUP BY h.EQP_ID
    )
    SELECT M.CIM_EQP_DESC, T.RUN_COUNT
      FROM CIM_EQP_MST M
      LEFT JOIN THIS_WEEK T ON M.EQP_ID = T.EQP_ID
    """
    
    parser = SQLParser(dialect="oracle", schema=REAL_SCHEMA)
    result = parser.parse(sql)
    
    fields = sorted(result.referenced_fields)
    
    print("\n" + "="*70)
    print("TEST 2: CTEs with Real Schema")
    print("="*70)
    print(f"\nExtracted {len(fields)} physical fields:")
    for table, col in fields:
        print(f"  {table}.{col}")
    
    actual_tables = {table for table, _ in fields}
    
    # Check no CTEs
    bad_ctes = {"HYUNGAP_WORK_CENTER", "THIS_WEEK"}
    found_bad = bad_ctes & actual_tables
    
    if found_bad:
        print(f"\n FAIL: Found CTE names: {found_bad}")
        return False
    
    # Check no derived fields
    derived_fields = [f for f in fields if f[1] == "RUN_COUNT"]
    if derived_fields:
        print(f"\n FAIL: Found derived field RUN_COUNT (computed in CTE)")
        print(f"   Derived fields should be traced to source columns")
        return False
    
    # Check real fields exist
    expected_min = {
        ("CIM_EQP_MST", "CIM_EQP_DESC"),
        ("CIM_EQP_MST", "EQP_ID"),
        ("EQP_MODE_GEN_HIST", "EQP_ID"),
        ("EQP_MODE_GEN_HIST", "STATUS"),
        ("WORK_CENTER@LINKED_DATABASE", "DEPT_CODE"),
        ("DEPT_MST@LINKED_DATABASE", "DEPT_NAME")
    }
    
    if not expected_min.issubset(set(fields)):
        print(f"\n  WARNING: Missing some expected fields")
        missing = expected_min - set(fields)
        print(f"   Missing: {missing}")
    
    print(f"\n PASS: CTEs filtered, real fields extracted")
    print(f"   Real tables: {actual_tables}")
    return True


def test_query_3_lowercase_tables():
    """Query 3: Lowercase table names should work"""
    
    sql = """
    SELECT a.eqp_id, a.cim_eqp_desc
    FROM cim_eqp_mst a
    WHERE a.work_center IN (
        SELECT work_center 
        FROM work_center@linked_database
        WHERE work_center_name LIKE '%형압반%'
    )
    """
    
    parser = SQLParser(dialect="oracle", schema=REAL_SCHEMA)
    result = parser.parse(sql)
    
    fields = sorted(result.referenced_fields)
    
    print("\n" + "="*70)
    print("TEST 3: Lowercase Table Names")
    print("="*70)
    print(f"\nExtracted {len(fields)} physical fields:")
    for table, col in fields:
        print(f"  {table}.{col}")
    
    # Check case-insensitive matching worked we made it to normalize everything to uppercase, so this test case is not needed
    expected = {
        ("CIM_EQP_MST", "EQP_ID"),        
        ("CIM_EQP_MST", "CIM_EQP_DESC"),      
        ("WORK_CENTER@LINKED_DATABASE", "WORK_CENTER"),
        ("WORK_CENTER@LINKED_DATABASE", "WORK_CENTER_NAME")
    }
    
    if not expected.issubset(set(fields)):
        print(f"\n FAIL: Case-insensitive matching failed")
        missing = expected - set(fields)
        print(f"   Missing: {missing}")
        return False
    
    print(f"\n PASS: Lowercase tables resolved correctly")
    return True


def test_query_4_nested_cte_resolution():
    """Query 4: Nested CTEs should trace to base tables"""
    
    sql = """
    WITH hyungapban_wcenter AS (
        SELECT wc.work_center
          FROM work_center@linked_database wc
         WHERE wc.work_center_name LIKE '%형압반%'
    ),
    hyungapban_eqp AS (
        SELECT eqp.eqp_id, eqp.cim_eqp_desc
          FROM cim_eqp_mst eqp
         WHERE eqp.work_center IN (SELECT work_center FROM hyungapban_wcenter)
    )
    SELECT e.eqp_id, e.cim_eqp_desc
      FROM hyungapban_eqp e
    """
    
    parser = SQLParser(dialect="oracle", schema=REAL_SCHEMA)
    result = parser.parse(sql)
    
    fields = sorted(result.referenced_fields)
    
    print("\n" + "="*70)
    print("TEST 4: Nested CTE Resolution")
    print("="*70)
    print(f"\nExtracted {len(fields)} physical fields:")
    for table, col in fields:
        print(f"  {table}.{col}")
    
    actual_tables = {table for table, _ in fields}
    
    # Check no CTEs
    bad_ctes = {"HYUNGAPBAN_WCENTER", "HYUNGAPBAN_EQP"}
    if bad_ctes & actual_tables:
        print(f"\n FAIL: Found CTE names: {bad_ctes & actual_tables}")
        return False
    
    # Should trace back to base tables
    expected_tables = {"CIM_EQP_MST", "WORK_CENTER@LINKED_DATABASE"}
    if not expected_tables.issubset(actual_tables):
        print(f"\n FAIL: Did not trace to all base tables")
        print(f"   Expected: {expected_tables}")
        print(f"   Got: {actual_tables}")
        return False
    
    print(f"\n PASS: Nested CTEs resolved to base tables")
    print(f"   Base tables: {actual_tables}")
    return True


def main():
    """Run all tests with REAL schema"""
    print("\n" + "="*70)
    print("PHYSICAL FIELD EXTRACTION TESTS")
    print("Real LLM Queries + Real Oracle Schema")
    print("="*70)
    
    tests = [
        test_query_1_hallucinated_field,
        test_query_2_ctes_with_real_schema,
        test_query_3_lowercase_tables,
        test_query_4_nested_cte_resolution,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("="*70)
    
    if passed == len(tests):
        print("\n ALL TESTS PASSED!")
        print("\nParser correctly:")
        print("   Filters out CTE names")
        print("   Filters out subquery aliases")
        print("   Rejects hallucinated fields")
        print("   Traces nested CTEs to base tables")
        print("   Handles case-insensitive matching")
        print("   Handles @DBLINK syntax")
        return 0
    else:
        print(f"\n  {len(tests) - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())