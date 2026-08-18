import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlglot
from sqlglot.errors import ParseError

# --- 1. The Suspect Function ---
def _extract_sql_from_response(response: str) -> str:
    """Extract SQL from LLM response (Current Implementation)."""
    # Remove markdown code blocks
    sql = response.strip()
    
    # Check if empty before processing
    if not sql:
        return sql
    
    if "```sql" in sql:
        # Extract from ```sql ... ```
        start = sql.find("```sql") + 6
        end = sql.find("```", start)
        if end > start:
            sql = sql[start:end].strip()
    elif "```" in sql:
        # Extract from ``` ... ```
        start = sql.find("```") + 3
        end = sql.find("```", start)
        if end > start:
            sql = sql[start:end].strip()
    
    # Remove common prefixes
    for prefix in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]:
        if prefix in sql.upper():
            idx = sql.upper().find(prefix)
            sql = sql[idx:]
            break
    
    return sql.strip()

# --- 2. The Validation Logic ---
def validate_sql(sql: str) -> str:
    try:
        sqlglot.parse_one(sql)
        return " Valid"
    except ParseError as e:
        return f" Invalid: {str(e)[:50]}..."

# --- 3. Test Runner ---
def run_tests(test_cases):
    print(f"{'TEST CASE':<10} | {'SOURCE':<15} | {'STATUS':<10} | {'SQL PREVIEW (Truncated)'}")
    print("-" * 80)

    for i, case in enumerate(test_cases):
        raw = case["raw"]
        manual = case["manual"] # What it SHOULD be
        extracted = _extract_sql_from_response(raw) # What function does

        # Test 1: Manual Extraction (The Truth)
        manual_status = validate_sql(manual)
        print(f"Case {i+1:<10} | Manual (Truth)  | {manual_status:<10} | {manual[:40].replace(chr(10), ' ')}...")

        # Test 2: Function's Extraction
        func_status = validate_sql(extracted)
        print(f"           | Function   | {func_status:<10} | {extracted[:40].replace(chr(10), ' ')}...")

        # Comparison
        if manual_status.startswith("") and func_status.startswith(""):
            print(f" MATCH FAILURE: Function corrupted valid SQL!")
            print(f"   Raw Input snippet: {raw[:50]}...")
            print(f"   Extraction:   {extracted}")
        print("-" * 80)
        
        if manual != extracted:
            print(f"      EXTRACTION MISMATCH:")
            print(f"      Expected length: {len(manual)} chars")
            print(f"      Got length:      {len(extracted)} chars")
            print(f"      Lost:            {len(manual) - len(extracted)} chars")
            
            # Show what was cut
            if len(extracted) < len(manual):
                print(f"      Missing prefix: '{manual[:50]}'")
                print(f"      Got instead:    '{extracted[:50]}'")

# --- 4. DATA ENTRY ---
TEST_CASES = [
    {
        "raw": "```sql\nWITH last_week AS (\n    SELECT\n        h.EQP_ID,\n        SUM(CASE WHEN h.STATUS = 'RUN' THEN 1 ELSE 0 END) AS run_count,\n        COUNT(*) AS total_count\n    FROM EQP_MODE_GEN_HIST h\n    JOIN CIM_EQP_MST m ON h.EQP_ID = m.EQP_ID\n    WHERE m.CIM_EQP_DESC LIKE '%형압반%'\n      AND h.CREATE_TIME >= TO_CHAR(TRUNC(SYSDATE - 7), 'YYYYMMDD') || ' 000000000'\n      AND h.CREATE_TIME < TO_CHAR(TRUNC(SYSDATE), 'YYYYMMDD') || ' 000000000'\n    GROUP BY h.EQP_ID\n),\nthis_week AS (\n    SELECT\n        h.EQP_ID,\n        SUM(CASE WHEN h.STATUS = 'RUN' THEN 1 ELSE 0 END) AS run_count,\n        COUNT(*) AS total_count\n    FROM EQP_MODE_GEN_HIST h\n    JOIN CIM_EQP_MST m ON h.EQP_ID = m.EQP_ID\n    WHERE m.CIM_EQP_DESC LIKE '%형압반%'\n      AND h.CREATE_TIME >= TO_CHAR(TRUNC(SYSDATE), 'YYYYMMDD') || ' 000000000'\n      AND h.CREATE_TIME < TO_CHAR(TRUNC(SYSDATE + 1), 'YYYYMMDD') || ' 000000000'\n    GROUP BY h.EQP_ID\n)\nSELECT\n    m.CIM_EQP_DESC,\n    l.EQP_ID,\n    ROUND(NVL(t.run_count, 0) / NULLIF(t.total_count, 0) * 100, 2) AS this_week_rate,\n    ROUND(NVL(l.run_count, 0) / NULLIF(l.total_count, 0) * 100, 2) AS last_week_rate\nFROM last_week l\nJOIN this_week t ON l.EQP_ID = t.EQP_ID\nJOIN CIM_EQP_MST m ON l.EQP_ID = m.EQP_ID\nWHERE \n    ROUND(NVL(t.run_count, 0) / NULLIF(t.total_count, 0) * 100, 2) < ROUND(NVL(l.run_count, 0) / NULLIF(l.total_count, 0) * 100, 2)\nORDER BY last_week_rate - this_week_rate DESC;\n```",
        "manual": """WITH last_week AS (
    SELECT
        h.EQP_ID,
        SUM(CASE WHEN h.STATUS = 'RUN' THEN 1 ELSE 0 END) AS run_count,
        COUNT(*) AS total_count
    FROM EQP_MODE_GEN_HIST h
    JOIN CIM_EQP_MST m ON h.EQP_ID = m.EQP_ID
    WHERE m.CIM_EQP_DESC LIKE '%형압반%'
      AND h.CREATE_TIME >= TO_CHAR(TRUNC(SYSDATE - 7), 'YYYYMMDD') || ' 000000000'
      AND h.CREATE_TIME < TO_CHAR(TRUNC(SYSDATE), 'YYYYMMDD') || ' 000000000'
    GROUP BY h.EQP_ID
),
this_week AS (
    SELECT
        h.EQP_ID,
        SUM(CASE WHEN h.STATUS = 'RUN' THEN 1 ELSE 0 END) AS run_count,
        COUNT(*) AS total_count
    FROM EQP_MODE_GEN_HIST h
    JOIN CIM_EQP_MST m ON h.EQP_ID = m.EQP_ID
    WHERE m.CIM_EQP_DESC LIKE '%형압반%'
      AND h.CREATE_TIME >= TO_CHAR(TRUNC(SYSDATE), 'YYYYMMDD') || ' 000000000'
      AND h.CREATE_TIME < TO_CHAR(TRUNC(SYSDATE + 1), 'YYYYMMDD') || ' 000000000'
    GROUP BY h.EQP_ID
)
SELECT
    m.CIM_EQP_DESC,
    l.EQP_ID,
    ROUND(NVL(t.run_count, 0) / NULLIF(t.total_count, 0) * 100, 2) AS this_week_rate,
    ROUND(NVL(l.run_count, 0) / NULLIF(l.total_count, 0) * 100, 2) AS last_week_rate
FROM last_week l
JOIN this_week t ON l.EQP_ID = t.EQP_ID
JOIN CIM_EQP_MST m ON l.EQP_ID = m.EQP_ID
WHERE 
    ROUND(NVL(t.run_count, 0) / NULLIF(t.total_count, 0) * 100, 2) < ROUND(NVL(l.run_count, 0) / NULLIF(l.total_count, 0) * 100, 2)
ORDER BY last_week_rate - this_week_rate DESC;"""
    },
    {
        "raw": "```sql\nWITH last_week AS (\n    SELECT\n        h.EQP_ID,\n        SUM(CASE WHEN h.STATUS = 'RUN' THEN 1 ELSE 0 END) AS run_count,\n        COUNT(*) AS total_count\n    FROM EQP_MODE_GEN_HIST h\n    JOIN CIM_EQP_MST m ON h.EQP_ID = m.EQP_ID\n    WHERE m.CIM_EQP_DESC LIKE '%형압반%'\n      AND h.CREATE_TIME >= TO_CHAR(TRUNC(SYSDATE - 7), 'YYYYMMDD') || ' 000000000'\n      AND h.CREATE_TIME < TO_CHAR(TRUNC(SYSDATE), 'YYYYMMDD') || ' 000000000'\n    GROUP BY h.EQP_ID\n),\nthis_week AS (\n    SELECT\n        h.EQP_ID,\n        SUM(CASE WHEN h.STATUS = 'RUN' THEN 1 ELSE 0 END) AS run_count,\n        COUNT(*) AS total_count\n    FROM EQP_MODE_GEN_HIST h\n    JOIN CIM_EQP_MST m ON h.EQP_ID = m.EQP_ID\n    WHERE m.CIM_EQP_DESC LIKE '%형압반%'\n      AND h.CREATE_TIME >= TO_CHAR(TRUNC(SYSDATE), 'YYYYMMDD') || ' 000000000'\n      AND h.CREATE_TIME < TO_CHAR(TRUNC(SYSDATE + 1), 'YYYYMMDD') || ' 000000000'\n    GROUP BY h.EQP_ID\n)\nSELECT\n    m.CIM_EQP_DESC,\n    l.EQP_ID,\n    ROUND(NVL(t.run_count, 0) / NULLIF(t.total_count, 0) * 100, 2) AS this_week_rate,\n    ROUND(NVL(l.run_count, 0) / NULLIF(l.total_count, 0) * 100, 2) AS last_week_rate\nFROM last_week l\nJOIN this_week t ON l.EQP_ID = t.EQP_ID\nJOIN CIM_EQP_MST m ON l.EQP_ID = m.EQP_ID\nWHERE \n    ROUND(NVL(t.run_count, 0) / NULLIF(t.total_count, 0) * 100, 2) < ROUND(NVL(l.run_count, 0) / NULLIF(l.total_count, 0) * 100, 2)\nORDER BY (ROUND(NVL(l.run_count, 0) / NULLIF(l.total_count, 0) * 100, 2) - ROUND(NVL(t.run_count, 0) / NULLIF(t.total_count, 0) * 100, 2)) DESC;\n```",
        "manual": """WITH last_week AS (
    SELECT
        h.EQP_ID,
        SUM(CASE WHEN h.STATUS = 'RUN' THEN 1 ELSE 0 END) AS run_count,
        COUNT(*) AS total_count
    FROM EQP_MODE_GEN_HIST h
    JOIN CIM_EQP_MST m ON h.EQP_ID = m.EQP_ID
    WHERE m.CIM_EQP_DESC LIKE '%형압반%'
      AND h.CREATE_TIME >= TO_CHAR(TRUNC(SYSDATE - 7), 'YYYYMMDD') || ' 000000000'
      AND h.CREATE_TIME < TO_CHAR(TRUNC(SYSDATE), 'YYYYMMDD') || ' 000000000'
    GROUP BY h.EQP_ID
),
this_week AS (
    SELECT
        h.EQP_ID,
        SUM(CASE WHEN h.STATUS = 'RUN' THEN 1 ELSE 0 END) AS run_count,
        COUNT(*) AS total_count
    FROM EQP_MODE_GEN_HIST h
    JOIN CIM_EQP_MST m ON h.EQP_ID = m.EQP_ID
    WHERE m.CIM_EQP_DESC LIKE '%형압반%'
      AND h.CREATE_TIME >= TO_CHAR(TRUNC(SYSDATE), 'YYYYMMDD') || ' 000000000'
      AND h.CREATE_TIME < TO_CHAR(TRUNC(SYSDATE + 1), 'YYYYMMDD') || ' 000000000'
    GROUP BY h.EQP_ID
)
SELECT
    m.CIM_EQP_DESC,
    l.EQP_ID,
    ROUND(NVL(t.run_count, 0) / NULLIF(t.total_count, 0) * 100, 2) AS this_week_rate,
    ROUND(NVL(l.run_count, 0) / NULLIF(l.total_count, 0) * 100, 2) AS last_week_rate
FROM last_week l
JOIN this_week t ON l.EQP_ID = t.EQP_ID
JOIN CIM_EQP_MST m ON l.EQP_ID = m.EQP_ID
WHERE 
    ROUND(NVL(t.run_count, 0) / NULLIF(t.total_count, 0) * 100, 2) < ROUND(NVL(l.run_count, 0) / NULLIF(l.total_count, 0) * 100, 2)
ORDER BY (ROUND(NVL(l.run_count, 0) / NULLIF(l.total_count, 0) * 100, 2) - ROUND(NVL(t.run_count, 0) / NULLIF(t.total_count, 0) * 100, 2)) DESC;"""
    },
    {
        "raw": "```sql\nWITH last_week AS (\n    SELECT\n        h.EQP_ID,\n        SUM(CASE WHEN h.STATUS = 'RUN' THEN 1 ELSE 0 END) AS RUN_COUNT,\n        COUNT(*) AS TOTAL_COUNT\n    FROM EQP_MODE_GEN_HIST h\n    JOIN CIM_EQP_MST m\n        ON h.EQP_ID = m.EQP_ID\n    WHERE m.CIM_EQP_DESC LIKE '%형압반%'\n      AND h.CREATE_TIME BETWEEN TO_CHAR(TRUNC(SYSDATE, 'IW') - 7, 'YYYYMMDD') || '000000000'\n                            AND TO_CHAR(TRUNC(SYSDATE, 'IW') - 1, 'YYYYMMDD') || '235959999'\n    GROUP BY h.EQP_ID\n),\nthis_week AS (\n    SELECT\n        h.EQP_ID,\n        SUM(CASE WHEN h.STATUS = 'RUN' THEN 1 ELSE 0 END) AS RUN_COUNT,\n        COUNT(*) AS TOTAL_COUNT\n    FROM EQP_MODE_GEN_HIST h\n    JOIN CIM_EQP_MST m\n        ON h.EQP_ID = m.EQP_ID\n    WHERE m.CIM_EQP_DESC LIKE '%형압반%'\n      AND h.CREATE_TIME BETWEEN TO_CHAR(TRUNC(SYSDATE, 'IW'), 'YYYYMMDD') || '000000000'\n                            AND TO_CHAR(TRUNC(SYSDATE, 'IW') + 6, 'YYYYMMDD') || '235959999'\n    GROUP BY h.EQP_ID\n)\nSELECT\n    m.CIM_EQP_DESC,\n    t.EQP_ID,\n    ROUND(NVL(t.RUN_COUNT,0)/NULLIF(t.TOTAL_COUNT,0)*100, 2) AS THIS_WEEK_RATE,\n    ROUND(NVL(l.RUN_COUNT,0)/NULLIF(l.TOTAL_COUNT,0)*100, 2) AS LAST_WEEK_RATE\nFROM this_week t\nJOIN last_week l ON t.EQP_ID = l.EQP_ID\nJOIN CIM_EQP_MST m ON t.EQP_ID = m.EQP_ID\nWHERE NVL(t.RUN_COUNT,0)/NULLIF(t.TOTAL_COUNT,0) < NVL(l.RUN_COUNT,0)/NULLIF(l.TOTAL_COUNT,0)\nORDER BY THIS_WEEK_RATE ASC;\n```",
        "manual": """WITH last_week AS (
    SELECT
        h.EQP_ID,
        SUM(CASE WHEN h.STATUS = 'RUN' THEN 1 ELSE 0 END) AS RUN_COUNT,
        COUNT(*) AS TOTAL_COUNT
    FROM EQP_MODE_GEN_HIST h
    JOIN CIM_EQP_MST m
        ON h.EQP_ID = m.EQP_ID
    WHERE m.CIM_EQP_DESC LIKE '%형압반%'
      AND h.CREATE_TIME BETWEEN TO_CHAR(TRUNC(SYSDATE, 'IW') - 7, 'YYYYMMDD') || '000000000'
                            AND TO_CHAR(TRUNC(SYSDATE, 'IW') - 1, 'YYYYMMDD') || '235959999'
    GROUP BY h.EQP_ID
),
this_week AS (
    SELECT
        h.EQP_ID,
        SUM(CASE WHEN h.STATUS = 'RUN' THEN 1 ELSE 0 END) AS RUN_COUNT,
        COUNT(*) AS TOTAL_COUNT
    FROM EQP_MODE_GEN_HIST h
    JOIN CIM_EQP_MST m
        ON h.EQP_ID = m.EQP_ID
    WHERE m.CIM_EQP_DESC LIKE '%형압반%'
      AND h.CREATE_TIME BETWEEN TO_CHAR(TRUNC(SYSDATE, 'IW'), 'YYYYMMDD') || '000000000'
                            AND TO_CHAR(TRUNC(SYSDATE, 'IW') + 6, 'YYYYMMDD') || '235959999'
    GROUP BY h.EQP_ID
)
SELECT
    m.CIM_EQP_DESC,
    t.EQP_ID,
    ROUND(NVL(t.RUN_COUNT,0)/NULLIF(t.TOTAL_COUNT,0)*100, 2) AS THIS_WEEK_RATE,
    ROUND(NVL(l.RUN_COUNT,0)/NULLIF(l.TOTAL_COUNT,0)*100, 2) AS LAST_WEEK_RATE
FROM this_week t
JOIN last_week l ON t.EQP_ID = l.EQP_ID
JOIN CIM_EQP_MST m ON t.EQP_ID = m.EQP_ID
WHERE NVL(t.RUN_COUNT,0)/NULLIF(t.TOTAL_COUNT,0) < NVL(l.RUN_COUNT,0)/NULLIF(l.TOTAL_COUNT,0)
ORDER BY THIS_WEEK_RATE ASC;"""
    },
    {
        "raw": "```sql\nWITH eqp_map AS (\n    SELECT \n        MST.EQP_ID,\n        MST.CIM_EQP_DESC\n    FROM CIM_EQP_MST MST\n    WHERE MST.CIM_EQP_DESC LIKE '%형압반%' -- 형압반 설비만\n        AND MST.USE_YN = 'Y'\n),\nhist AS (\n    SELECT \n        H.EQP_ID,\n        TO_CHAR(TO_DATE(SUBSTR(H.CREATE_TIME, 1, 8), 'YYYYMMDD'), 'IW') AS ISO_WEEK,\n        TO_CHAR(TO_DATE(SUBSTR(H.CREATE_TIME, 1, 8), 'YYYYMMDD'), 'YYYY') AS YEAR,\n        SUM(CASE WHEN H.STATUS = 'RUN' THEN 1 ELSE 0 END) AS RUN_CNT,\n        COUNT(*) AS TOTAL_CNT\n    FROM EQP_MODE_GEN_HIST H\n    WHERE H.EQP_ID IN (SELECT EQP_ID FROM eqp_map)\n        AND TO_DATE(SUBSTR(H.CREATE_TIME, 1, 8), 'YYYYMMDD') >= TRUNC(SYSDATE, 'IW') - 14\n    GROUP BY H.EQP_ID,\n        TO_CHAR(TO_DATE(SUBSTR(H.CREATE_TIME, 1, 8), 'YYYYMMDD'), 'IW'),\n        TO_CHAR(TO_DATE(SUBSTR(H.CREATE_TIME, 1, 8), 'YYYYMMDD'), 'YYYY')\n),\nagg AS (\n    SELECT\n        h.EQP_ID,\n        MAX(CASE WHEN (h.YEAR || h.ISO_WEEK) = TO_CHAR(TRUNC(SYSDATE, 'IW'), 'YYYYIW') THEN \n            CASE WHEN TOTAL_CNT = 0 THEN 0 ELSE RUN_CNT*1.0/TOTAL_CNT END \n        END) AS LAST_WEEK_RATE,\n        MAX(CASE WHEN (h.YEAR || h.ISO_WEEK) = TO_CHAR(TRUNC(SYSDATE, 'IW') - 7, 'YYYYIW') THEN \n            CASE WHEN TOTAL_CNT = 0 THEN 0 ELSE RUN_CNT*1.0/TOTAL_CNT END \n        END) AS PREV_WEEK_RATE\n    FROM hist h\n    GROUP BY h.EQP_ID\n)\nSELECT\n    e.EQP_ID,\n    e.CIM_EQP_DESC,\n    a.PREV_WEEK_RATE AS PREVIOUS_WEEK_RATE,\n    a.LAST_WEEK_RATE AS LAST_WEEK_RATE\nFROM agg a\nJOIN eqp_map e ON a.EQP_ID = e.EQP_ID\nWHERE \n    a.PREV_WEEK_RATE > a.LAST_WEEK_RATE\n    AND a.PREV_WEEK_RATE IS NOT NULL\n    AND a.LAST_WEEK_RATE IS NOT NULL\nORDER BY a.PREV_WEEK_RATE - a.LAST_WEEK_RATE DESC;\n```",
        "manual": """WITH eqp_map AS (
    SELECT 
        MST.EQP_ID,
        MST.CIM_EQP_DESC
    FROM CIM_EQP_MST MST
    WHERE MST.CIM_EQP_DESC LIKE '%형압반%' -- 형압반 설비만
        AND MST.USE_YN = 'Y'
),
hist AS (
    SELECT 
        H.EQP_ID,
        TO_CHAR(TO_DATE(SUBSTR(H.CREATE_TIME, 1, 8), 'YYYYMMDD'), 'IW') AS ISO_WEEK,
        TO_CHAR(TO_DATE(SUBSTR(H.CREATE_TIME, 1, 8), 'YYYYMMDD'), 'YYYY') AS YEAR,
        SUM(CASE WHEN H.STATUS = 'RUN' THEN 1 ELSE 0 END) AS RUN_CNT,
        COUNT(*) AS TOTAL_CNT
    FROM EQP_MODE_GEN_HIST H
    WHERE H.EQP_ID IN (SELECT EQP_ID FROM eqp_map)
        AND TO_DATE(SUBSTR(H.CREATE_TIME, 1, 8), 'YYYYMMDD') >= TRUNC(SYSDATE, 'IW') - 14
    GROUP BY H.EQP_ID,
        TO_CHAR(TO_DATE(SUBSTR(H.CREATE_TIME, 1, 8), 'YYYYMMDD'), 'IW'),
        TO_CHAR(TO_DATE(SUBSTR(H.CREATE_TIME, 1, 8), 'YYYYMMDD'), 'YYYY')
),
agg AS (
    SELECT
        h.EQP_ID,
        MAX(CASE WHEN (h.YEAR || h.ISO_WEEK) = TO_CHAR(TRUNC(SYSDATE, 'IW'), 'YYYYIW') THEN 
            CASE WHEN TOTAL_CNT = 0 THEN 0 ELSE RUN_CNT*1.0/TOTAL_CNT END 
        END) AS LAST_WEEK_RATE,
        MAX(CASE WHEN (h.YEAR || h.ISO_WEEK) = TO_CHAR(TRUNC(SYSDATE, 'IW') - 7, 'YYYYIW') THEN 
            CASE WHEN TOTAL_CNT = 0 THEN 0 ELSE RUN_CNT*1.0/TOTAL_CNT END 
        END) AS PREV_WEEK_RATE
    FROM hist h
    GROUP BY h.EQP_ID
)
SELECT
    e.EQP_ID,
    e.CIM_EQP_DESC,
    a.PREV_WEEK_RATE AS PREVIOUS_WEEK_RATE,
    a.LAST_WEEK_RATE AS LAST_WEEK_RATE
FROM agg a
JOIN eqp_map e ON a.EQP_ID = e.EQP_ID
WHERE 
    a.PREV_WEEK_RATE > a.LAST_WEEK_RATE
    AND a.PREV_WEEK_RATE IS NOT NULL
    AND a.LAST_WEEK_RATE IS NOT NULL
ORDER BY a.PREV_WEEK_RATE - a.LAST_WEEK_RATE DESC;"""
    },
    {
        "raw": "```sql\nWITH recent_weeks AS (\n    SELECT \n        TO_CHAR(TRUNC(TO_DATE(SUBSTR(EMGH.CREATE_TIME,1,8),'YYYYMMDD'),'IW'),'YYYYMMDD') AS WEEK_START,\n        EMGH.EQP_ID,\n        SUM(CASE WHEN EMGH.STATUS = 'RUN' THEN 1 ELSE 0 END) AS RUN_COUNT,\n        COUNT(*) AS TOTAL_COUNT\n    FROM EQP_MODE_GEN_HIST EMGH\n    JOIN CIM_EQP_MST CEM ON EMGH.EQP_ID = CEM.EQP_ID\n    WHERE CEM.CIM_EQP_DESC LIKE '%형압반%'\n      AND EMGH.CREATE_TIME >= TO_CHAR(TRUNC(SYSDATE-14,'IW'),'YYYYMMDD')\n    GROUP BY TO_CHAR(TRUNC(TO_DATE(SUBSTR(EMGH.CREATE_TIME,1,8),'YYYYMMDD'),'IW'),'YYYYMMDD'), EMGH.EQP_ID\n),\nagg AS (\n    SELECT\n        EQP_ID,\n        MAX(CASE WHEN WEEK_START = TO_CHAR(TRUNC(SYSDATE-7,'IW'),'YYYYMMDD') THEN RUN_COUNT / NULLIF(TOTAL_COUNT,0) END) AS LAST_WEEK_RATE,\n        MAX(CASE WHEN WEEK_START = TO_CHAR(TRUNC(SYSDATE,'IW'),'YYYYMMDD') THEN RUN_COUNT / NULLIF(TOTAL_COUNT,0) END) AS THIS_WEEK_RATE\n    FROM recent_weeks\n    GROUP BY EQP_ID\n)\nSELECT CEM.CIM_EQP_DESC, agg.LAST_WEEK_RATE, agg.THIS_WEEK_RATE\nFROM agg\nJOIN CIM_EQP_MST CEM ON agg.EQP_ID = CEM.EQP_ID\nWHERE agg.LAST_WEEK_RATE > NVL(agg.THIS_WEEK_RATE, 0)\n  AND CEM.CIM_EQP_DESC LIKE '%형압반%'\nORDER BY CEM.CIM_EQP_DESC;\n```",
        "manual": """WITH recent_weeks AS (
    SELECT 
        TO_CHAR(TRUNC(TO_DATE(SUBSTR(EMGH.CREATE_TIME,1,8),'YYYYMMDD'),'IW'),'YYYYMMDD') AS WEEK_START,
        EMGH.EQP_ID,
        SUM(CASE WHEN EMGH.STATUS = 'RUN' THEN 1 ELSE 0 END) AS RUN_COUNT,
        COUNT(*) AS TOTAL_COUNT
    FROM EQP_MODE_GEN_HIST EMGH
    JOIN CIM_EQP_MST CEM ON EMGH.EQP_ID = CEM.EQP_ID
    WHERE CEM.CIM_EQP_DESC LIKE '%형압반%'
      AND EMGH.CREATE_TIME >= TO_CHAR(TRUNC(SYSDATE-14,'IW'),'YYYYMMDD')
    GROUP BY TO_CHAR(TRUNC(TO_DATE(SUBSTR(EMGH.CREATE_TIME,1,8),'YYYYMMDD'),'IW'),'YYYYMMDD'), EMGH.EQP_ID
),
agg AS (
    SELECT
        EQP_ID,
        MAX(CASE WHEN WEEK_START = TO_CHAR(TRUNC(SYSDATE-7,'IW'),'YYYYMMDD') THEN RUN_COUNT / NULLIF(TOTAL_COUNT,0) END) AS LAST_WEEK_RATE,
        MAX(CASE WHEN WEEK_START = TO_CHAR(TRUNC(SYSDATE,'IW'),'YYYYMMDD') THEN RUN_COUNT / NULLIF(TOTAL_COUNT,0) END) AS THIS_WEEK_RATE
    FROM recent_weeks
    GROUP BY EQP_ID
)
SELECT CEM.CIM_EQP_DESC, agg.LAST_WEEK_RATE, agg.THIS_WEEK_RATE
FROM agg
JOIN CIM_EQP_MST CEM ON agg.EQP_ID = CEM.EQP_ID
WHERE agg.LAST_WEEK_RATE > NVL(agg.THIS_WEEK_RATE, 0)
  AND CEM.CIM_EQP_DESC LIKE '%형압반%'
ORDER BY CEM.CIM_EQP_DESC;"""
    }
]

if __name__ == "__main__":
    run_tests(TEST_CASES)