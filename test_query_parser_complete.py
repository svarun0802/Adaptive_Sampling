from query.parser import SQLQueryParser
from testing.benchmarks import SSBBenchmark

def test_parser():
    """Test the SQL query parser with various query types."""
    parser = SQLQueryParser()
    
    print("Testing SQL Query Parser")
    print("=" * 50)
    
    # Test 1: Simple SELECT
    print("\n1. Testing Simple SELECT")
    simple_query = "SELECT customer_name, customer_id FROM customer WHERE c_region = 'ASIA'"
    result = parser.parse(simple_query)
    print(f"Query Type: {result['query_type']}")
    print(f"Tables: {result['tables']}")
    print(f"Select Columns: {result['select_columns']}")
    print(f"Where Conditions: {result['where_conditions']}")
    
    # Test 2: Simple JOIN
    print("\n2. Testing Simple JOIN")
    join_query = """
    SELECT lo_orderkey, c_name 
    FROM lineorder lo 
    JOIN customer c ON lo.lo_custkey = c.c_custkey
    """
    result = parser.parse(join_query)
    print(f"Query Type: {result['query_type']}")
    print(f"Tables: {result['tables']}")
    print(f"Table Aliases: {result['table_aliases']}")
    print(f"Join Conditions: {result['join_conditions']}")
    print(f"Formatted for Sampling: {parser.format_joins_for_sampling()}")
    
    # Test 3: Analytical Query
    print("\n3. Testing Analytical Query")
    analytical_query = """
    SELECT 
        c_region,
        COUNT(*) as order_count,
        SUM(lo_revenue) as total_revenue,
        AVG(lo_revenue) as avg_order_value
    FROM lineorder lo
    JOIN customer c ON lo.lo_custkey = c.c_custkey
    WHERE lo_orderdate >= 19940101
    GROUP BY c_region
    ORDER BY total_revenue DESC
    LIMIT 10
    """
    result = parser.parse(analytical_query)
    print(f"Query Type: {result['query_type']}")
    print(f"Tables: {result['tables']}")
    print(f"Aggregates: {result['aggregates']}")
    print(f"Group By: {result['group_by_columns']}")
    print(f"Order By: {result['order_by_columns']}")
    print(f"Limit: {result['limit']}")
    print(f"Complexity Score: {result['complexity_score']}")
    
    # Test 4: SSB Query 1.1 (Implicit Join)
    print("\n4. Testing SSB Query 1.1 (Implicit Join)")
    ssb_query = SSBBenchmark.get_query_1_1().strip()
    result = parser.parse(ssb_query)
    print(f"Query Type: {result['query_type']}")
    print(f"Tables: {result['tables']}")
    print(f"Join Conditions: {result['join_conditions']}")
    print(f"Aggregates: {result['aggregates']}")
    print(f"Group By: {result['group_by_columns']}")
    print(f"Fingerprint: {result['fingerprint']}")
    
    # Test 5: Complex Multi-table JOIN
    print("\n5. Testing Complex Multi-table JOIN")
    complex_query = """
    SELECT 
        d_year,
        c_nation,
        s_nation,
        p_category,
        SUM(lo_revenue) as revenue
    FROM 
        lineorder lo
    JOIN customer c ON lo.lo_custkey = c.c_custkey
    JOIN supplier s ON lo.lo_suppkey = s.s_suppkey  
    JOIN part p ON lo.lo_partkey = p.p_partkey
    JOIN date_dim d ON lo.lo_orderdate = d.d_datekey
    WHERE 
        c_region = 'AMERICA'
        AND s_region = 'AMERICA'
        AND p_mfgr IN ('MFGR#1', 'MFGR#2')
    GROUP BY 
        d_year, c_nation, s_nation, p_category
    ORDER BY 
        revenue DESC
    """
    result = parser.parse(complex_query)
    print(f"Query Type: {result['query_type']}")
    print(f"Tables: {result['tables']}")
    print(f"Table Aliases: {result['table_aliases']}")
    print(f"Join Conditions: {len(result['join_conditions'])} joins")
    print("Join details:")
    for i, join in enumerate(result['join_conditions']):
        print(f"  {i+1}. {join['left_table']}.{join['left_column']} = {join['right_table']}.{join['right_column']}")
    print(f"Aggregates: {result['aggregates']}")
    print(f"Group By: {result['group_by_columns']}")
    print(f"Formatted for Sampling: {parser.format_joins_for_sampling()}")
    print(f"Complexity Score: {result['complexity_score']}")
    
    # Test 6: Query with Multiple Aggregates
    print("\n6. Testing Multiple Aggregates")
    multi_agg_query = """
    SELECT 
        p_brand,
        COUNT(*) as order_count,
        SUM(lo_quantity) as total_quantity,
        AVG(lo_extendedprice) as avg_price,
        MIN(lo_orderdate) as first_order,
        MAX(lo_orderdate) as last_order
    FROM lineorder lo
    JOIN part p ON lo.lo_partkey = p.p_partkey
    WHERE p_category = 'MFGR#1'
    GROUP BY p_brand
    """
    result = parser.parse(multi_agg_query)
    print(f"Query Type: {result['query_type']}")
    print(f"Aggregates found: {len(result['aggregates'])}")
    for agg in result['aggregates']:
        print(f"  - {agg['function']}({agg['argument']}) AS {agg['alias']}")
    print(f"Group By: {result['group_by_columns']}")

def test_edge_cases():
    """Test edge cases and potential issues."""
    parser = SQLQueryParser()
    
    print("\n\n" + "=" * 50)
    print("TESTING EDGE CASES")
    print("=" * 50)
    
    # Test edge case 1: No alias in join
    print("\n1. JOIN without aliases")
    query1 = "SELECT lineorder.lo_orderkey FROM lineorder JOIN customer ON lineorder.lo_custkey = customer.c_custkey"
    result1 = parser.parse(query1)
    print(f"Tables: {result1['tables']}")
    print(f"Joins: {result1['join_conditions']}")
    
    # Test edge case 2: Complex WHERE with multiple join conditions
    print("\n2. Multiple implicit joins")
    query2 = """
    SELECT lo_orderkey 
    FROM lineorder, customer, supplier 
    WHERE lo_custkey = c_custkey 
    AND lo_suppkey = s_suppkey
    """
    result2 = parser.parse(query2)
    print(f"Tables: {result2['tables']}")
    print(f"Implicit joins found: {len(result2['join_conditions'])}")
    for join in result2['join_conditions']:
        print(f"  - {join['left_table']}.{join['left_column']} = {join['right_table']}.{join['right_column']}")

if __name__ == "__main__":
    test_parser()
    test_edge_cases()