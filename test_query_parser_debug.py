from query.parser import SQLQueryParser

def test_specific_issues():
    """Test specific problematic queries."""
    parser = SQLQueryParser()
    
    print("=== DEBUGGING SPECIFIC ISSUES ===\n")
    
    # Test 1: SELECT clause parsing
    print("1. Testing SELECT clause parsing")
    query1 = "SELECT customer_name, customer_id FROM customer"
    result1 = parser.parse(query1)
    print(f"Select columns: {result1['select_columns']}")
    print()
    
    # Test 2: Aggregate detection
    print("2. Testing aggregate detection")
    query2 = "SELECT COUNT(*), SUM(revenue) as total_rev FROM orders"
    result2 = parser.parse(query2)
    print(f"Aggregates: {result2['aggregates']}")
    print(f"Select columns: {result2['select_columns']}")
    print()
    
    # Test 3: GROUP BY parsing
    print("3. Testing GROUP BY parsing")
    query3 = "SELECT region, COUNT(*) FROM customer GROUP BY region"
    result3 = parser.parse(query3)
    print(f"Query type: {result3['query_type']}")
    print(f"Aggregates: {result3['aggregates']}")
    print(f"Group By: {result3['group_by_columns']}")
    print()
    
    # Test 4: Implicit join
    print("4. Testing implicit join")
    query4 = "SELECT d_year FROM lineorder, date_dim WHERE lo_orderdate = d_datekey"
    result4 = parser.parse(query4)
    print(f"Tables: {result4['tables']}")
    print(f"Join conditions: {result4['join_conditions']}")
    print(f"Where conditions: {result4['where_conditions']}")
    print()
    
    # Test 5: Full SSB query
    print("5. Testing full SSB query")
    query5 = """
    SELECT d_year, SUM(lo_extendedprice * lo_discount) AS revenue
    FROM lineorder, date_dim
    WHERE lo_orderdate = d_datekey
    AND d_year >= 1993 AND d_year <= 1997
    GROUP BY d_year
    ORDER BY d_year
    """
    result5 = parser.parse(query5)
    print(f"Query type: {result5['query_type']}")
    print(f"Tables: {result5['tables']}")  
    print(f"Aggregates: {result5['aggregates']}")
    print(f"Group By: {result5['group_by_columns']}")
    print(f"Join conditions: {result5['join_conditions']}")

if __name__ == "__main__":
    test_specific_issues()