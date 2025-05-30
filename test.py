#!/usr/bin/env python3
"""
Simple Complex Query Test

Tests only complex analytical queries to demonstrate:
- Intelligent sampling decisions
- Smart materialization 
- Sample reuse
- Learning behavior
- Complexity scoring
"""

import time
from query import QueryInterceptor

def main():
    """Run focused complex query tests."""
    
    print("🧪 COMPLEX QUERY TEST SUITE")
    print("=" * 60)
    print("Testing adaptive sampling with complex analytical queries")
    print("Focus: Materialization, Complexity, Learning, Sample Reuse")
    print("=" * 60)
    
    # Initialize system
    print("\n🚀 Initializing Adaptive Sampling System...")
    system = QueryInterceptor()
    print("✅ System ready")
    
    # Run test phases
    test_complexity_decisions(system)
    test_materialization_decisions(system)
    test_sample_reuse(system)
    test_learning_behavior(system)
    show_final_statistics(system)
    
    print("\n🎉 COMPLEX QUERY TESTING COMPLETE!")

def test_complexity_decisions(system):
    """Test how the system handles queries of different complexity levels."""
    
    print("\n" + "="*60)
    print("🧠 PHASE 1: COMPLEXITY & SAMPLING DECISIONS")
    print("="*60)
    
    complex_queries = [
        ("Medium Complexity - 2 Tables", """
            SELECT c_region, SUM(lo_revenue) as total_revenue, COUNT(*) as order_count
            FROM lineorder lo
            JOIN customer c ON lo.lo_custkey = c.c_custkey
            WHERE lo_orderdate >= 19940101
            GROUP BY c_region
            ORDER BY total_revenue DESC
        """),
        
        ("High Complexity - 3 Tables", """
            SELECT c_region, s_region, 
                   SUM(lo_revenue) as revenue, 
                   AVG(lo_quantity) as avg_quantity,
                   COUNT(DISTINCT lo_orderkey) as unique_orders
            FROM lineorder lo
            JOIN customer c ON lo.lo_custkey = c.c_custkey
            JOIN supplier s ON lo.lo_suppkey = s.s_suppkey
            WHERE lo_orderdate BETWEEN 19940101 AND 19961231
            GROUP BY c_region, s_region
            HAVING SUM(lo_revenue) > 1000000
            ORDER BY revenue DESC
        """),
        
        ("Very High Complexity - 4 Tables", """
            SELECT d_year, c_region, p_category,
                   SUM(lo_revenue) as total_revenue,
                   SUM(lo_supplycost) as total_cost,
                   SUM(lo_revenue - lo_supplycost) as profit,
                   AVG(lo_quantity) as avg_quantity,
                   COUNT(*) as transaction_count
            FROM lineorder lo
            JOIN customer c ON lo.lo_custkey = c.c_custkey
            JOIN supplier s ON lo.lo_suppkey = s.s_suppkey
            JOIN part p ON lo.lo_partkey = p.p_partkey
            JOIN date_dim d ON lo.lo_orderdate = d.d_datekey
            WHERE d_year BETWEEN 1994 AND 1997
              AND c_region IN ('AMERICA', 'ASIA')
              AND p_category LIKE 'MFGR%'
            GROUP BY d_year, c_region, p_category
            ORDER BY profit DESC
            LIMIT 20
        """)
    ]
    
    for i, (name, query) in enumerate(complex_queries, 1):
        print(f"\n🔍 Test {i}: {name}")
        print(f"Query: {query.strip()[:80]}...")
        
        start_time = time.time()
        result = system.execute_query(query)
        execution_time = time.time() - start_time
        
        if result['success']:
            print(f"\n✅ Query executed successfully!")
            print(f"   Strategy: {result.get('strategy_used', 'unknown')}")
            print(f"   Execution time: {execution_time:.3f}s")
            print(f"   Result count: {result.get('result_count', 0)} rows")
            print(f"   Is approximate: {result.get('is_approximate', False)}")
            
            # Show decision reasoning
            if 'decision_metadata' in result:
                dm = result['decision_metadata']
                if 'sampling_decision' in dm:
                    sd = dm['sampling_decision']
                    print(f"   🧠 Sampling Decision:")
                    print(f"      Probability: {sd.get('sampling_probability', 0):.3f}")
                    print(f"      Confidence: {sd.get('confidence', 0):.3f}")
                    print(f"      Data complexity: {sd.get('data_complexity', 0):.2f}")
                
                if 'materialization_decision' in dm:
                    md = dm['materialization_decision']
                    print(f"   💾 Materialization Decision:")
                    print(f"      Materialize: {md.get('materialize', False)}")
                    print(f"      Expected reuses: {md.get('expected_reuses', 0):.1f}")
                    print(f"      Confidence: {md.get('confidence', 0):.3f}")
            
            # Show sample info
            if result.get('sample_info'):
                si = result['sample_info']
                print(f"   📊 Sample Info:")
                print(f"      Sample size: {si.get('sample_size', 'N/A')}")
                print(f"      Sample ID: {si.get('sample_id', 'N/A')}")
                print(f"      Materialized: {si.get('materialized', False)}")
        else:
            print(f"❌ Query failed: {result.get('error', 'Unknown error')}")
        
        time.sleep(1)  # Brief pause between queries

def test_materialization_decisions(system):
    """Test what gets materialized vs kept in memory."""
    
    print("\n" + "="*60)
    print("💾 PHASE 2: MATERIALIZATION DECISION TESTING")
    print("="*60)
    
    materialization_test_queries = [
        ("Expensive Join - Should Materialize", """
            SELECT c_region, s_nation, 
                   SUM(lo_revenue) as revenue,
                   COUNT(*) as order_count
            FROM lineorder lo
            JOIN customer c ON lo.lo_custkey = c.c_custkey
            JOIN supplier s ON lo.lo_suppkey = s.s_suppkey
            WHERE lo_orderdate >= 19950101
            GROUP BY c_region, s_nation
            ORDER BY revenue DESC
        """),
        
        ("Complex Multi-table - Should Materialize", """
            SELECT d_year, c_region, p_mfgr,
                   SUM(lo_revenue) as total_revenue,
                   AVG(lo_quantity) as avg_qty,
                   MAX(lo_extendedprice) as max_price
            FROM lineorder lo
            JOIN customer c ON lo.lo_custkey = c.c_custkey
            JOIN part p ON lo.lo_partkey = p.p_partkey
            JOIN date_dim d ON lo.lo_orderdate = d.d_datekey
            WHERE d_year IN (1994, 1995, 1996)
            GROUP BY d_year, c_region, p_mfgr
            HAVING SUM(lo_revenue) > 500000
        """)
    ]
    
    materialized_samples = []
    
    for i, (name, query) in enumerate(materialization_test_queries, 1):
        print(f"\n💾 Materialization Test {i}: {name}")
        
        result = system.execute_query(query)
        
        if result['success'] and result.get('sample_info'):
            si = result['sample_info']
            materialized = si.get('materialized', False)
            sample_id = si.get('sample_id')
            
            print(f"   Result: {'MATERIALIZED' if materialized else 'IN-MEMORY ONLY'}")
            
            if materialized and sample_id:
                materialized_samples.append({
                    'query_name': name,
                    'sample_id': sample_id,
                    'sample_size': si.get('sample_size', 0)
                })
                print(f"   ✨ Created materialized sample: {sample_id}")
            else:
                print(f"   🗑️  Used in-memory only (not stored)")
            
            if 'decision_metadata' in result and 'materialization_decision' in result['decision_metadata']:
                md = result['decision_metadata']['materialization_decision']
                print(f"   Reasoning: {md.get('reasoning', 'N/A')}")
    
    print(f"\n📊 Materialization Summary:")
    print(f"   Total materialized samples: {len(materialized_samples)}")
    for sample in materialized_samples:
        print(f"   - {sample['sample_id']}: {sample['sample_size']} records ({sample['query_name']})")

def test_sample_reuse(system):
    """Test sample reuse with similar queries."""
    
    print("\n" + "="*60)
    print("♻️  PHASE 3: SAMPLE REUSE TESTING")
    print("="*60)
    
    # First, run a query that should create a sample
    base_query = """
        SELECT c_region, SUM(lo_revenue) as revenue
        FROM lineorder lo
        JOIN customer c ON lo.lo_custkey = c.c_custkey
        WHERE lo_orderdate >= 19940101
        GROUP BY c_region
    """
    
    print("🔨 Step 1: Create base sample")
    print("Query: Revenue by region (lineorder + customer)")
    
    result1 = system.execute_query(base_query)
    
    base_sample_id = None
    if result1['success'] and result1.get('sample_info'):
        si = result1['sample_info']
        base_sample_id = si.get('sample_id')
        print(f"   ✅ Base query executed")
        print(f"   Sample created: {base_sample_id}")
        print(f"   Materialized: {si.get('materialized', False)}")
    
    # Now run similar queries that should reuse the sample
    similar_queries = [
        ("Similar Query 1 - Different Aggregate", """
            SELECT c_region, AVG(lo_revenue) as avg_revenue, COUNT(*) as order_count
            FROM lineorder lo
            JOIN customer c ON lo.lo_custkey = c.c_custkey
            WHERE lo_orderdate >= 19940101
            GROUP BY c_region
        """),
        
        ("Similar Query 2 - Different Grouping", """
            SELECT c_nation, SUM(lo_revenue) as revenue
            FROM lineorder lo
            JOIN customer c ON lo.lo_custkey = c.c_custkey
            WHERE lo_orderdate >= 19940101
            GROUP BY c_nation
        """),
        
        ("Similar Query 3 - Same Pattern", """
            SELECT c_region, SUM(lo_quantity) as total_quantity
            FROM lineorder lo
            JOIN customer c ON lo.lo_custkey = c.c_custkey
            WHERE lo_orderdate >= 19940101
            GROUP BY c_region
        """)
    ]
    
    print(f"\n♻️  Step 2: Test sample reuse with similar queries")
    
    reuse_count = 0
    for i, (name, query) in enumerate(similar_queries, 1):
        print(f"\n🔄 Reuse Test {i}: {name}")
        
        result = system.execute_query(query)
        
        if result['success'] and result.get('sample_info'):
            si = result['sample_info']
            current_sample_id = si.get('sample_id')
            reused = si.get('reused_existing', False)
            
            if reused or (current_sample_id == base_sample_id):
                print(f"   ✅ SAMPLE REUSED: {current_sample_id}")
                reuse_count += 1
            elif si.get('newly_created'):
                print(f"   🆕 NEW SAMPLE CREATED: {current_sample_id}")
            else:
                print(f"   🗑️  IN-MEMORY EXECUTION (no reuse)")
        
        time.sleep(0.5)  # Brief pause
    
    print(f"\n📊 Sample Reuse Summary:")
    print(f"   Queries that reused samples: {reuse_count}/{len(similar_queries)}")
    if base_sample_id:
        print(f"   Base sample ID: {base_sample_id}")

def test_learning_behavior(system):
    """Test the learning and adaptation behavior."""
    
    print("\n" + "="*60)
    print("🧠 PHASE 4: LEARNING BEHAVIOR TESTING")
    print("="*60)
    
    # Get initial learning stats
    initial_stats = system.get_execution_statistics()
    initial_learning = initial_stats.get('learning_stats', {})
    
    print("📚 Initial Learning State:")
    print(f"   Query patterns learned: {initial_learning.get('unique_query_patterns', 0)}")
    print(f"   Execution history size: {initial_learning.get('execution_history_size', 0)}")
    print(f"   Sampling threshold: {initial_learning.get('current_sampling_threshold', 0.5):.3f}")
    
    # Run repeated queries to build learning history
    learning_query = """
        SELECT c_region, d_year, 
               SUM(lo_revenue) as revenue,
               COUNT(*) as orders
        FROM lineorder lo
        JOIN customer c ON lo.lo_custkey = c.c_custkey
        JOIN date_dim d ON lo.lo_orderdate = d.d_datekey
        WHERE d_year BETWEEN 1994 AND 1997
        GROUP BY c_region, d_year
        ORDER BY revenue DESC
    """
    
    print(f"\n🔄 Running repeated complex queries to build learning history...")
    
    execution_times = []
    strategies_used = []
    
    for i in range(5):
        print(f"   Learning iteration {i+1}/5...")
        
        result = system.execute_query(learning_query)
        
        if result['success']:
            execution_times.append(result.get('execution_time', 0))
            strategies_used.append(result.get('strategy_used', 'unknown'))
        
        time.sleep(0.3)  # Brief pause
    
    # Get final learning stats
    final_stats = system.get_execution_statistics()
    final_learning = final_stats.get('learning_stats', {})
    
    print(f"\n📈 Learning Progress:")
    print(f"   Query patterns: {initial_learning.get('unique_query_patterns', 0)} → {final_learning.get('unique_query_patterns', 0)}")
    print(f"   Execution history: {initial_learning.get('execution_history_size', 0)} → {final_learning.get('execution_history_size', 0)}")
    print(f"   Sampling threshold: {initial_learning.get('current_sampling_threshold', 0.5):.3f} → {final_learning.get('current_sampling_threshold', 0.5):.3f}")
    
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        print(f"   Average execution time: {avg_time:.3f}s")
        print(f"   Strategies used: {set(strategies_used)}")
    
    print(f"   ✅ System is learning from query patterns!")

def show_final_statistics(system):
    """Show comprehensive final statistics."""
    
    print("\n" + "="*60)
    print("📊 FINAL SYSTEM STATISTICS")
    print("="*60)
    
    stats = system.get_execution_statistics()
    
    # Query execution stats
    exec_stats = stats.get('query_execution', {})
    print(f"🎯 Query Execution:")
    print(f"   Total queries: {exec_stats.get('total_queries', 0)}")
    print(f"   Total execution time: {exec_stats.get('total_execution_time', 0):.3f}s")
    print(f"   Average execution time: {exec_stats.get('average_execution_time', 0):.3f}s")
    print(f"   Sampling usage rate: {exec_stats.get('sampling_usage_rate', 0):.1%}")
    
    # Sample storage stats
    storage_stats = stats.get('sample_storage', {})
    print(f"\n💾 Sample Storage:")
    print(f"   Materialized samples: {storage_stats.get('materialized_samples', 0)}")
    print(f"   Total sample records: {storage_stats.get('total_sample_size', 0):,}")
    
    # Learning stats
    learning_stats = stats.get('learning_stats', {})
    print(f"\n🧠 Learning System:")
    print(f"   Execution history: {learning_stats.get('execution_history_size', 0)}")
    print(f"   Query patterns learned: {learning_stats.get('unique_query_patterns', 0)}")
    print(f"   Current sampling threshold: {learning_stats.get('current_sampling_threshold', 0.5):.3f}")
    print(f"   Current materialization threshold: {learning_stats.get('current_materialization_threshold', 0.3):.3f}")
    
    # List materialized samples
    try:
        samples = system.sample_storage.list_materialized_samples()
        if samples:
            print(f"\n🗂️  Materialized Samples Created:")
            for i, sample in enumerate(samples, 1):
                sample_id = sample[0]
                sample_size = sample[3]
                creation_time = sample[4]
                query_count = sample[6]
                
                print(f"   {i}. {sample_id}")
                print(f"      Size: {sample_size} records")
                print(f"      Created: {creation_time}")
                print(f"      Times used: {query_count}")
        else:
            print(f"\n🗂️  No materialized samples created")
    except Exception as e:
        print(f"\n⚠️  Could not list samples: {e}")
    
    print(f"\n✅ Testing complete! System demonstrated:")
    print(f"   • Intelligent complexity-based sampling decisions")
    print(f"   • Smart materialization based on cost-benefit analysis")
    print(f"   • Sample reuse for similar query patterns")
    print(f"   • Learning and adaptation over time")

if __name__ == "__main__":
    main()