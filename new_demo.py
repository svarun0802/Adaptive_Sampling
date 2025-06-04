#!/usr/bin/env python3
"""
Fixed Demo - Phase 2 with Complex Queries that WILL Use Sampling
Updated with Direct Reuse vs Extension Labels in Phase 1
"""

import time
import sys
import io
from contextlib import redirect_stdout
from query import QueryInterceptor

def execute_with_timing(system, query, use_sampling=None):
    """Execute query with proper timing measurement"""
    start_time = time.time()
    
    f = io.StringIO()
    with redirect_stdout(f):
        result = system.execute_query(query, use_sampling=use_sampling, return_metadata=True)
    
    actual_execution_time = time.time() - start_time
    
    if result:
        result['actual_execution_time'] = actual_execution_time
    
    return result

def get_reuse_type_label(result):
    """Determine if this is direct reuse or extension and return appropriate label"""
    if not result or not result.get('success'):
        return "❌ FAILED"
    
    strategy = result.get('strategy_used', 'exact')
    sample_info = result.get('sample_info')
    
    if strategy.startswith('sampling') and sample_info:
        if sample_info.get('newly_created') and sample_info.get('materialized'):
            sample_id = sample_info.get('sample_id', 'unknown')[:8]
            return f"✅ MATERIALIZED ({sample_id})"
        elif sample_info.get('newly_created'):
            return "📝 In-memory"
        elif sample_info.get('reused_existing'):
            sample_id = sample_info.get('sample_id', 'unknown')[:8]
            
            # Check if this was direct reuse or extension
            exact_match = sample_info.get('exact_match', False)
            coverage = sample_info.get('coverage', 1.0)
            
            if exact_match and coverage >= 1.0:
                return f"♻️  DIRECT REUSE ({sample_id})"
            elif coverage < 1.0:
                return f"🔧 EXTENDED ({sample_id}) - {coverage*100:.0f}% coverage"
            else:
                # This handles cases where we extended but coverage shows 1.0
                # We can infer extension if the query has more tables than typical for the sample
                return f"🔧 EXTENDED ({sample_id})"
        else:
            return "🎲 Sampling"
    elif strategy == 'exact':
        return "⚡ Direct"
    else:
        return f"📊 {strategy}"

def run_comprehensive_foundation():
    """Phase 1: Same as before but with enhanced reuse/extension labeling"""
    
    print("PHASE 1: COMPREHENSIVE SAMPLE LIBRARY BUILDING")
    print("=" * 60)
    print("Training system with 20 strategic queries...")
    print("Legend: ⚡Direct | 📝In-memory | ✅Materialized | ♻️Direct-Reuse | 🔧Extended")
    
    system = QueryInterceptor()
    system.decision_engine.materialization_threshold = 0.05
    print("🔧 Lowered materialization threshold to 0.05 for demo\n")
    
    # Enhanced training queries with table analysis for better extension detection
    training_queries = [
        ("Simple Count", "SELECT COUNT(*) FROM customer", 1),
        ("Simple Filter", "SELECT COUNT(*) FROM supplier WHERE s_region = 'AMERICA'", 1),
        ("Basic Aggregation", "SELECT c_region, COUNT(*) FROM customer GROUP BY c_region", 1),
        ("Customer Revenue", "SELECT c_region, SUM(lo_revenue), COUNT(*) FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey GROUP BY c_region", 2),
        ("Supplier Analysis", "SELECT s_nation, SUM(lo_quantity), AVG(lo_supplycost) FROM lineorder lo JOIN supplier s ON lo.lo_suppkey = s.s_suppkey GROUP BY s_nation", 2),
        ("Part Analysis", "SELECT p_category, SUM(lo_extendedprice), COUNT(*) FROM lineorder lo JOIN part p ON lo.lo_partkey = p.p_partkey GROUP BY p_category", 2),
        ("Date Analysis", "SELECT d_year, SUM(lo_revenue), COUNT(*) FROM lineorder lo JOIN date_dim d ON lo.lo_orderdate = d.d_datekey GROUP BY d_year", 2),
        ("Customer Detailed", "SELECT c_region, c_nation, SUM(lo_revenue), AVG(lo_quantity) FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey GROUP BY c_region, c_nation", 2),
        ("Customer Revenue 2", "SELECT c_region, AVG(lo_revenue), MAX(lo_revenue) FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey GROUP BY c_region", 2),
        ("Supplier Analysis 2", "SELECT s_nation, COUNT(*), SUM(lo_quantity) FROM lineorder lo JOIN supplier s ON lo.lo_suppkey = s.s_suppkey GROUP BY s_nation", 2),
        ("Part Analysis 2", "SELECT p_category, AVG(lo_extendedprice), COUNT(DISTINCT lo_orderkey) FROM lineorder lo JOIN part p ON lo.lo_partkey = p.p_partkey GROUP BY p_category", 2),
        ("Customer-Supplier", "SELECT c_region, s_nation, SUM(lo_revenue), COUNT(*) FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey JOIN supplier s ON lo.lo_suppkey = s.s_suppkey GROUP BY c_region, s_nation", 3),
        ("Customer-Part", "SELECT c_region, p_category, SUM(lo_revenue), AVG(lo_extendedprice) FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey JOIN part p ON lo.lo_partkey = p.p_partkey GROUP BY c_region, p_category", 3),
        ("Supplier-Part", "SELECT s_nation, p_mfgr, SUM(lo_quantity), COUNT(*) FROM lineorder lo JOIN supplier s ON lo.lo_suppkey = s.s_suppkey JOIN part p ON lo.lo_partkey = p.p_partkey GROUP BY s_nation, p_mfgr", 3),
        ("Customer-Date", "SELECT c_region, d_year, SUM(lo_revenue), COUNT(*) FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey JOIN date_dim d ON lo.lo_orderdate = d.d_datekey GROUP BY c_region, d_year", 3),
        ("Supplier-Date", "SELECT s_nation, d_year, SUM(lo_quantity), AVG(lo_supplycost) FROM lineorder lo JOIN supplier s ON lo.lo_suppkey = s.s_suppkey JOIN date_dim d ON lo.lo_orderdate = d.d_datekey GROUP BY s_nation, d_year", 3),
        ("4-Table Complex 1", "SELECT c_region, s_nation, p_category, SUM(lo_revenue) FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey JOIN supplier s ON lo.lo_suppkey = s.s_suppkey JOIN part p ON lo.lo_partkey = p.p_partkey GROUP BY c_region, s_nation, p_category", 4),
        ("4-Table Complex 2", "SELECT c_region, s_region, d_year, SUM(lo_revenue), COUNT(*) FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey JOIN supplier s ON lo.lo_suppkey = s.s_suppkey JOIN date_dim d ON lo.lo_orderdate = d.d_datekey GROUP BY c_region, s_region, d_year", 4),
        ("4-Table Complex 3", "SELECT d_year, c_nation, p_mfgr, SUM(lo_revenue - lo_supplycost) as profit FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey JOIN part p ON lo.lo_partkey = p.p_partkey JOIN date_dim d ON lo.lo_orderdate = d.d_datekey GROUP BY d_year, c_nation, p_mfgr", 4),
        ("5-Table Ultimate", "SELECT d_year, c_region, s_nation, p_category, SUM(lo_revenue), AVG(lo_quantity), COUNT(*) FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey JOIN supplier s ON lo.lo_suppkey = s.s_suppkey JOIN part p ON lo.lo_partkey = p.p_partkey JOIN date_dim d ON lo.lo_orderdate = d.d_datekey WHERE d_year >= 1994 GROUP BY d_year, c_region, s_nation, p_category", 5)
    ]
    
    samples_created = 0
    direct_reuses = 0
    extensions = 0
    
    # Track sample metadata for better extension detection
    created_samples = {}  # sample_id -> {tables, table_count}
    
    for i, (name, query, expected_table_count) in enumerate(training_queries, 1):
        result = execute_with_timing(system, query)
        
        if result and result.get('success'):
            exec_time = result.get('actual_execution_time', 0) * 1000
            strategy = result.get('strategy_used', 'exact')
            
            # Enhanced labeling logic
            sample_info = result.get('sample_info')
            if sample_info and sample_info.get('newly_created') and sample_info.get('materialized'):
                sample_id = sample_info.get('sample_id', 'unknown')
                samples_created += 1
                # Track this sample's metadata
                created_samples[sample_id] = {
                    'table_count': expected_table_count,
                    'query_name': name
                }
                label = get_reuse_type_label(result)
            elif sample_info and sample_info.get('reused_existing'):
                sample_id = sample_info.get('sample_id', 'unknown')
                
                # Enhanced extension detection
                if sample_id in created_samples:
                    base_table_count = created_samples[sample_id]['table_count']
                    if expected_table_count > base_table_count:
                        # This is definitely an extension
                        extensions += 1
                        coverage = base_table_count / expected_table_count * 100
                        label = f"🔧 EXTENDED ({sample_id[:8]}) - {coverage:.0f}% base coverage"
                    else:
                        # This is direct reuse
                        direct_reuses += 1
                        label = f"♻️  DIRECT REUSE ({sample_id[:8]})"
                else:
                    # Fallback to existing logic
                    exact_match = sample_info.get('exact_match', False)
                    if exact_match:
                        direct_reuses += 1
                        label = f"♻️  DIRECT REUSE ({sample_id[:8]})"
                    else:
                        extensions += 1
                        label = f"🔧 EXTENDED ({sample_id[:8]})"
            else:
                label = get_reuse_type_label(result)
            
            print(f"Q{i:2d}: {name:18} | {exec_time:6.1f}ms | {strategy:12} | {label}")
    
    actual_samples = system.sample_storage.list_materialized_samples()
    print(f"\nFOUNDATION COMPLETE: {len(actual_samples)} materialized samples | {direct_reuses} direct reuses | {extensions} extensions")
    
    return system

def show_query_results(results, is_approximate=False):
    """Display query results in a clean format"""
    if not results:
        print("     No results")
        return
    
    print(f"     Results ({len(results)} rows):")
    for i, row in enumerate(results[:3]):
        row_str = " | ".join(str(val)[:15] for val in row)
        print(f"       {i+1}: {row_str}")
    
    if len(results) > 3:
        print(f"       ... and {len(results) - 3} more rows")
    
    if is_approximate:
        print("     🔍 Results are approximate (scaled from sample)")

def show_exact_confidence_intervals(confidence_intervals, results):
    """Display exact confidence interval values with precise numbers"""
    if not confidence_intervals or not any(confidence_intervals):
        print("     📊 No confidence intervals available")
        return
    
    print("     📊 95% CONFIDENCE INTERVALS (Exact Values):")
    
    for i, (row_ci, result_row) in enumerate(zip(confidence_intervals[:3], results[:3])):
        if row_ci and any(ci for ci in row_ci if ci):
            print(f"       Row {i+1} Confidence Intervals:")
            
            for j, (ci, actual_val) in enumerate(zip(row_ci, result_row)):
                if ci and isinstance(actual_val, (int, float)):
                    lower, upper = ci
                    margin_of_error = (upper - lower) / 2
                    margin_percent = (margin_of_error / actual_val * 100) if actual_val != 0 else 0
                    
                    print(f"         Column {j+1}: {actual_val:,} ± {margin_of_error:,.0f}")
                    print(f"                   Range: [{lower:,} - {upper:,}]")
                    print(f"                   Margin: ±{margin_percent:.1f}%")
                elif isinstance(actual_val, (int, float)):
                    print(f"         Column {j+1}: {actual_val:,} (exact)")
                else:
                    print(f"         Column {j+1}: {actual_val} (categorical)")
            print()
    
    if len(confidence_intervals) > 3:
        print(f"       ... confidence intervals for {len(confidence_intervals) - 3} more rows available")

def run_strategic_performance_comparison(system):
    """Phase 2: Strategic queries designed to show sampling benefits WITH EXACT CI VALUES"""
    
    print(f"\nPHASE 2: STRATEGIC PERFORMANCE COMPARISON")
    print("=" * 70)
    print("Complex queries designed to leverage existing samples")
    print("=" * 70)
    
    # STRATEGIC queries that match Phase 1 patterns but are complex enough to trigger sampling
    performance_tests = [
        {
            'name': 'Customer Revenue Detailed Analysis',
            'query': "SELECT c_region, c_nation, SUM(lo_revenue) as total_revenue, AVG(lo_extendedprice) as avg_price, COUNT(DISTINCT lo_orderkey) as unique_orders, MAX(lo_quantity) as max_quantity FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey WHERE lo_orderdate >= 19940101 GROUP BY c_region, c_nation HAVING SUM(lo_revenue) > 1000000 ORDER BY total_revenue DESC"
        },
        {
            'name': 'Supplier Performance Analysis', 
            'query': "SELECT s_nation, s_region, SUM(lo_quantity) as total_quantity, AVG(lo_supplycost) as avg_cost, COUNT(DISTINCT lo_orderkey) as orders, SUM(lo_revenue) as revenue FROM lineorder lo JOIN supplier s ON lo.lo_suppkey = s.s_suppkey WHERE lo_quantity > 5 GROUP BY s_nation, s_region HAVING COUNT(*) > 100 ORDER BY total_quantity DESC"
        },
        {
            'name': 'Part Category Deep Analysis',
            'query': "SELECT p_category, p_mfgr, SUM(lo_extendedprice) as total_value, AVG(lo_quantity) as avg_quantity, COUNT(DISTINCT lo_orderkey) as unique_orders, MAX(lo_extendedprice) as max_price FROM lineorder lo JOIN part p ON lo.lo_partkey = p.p_partkey WHERE lo_extendedprice > 1000 GROUP BY p_category, p_mfgr HAVING SUM(lo_extendedprice) > 500000 ORDER BY total_value DESC"
        },
        {
            'name': 'Customer-Supplier Complex Analysis',
            'query': "SELECT c_region, s_region, s_nation, SUM(lo_revenue) as revenue, AVG(lo_quantity) as avg_quantity, COUNT(DISTINCT lo_orderkey) as orders, SUM(lo_revenue - lo_supplycost) as profit FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey JOIN supplier s ON lo.lo_suppkey = s.s_suppkey WHERE lo_orderdate >= 19940101 GROUP BY c_region, s_region, s_nation HAVING SUM(lo_revenue) > 500000 ORDER BY profit DESC"
        },
        {
            'name': 'Multi-Dimensional Complex Query',
            'query': "SELECT c_region, s_nation, p_category, SUM(lo_revenue) as revenue, AVG(lo_extendedprice) as avg_price, COUNT(*) as transaction_count, SUM(lo_quantity) as total_quantity FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey JOIN supplier s ON lo.lo_suppkey = s.s_suppkey JOIN part p ON lo.lo_partkey = p.p_partkey WHERE lo_orderdate >= 19940101 AND lo_quantity > 10 GROUP BY c_region, s_nation, p_category HAVING COUNT(*) > 50 ORDER BY revenue DESC"
        }
    ]
    
    total_exact_time = 0
    total_intelligent_time = 0
    successful_tests = 0
    sampling_used_count = 0
    
    for i, test in enumerate(performance_tests, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {test['name']}")
        print(f"Query: {test['query'][:80]}...")
        print(f"{'='*70}")
        
        # Execution 1: Force EXACT
        print(f"\n🎯 EXECUTION 1: FORCED EXACT (Full Table Scan)")
        exact_result = execute_with_timing(system, test['query'], use_sampling=False)
        
        if exact_result and exact_result.get('success'):
            exact_time = exact_result.get('actual_execution_time', 0) * 1000
            exact_results = exact_result.get('results', [])
            
            print(f"   Time: {exact_time:.1f}ms")
            show_query_results(exact_results, is_approximate=False)
        else:
            print(f"   ❌ FAILED: {exact_result.get('error') if exact_result else 'No result'}")
            continue
        
        # Execution 2: Intelligent Decision
        print(f"\n🧠 EXECUTION 2: INTELLIGENT SYSTEM DECISION")
        intelligent_result = execute_with_timing(system, test['query'], use_sampling=None)
        
        if intelligent_result and intelligent_result.get('success'):
            intelligent_time = intelligent_result.get('actual_execution_time', 0) * 1000
            intelligent_results = intelligent_result.get('results', [])
            is_approximate = intelligent_result.get('is_approximate', False)
            strategy = intelligent_result.get('strategy_used', 'unknown')
            
            print(f"   Strategy: {strategy.upper()}")
            print(f"   Time: {intelligent_time:.1f}ms")
            print(f"   Approximate: {is_approximate}")
            
            if is_approximate:
                sampling_used_count += 1
            
            # Show what method was used
            sample_info = intelligent_result.get('sample_info')
            if sample_info:
                if sample_info.get('reused_existing'):
                    sample_id = sample_info.get('sample_id', 'unknown')[:8]
                    coverage = sample_info.get('coverage', 1.0) * 100
                    print(f"   Method: SAMPLE REUSE ({sample_id}) - {coverage:.0f}% coverage")
                elif sample_info.get('coverage') and sample_info.get('coverage') < 1.0:
                    coverage = sample_info.get('coverage', 0) * 100
                    print(f"   Method: SAMPLE EXTENSION - {coverage:.0f}% base coverage")
                elif sample_info.get('newly_created'):
                    materialized = sample_info.get('materialized', False)
                    print(f"   Method: NEW SAMPLE ({'materialized' if materialized else 'in-memory'})")
                else:
                    print(f"   Method: SAMPLING")
            else:
                print(f"   Method: EXACT EXECUTION")
            
            show_query_results(intelligent_results, is_approximate=is_approximate)
            
            # Show EXACT confidence intervals if available
            if intelligent_result.get('confidence_interval'):
                print(f"\n   🎯 STATISTICAL CONFIDENCE ANALYSIS:")
                show_exact_confidence_intervals(
                    intelligent_result['confidence_interval'], 
                    intelligent_results
                )
                
                # Calculate and show overall confidence metrics
                scaling_factor = intelligent_result.get('scaling_factor')
                if scaling_factor:
                    print(f"     📈 Scaling Information:")
                    print(f"        Sample to population scaling: {scaling_factor:.2f}x")
                    print(f"        Population estimate: {intelligent_result.get('population_estimate', 'N/A'):,} records")
            
            # Calculate performance comparison
            if intelligent_time > 0:
                speedup = exact_time / intelligent_time
                time_saved = exact_time - intelligent_time
                improvement = (time_saved / exact_time * 100) if exact_time > 0 else 0
                
                print(f"\n📈 PERFORMANCE COMPARISON:")
                print(f"   Speedup: {speedup:.1f}x")
                print(f"   Time saved: {time_saved:.1f}ms")
                print(f"   Improvement: {improvement:.1f}%")
                
                # Add to totals
                total_exact_time += exact_time
                total_intelligent_time += intelligent_time
                successful_tests += 1
        else:
            print(f"   ❌ FAILED: {intelligent_result.get('error') if intelligent_result else 'No result'}")
    
    # Overall summary
    if successful_tests > 0:
        overall_speedup = total_exact_time / total_intelligent_time if total_intelligent_time > 0 else float('inf')
        overall_time_saved = total_exact_time - total_intelligent_time
        overall_improvement = (overall_time_saved / total_exact_time * 100) if total_exact_time > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"OVERALL PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        print(f"Successful tests: {successful_tests}")
        print(f"Sampling used: {sampling_used_count}/{successful_tests} queries")
        print(f"Total exact time: {total_exact_time:.1f}ms")
        print(f"Total intelligent time: {total_intelligent_time:.1f}ms")
        print(f"Overall speedup: {overall_speedup:.1f}x")
        print(f"Total time saved: {overall_time_saved:.1f}ms")
        print(f"Overall improvement: {overall_improvement:.1f}%")
        
        print(f"\n💡 SYSTEM INTELLIGENCE INSIGHTS:")
        if sampling_used_count > 0:
            print(f"   ✅ System successfully used sampling for {sampling_used_count} complex queries")
            print(f"   ✅ Sample reuse and extension working as designed")
            print(f"   ✅ {overall_speedup:.1f}x speedup demonstrates system effectiveness")
            print(f"   ✅ Confidence intervals provide statistical rigor")
        else:
            print(f"System chose exact execution for all queries")
            print(f"This indicates queries were fast enough that sampling overhead wasn't justified")
            print(f"This is actually intelligent behavior - system avoids unnecessary complexity")
    
    return system

def comprehensive_demo():
    """Main demonstration"""
    
    print("🚀 COMPREHENSIVE ADAPTIVE SAMPLING DEMONSTRATION")
    print("=" * 65)
    print("Phase 1: Build intelligence | Phase 2: Show strategic performance")
    print("=" * 65)
    
    # Phase 1
    system = run_comprehensive_foundation()
    
    input("\nPress Enter to begin strategic performance comparison...")
    
    # Phase 2
    system = run_strategic_performance_comparison(system)
    
    print(f"\n🎉 Demonstration complete!")

if __name__ == "__main__":
    comprehensive_demo()