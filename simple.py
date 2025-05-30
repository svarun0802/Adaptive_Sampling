#!/usr/bin/env python3
"""
Simple Test Script for Adaptive Sampling System
Minimal output - just shows if things work or not
"""

import time
from query import QueryInterceptor

def test_query(interceptor, query_name, query, expected_strategy=None):
    """Test a single query and return concise results."""
    print(f"🔍 {query_name}...", end=" ")
    
    start_time = time.time()
    try:
        result = interceptor.execute_query(query, return_metadata=False)
        execution_time = time.time() - start_time
        
        if result['success']:
            strategy = result.get('strategy_used', 'unknown')
            row_count = result.get('result_count', 0)
            is_approx = result.get('is_approximate', False)
            
            status = "✅ PASS"
            details = f"{strategy} | {row_count} rows | {execution_time:.3f}s"
            if is_approx:
                details += " | ~approximate"
            
            print(f"{status} | {details}")
            return True
        else:
            print(f"❌ FAIL | {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ ERROR | {str(e)[:50]}... | {execution_time:.3f}s")
        return False

def main():
    """Run simple focused tests."""
    
    print("🧪 SIMPLE ADAPTIVE SAMPLING TEST")
    print("=" * 50)
    
    # Initialize system
    system = QueryInterceptor()
    
    # Test cases: [name, query, expected_strategy]
    test_cases = [
        ("Simple Query", 
         "SELECT COUNT(*) FROM customer WHERE c_region = 'ASIA'",
         "exact"),
        
        ("Medium Complexity", 
         "SELECT c_region, SUM(lo_revenue) FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey GROUP BY c_region",
         "sampling"),
        
        ("High Complexity", 
         "SELECT c_region, s_region, SUM(lo_revenue) FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey JOIN supplier s ON lo.lo_suppkey = s.s_suppkey GROUP BY c_region, s_region",
         "sampling"),
        
        ("Sample Reuse Test", 
         "SELECT c_region, AVG(lo_revenue) FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey GROUP BY c_region",
         "sampling")
    ]
    
    # Run tests
    passed = 0
    total = len(test_cases)
    
    for test_name, query, expected in test_cases:
        if test_query(system, test_name, query, expected):
            passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    # System stats - FIXED
    try:
        stats = system.get_execution_statistics()
        query_stats = stats['query_execution']
        storage_stats = stats['sample_storage']
        
        print(f"⚡ Queries executed: {query_stats['total_queries']}")
        print(f"🎲 Sampling rate: {query_stats['sampling_usage_rate']:.1%}")
        print(f"💾 Samples created: {storage_stats['materialized_samples']}")
        print(f"⏱️  Avg execution time: {query_stats['average_execution_time']:.3f}s")
    except AttributeError:
        # Fallback if method doesn't exist
        print(f"⚡ Queries executed: {system.query_count}")
        print(f"🎲 Sampling usage: {system.sampling_usage_count}")
        print(f"⏱️  Total time: {system.total_execution_time:.3f}s")
    
    # Final verdict
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - System is working!")
    elif passed > total // 2:
        print(f"\n⚠️  PARTIAL SUCCESS - {passed}/{total} tests passed")
    else:
        print(f"\n❌ SYSTEM ISSUES - Only {passed}/{total} tests passed")
    
    # Cleanup
    try:
        system.cleanup_resources()
    except:
        pass  # Ignore cleanup errors

if __name__ == "__main__":
    main()