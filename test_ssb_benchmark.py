#!/usr/bin/env python3
"""
Star Schema Benchmark (SSB) Test Suite for Adaptive Sampling System

Tests the complete system using standard SSB queries in increasing complexity.
Assumes SSB data already exists in the database.

SSB Query Categories:
- Q1.x: Single dimension queries (date filtering)
- Q2.x: Two dimension queries (date + one other)  
- Q3.x: Three dimension queries (date + customer + supplier)
- Q4.x: Four dimension queries (all dimensions)

Usage:
    python test_ssb_benchmark.py
"""

import time
import sys
import os
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import DatabaseConnector
from query import SQLQueryParser, QueryInterceptor

class SSBBenchmarkTest:
    """Complete SSB benchmark test suite."""
    
    def __init__(self):
        """Initialize benchmark test environment."""
        print("📊 SSB BENCHMARK TEST SUITE")
        print("=" * 80)
        print("Testing Adaptive Sampling System with Standard SSB Queries")
        print("=" * 80)
        
        # Initialize components
        self.db = DatabaseConnector()
        self.interceptor = QueryInterceptor(self.db)
        self.parser = SQLQueryParser()
        
        # Test results tracking
        self.test_results = []
        self.performance_results = []
        
        print(f"✅ Benchmark environment initialized")
        
    def run_benchmark_suite(self):
        """Run the complete SSB benchmark suite."""
        try:
            print(f"\n📋 BENCHMARK EXECUTION PLAN")
            print(f"1. Verify existing data")
            print(f"2. Component integration check")
            print(f"3. SSB Q1.x queries (Single dimension)")
            print(f"4. SSB Q2.x queries (Two dimensions)")
            print(f"5. SSB Q3.x queries (Three dimensions)")
            print(f"6. SSB Q4.x queries (Four dimensions)")
            print(f"7. Performance analysis")
            print(f"8. System statistics")
            
            # Execute benchmark phases
            self.verify_existing_data()
            self.quick_component_check()
            self.run_q1_queries()
            self.run_q2_queries()
            self.run_q3_queries()
            self.run_q4_queries()
            self.analyze_performance()
            self.show_final_statistics()
            
            print(f"\n🎉 SSB BENCHMARK COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            print(f"\n❌ BENCHMARK FAILED: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def verify_existing_data(self):
        """Verify that SSB data exists in the database."""
        print(f"\n" + "="*60)
        print(f"📊 PHASE 1: VERIFYING EXISTING SSB DATA")
        print(f"="*60)
        
        required_tables = ['lineorder', 'customer', 'supplier', 'part', 'date_dim']
        
        print(f"🔍 Checking required tables...")
        for table in required_tables:
            try:
                count_result = self.db.execute_query(f"SELECT COUNT(*) FROM {table}")
                if count_result and count_result[0][0] > 0:
                    count = count_result[0][0]
                    print(f"   ✅ {table}: {count:,} rows")
                else:
                    raise Exception(f"Table {table} is empty or doesn't exist")
            except Exception as e:
                print(f"   ❌ {table}: {e}")
                raise Exception(f"Required table {table} not found. Please run generate_data.py first.")
        
        # Check for reasonable data relationships
        print(f"\n🔗 Checking data relationships...")
        
        # Check foreign key relationships
        fk_checks = [
            ("lineorder-customer", "SELECT COUNT(*) FROM lineorder lo JOIN customer c ON lo.lo_custkey = c.c_custkey"),
            ("lineorder-supplier", "SELECT COUNT(*) FROM lineorder lo JOIN supplier s ON lo.lo_suppkey = s.s_suppkey"),
            ("lineorder-part", "SELECT COUNT(*) FROM lineorder lo JOIN part p ON lo.lo_partkey = p.p_partkey"),
            ("lineorder-date", "SELECT COUNT(*) FROM lineorder lo JOIN date_dim d ON lo.lo_orderdate = d.d_datekey")
        ]
        
        for check_name, check_query in fk_checks:
            try:
                result = self.db.execute_query(check_query)
                if result and result[0][0] > 0:
                    print(f"   ✅ {check_name}: {result[0][0]:,} valid joins")
                else:
                    print(f"   ⚠️  {check_name}: No valid joins found")
            except Exception as e:
                print(f"   ❌ {check_name}: {e}")
        
        print(f"✅ Data verification complete")
    
    def quick_component_check(self):
        """Quick check that all components are working."""
        print(f"\n" + "="*60)
        print(f"🔧 PHASE 2: COMPONENT INTEGRATION CHECK")
        print(f"="*60)
        
        # Test 1: Simple query parsing
        print(f"📝 Testing query parsing...")
        test_query = "SELECT COUNT(*) FROM customer"
        parsed = self.parser.parse(test_query)
        print(f"   ✅ Parser working - complexity: {parsed['complexity_score']}")
        
        # Test 2: Simple exact execution
        print(f"🎯 Testing exact execution...")
        result = self.interceptor.execute_query(test_query, use_sampling=False)
        if result['success']:
            print(f"   ✅ Exact execution working - result: {result['results'][0][0]:,}")
        else:
            print(f"   ❌ Exact execution failed: {result.get('error')}")
        
        # Test 3: Check sampling components exist
        print(f"🎲 Testing sampling components...")
        try:
            from sampling import WanderJoin, SampleStorage
            wj = WanderJoin(self.db)
            storage = SampleStorage(self.db)
            samples = storage.list_materialized_samples()
            print(f"   ✅ Sampling components working - existing samples: {len(samples)}")
        except Exception as e:
            print(f"   ❌ Sampling components failed: {e}")
        
        print(f"✅ Component integration check complete")
    
    def run_q1_queries(self):
        """Run SSB Q1.x queries (Single dimension - Date filtering)."""
        print(f"\n" + "="*60)
        print(f"📊 PHASE 3: SSB Q1.x QUERIES (Single Dimension)")
        print(f"="*60)
        
        q1_queries = [
            ("SSB Q1.1", """
                SELECT SUM(lo_extendedprice * lo_discount) AS revenue
                FROM lineorder lo, date_dim d
                WHERE lo.lo_orderdate = d.d_datekey
                AND d.d_year = 1993
                AND lo.lo_discount BETWEEN 1 AND 3
                AND lo.lo_quantity < 25
            """, "Simple date filtering with range conditions"),
            
            ("SSB Q1.2", """
                SELECT SUM(lo_extendedprice * lo_discount) AS revenue
                FROM lineorder lo, date_dim d
                WHERE lo.lo_orderdate = d.d_datekey
                AND d.d_yearmonthnum = 199401
                AND lo.lo_discount BETWEEN 4 AND 6
                AND lo.lo_quantity BETWEEN 26 AND 35
            """, "Month-specific filtering"),
            
            ("SSB Q1.3", """
                SELECT SUM(lo_extendedprice * lo_discount) AS revenue
                FROM lineorder lo, date_dim d
                WHERE lo.lo_orderdate = d.d_datekey
                AND d.d_weeknuminyear = 6
                AND d.d_year = 1994
                AND lo.lo_discount BETWEEN 5 AND 7
                AND lo.lo_quantity BETWEEN 26 AND 35
            """, "Week-specific filtering")
        ]
        
        self._run_query_group("Q1", q1_queries, expected_complexity="medium")
    
    def run_q2_queries(self):
        """Run SSB Q2.x queries (Two dimensions - Date + one other)."""
        print(f"\n" + "="*60)
        print(f"📊 PHASE 4: SSB Q2.x QUERIES (Two Dimensions)")
        print(f"="*60)
        
        q2_queries = [
            ("SSB Q2.1", """
                SELECT SUM(lo_revenue) AS lo_revenue, d_year, p_brand
                FROM lineorder lo, date_dim d, part p, supplier s
                WHERE lo.lo_orderdate = d.d_datekey
                AND lo.lo_partkey = p.p_partkey
                AND lo.lo_suppkey = s.s_suppkey
                AND p.p_category = 'MFGR#12'
                AND s.s_region = 'AMERICA'
                GROUP BY d_year, p_brand
                ORDER BY d_year, p_brand
            """, "Date + Part dimension with supplier filtering"),
            
            ("SSB Q2.2", """
                SELECT SUM(lo_revenue) AS lo_revenue, d_year, p_brand
                FROM lineorder lo, date_dim d, part p, supplier s
                WHERE lo.lo_orderdate = d.d_datekey
                AND lo.lo_partkey = p.p_partkey
                AND lo.lo_suppkey = s.s_suppkey
                AND p.p_brand BETWEEN 'MFGR#2221' AND 'MFGR#2228'
                AND s.s_region = 'ASIA'
                GROUP BY d_year, p_brand
                ORDER BY d_year, p_brand
            """, "Date + Part with brand range filtering"),
            
            ("SSB Q2.3", """
                SELECT SUM(lo_revenue) AS lo_revenue, d_year, p_brand
                FROM lineorder lo, date_dim d, part p, supplier s
                WHERE lo.lo_orderdate = d.d_datekey
                AND lo.lo_partkey = p.p_partkey
                AND lo.lo_suppkey = s.s_suppkey
                AND p.p_brand = 'MFGR#2239'
                AND s.s_region = 'EUROPE'
                GROUP BY d_year, p_brand
                ORDER BY d_year, p_brand
            """, "Date + Part with specific brand")
        ]
        
        self._run_query_group("Q2", q2_queries, expected_complexity="high")
    
    def run_q3_queries(self):
        """Run SSB Q3.x queries (Three dimensions - Date + Customer + Supplier)."""
        print(f"\n" + "="*60)
        print(f"📊 PHASE 5: SSB Q3.x QUERIES (Three Dimensions)")
        print(f"="*60)
        
        q3_queries = [
            ("SSB Q3.1", """
                SELECT c_nation, s_nation, d_year,
                       SUM(lo_revenue) AS lo_revenue
                FROM customer c, lineorder lo, supplier s, date_dim d
                WHERE lo.lo_custkey = c.c_custkey
                AND lo.lo_suppkey = s.s_suppkey
                AND lo.lo_orderdate = d.d_datekey
                AND c.c_region = 'ASIA'
                AND s.s_region = 'ASIA'
                AND d.d_year >= 1992 AND d.d_year <= 1997
                GROUP BY c_nation, s_nation, d_year
                ORDER BY d_year ASC, lo_revenue DESC
            """, "Customer + Supplier + Date with region filtering"),
            
            ("SSB Q3.2", """
                SELECT c_city, s_city, d_year,
                       SUM(lo_revenue) AS lo_revenue
                FROM customer c, lineorder lo, supplier s, date_dim d
                WHERE lo.lo_custkey = c.c_custkey
                AND lo.lo_suppkey = s.s_suppkey
                AND lo.lo_orderdate = d.d_datekey
                AND c.c_nation = 'UNITED STATES'
                AND s.s_nation = 'UNITED STATES'
                AND d.d_year >= 1992 AND d.d_year <= 1997
                GROUP BY c_city, s_city, d_year
                ORDER BY d_year ASC, lo_revenue DESC
            """, "Customer + Supplier + Date with nation filtering"),
            
            ("SSB Q3.3", """
                SELECT c_city, s_city, d_year,
                       SUM(lo_revenue) AS lo_revenue
                FROM customer c, lineorder lo, supplier s, date_dim d
                WHERE lo.lo_custkey = c.c_custkey
                AND lo.lo_suppkey = s.s_suppkey
                AND lo.lo_orderdate = d.d_datekey
                AND (c.c_city = 'UNITED KI1' OR c.c_city = 'UNITED KI5')
                AND (s.s_city = 'UNITED KI1' OR s.s_city = 'UNITED KI5')
                AND d.d_year >= 1992 AND d.d_year <= 1997
                GROUP BY c_city, s_city, d_year
                ORDER BY d_year ASC, lo_revenue DESC
            """, "Customer + Supplier + Date with city filtering"),
            
            ("SSB Q3.4", """
                SELECT c_city, s_city, d_year,
                       SUM(lo_revenue) AS lo_revenue
                FROM customer c, lineorder lo, supplier s, date_dim d
                WHERE lo.lo_custkey = c.c_custkey
                AND lo.lo_suppkey = s.s_suppkey
                AND lo.lo_orderdate = d.d_datekey
                AND (c.c_city = 'UNITED KI1' OR c.c_city = 'UNITED KI5')
                AND (s.s_city = 'UNITED KI1' OR s.s_city = 'UNITED KI5')
                AND d.d_yearmonth = 'Dec1997'
                GROUP BY c_city, s_city, d_year
                ORDER BY d_year ASC, lo_revenue DESC
            """, "Customer + Supplier + Date with specific month")
        ]
        
        self._run_query_group("Q3", q3_queries, expected_complexity="high")
    
    def run_q4_queries(self):
        """Run SSB Q4.x queries (Four dimensions - All dimensions)."""
        print(f"\n" + "="*60)
        print(f"📊 PHASE 6: SSB Q4.x QUERIES (Four Dimensions - Most Complex)")
        print(f"="*60)
        
        q4_queries = [
            ("SSB Q4.1", """
                SELECT d_year, c_nation,
                       SUM(lo_revenue - lo_supplycost) AS profit
                FROM date_dim d, customer c, supplier s, part p, lineorder lo
                WHERE lo.lo_custkey = c.c_custkey
                AND lo.lo_suppkey = s.s_suppkey
                AND lo.lo_partkey = p.p_partkey
                AND lo.lo_orderdate = d.d_datekey
                AND c.c_region = 'AMERICA'
                AND s.s_region = 'AMERICA'
                AND (p.p_mfgr = 'MFGR#1' OR p.p_mfgr = 'MFGR#2')
                GROUP BY d_year, c_nation
                ORDER BY d_year, c_nation
            """, "All dimensions with profit calculation"),
            
            ("SSB Q4.2", """
                SELECT d_year, s_nation, p_category,
                       SUM(lo_revenue - lo_supplycost) AS profit
                FROM date_dim d, customer c, supplier s, part p, lineorder lo
                WHERE lo.lo_custkey = c.c_custkey
                AND lo.lo_suppkey = s.s_suppkey
                AND lo.lo_partkey = p.p_partkey
                AND lo.lo_orderdate = d.d_datekey
                AND c.c_region = 'AMERICA'
                AND s.s_region = 'AMERICA'
                AND (d.d_year = 1997 OR d.d_year = 1998)
                AND (p.p_mfgr = 'MFGR#1' OR p.p_mfgr = 'MFGR#2')
                GROUP BY d_year, s_nation, p_category
                ORDER BY d_year, s_nation, p_category
            """, "All dimensions with year and category grouping"),
            
            ("SSB Q4.3", """
                SELECT d_year, s_city, p_brand,
                       SUM(lo_revenue - lo_supplycost) AS profit
                FROM date_dim d, customer c, supplier s, part p, lineorder lo
                WHERE lo.lo_custkey = c.c_custkey
                AND lo.lo_suppkey = s.s_suppkey
                AND lo.lo_partkey = p.p_partkey
                AND lo.lo_orderdate = d.d_datekey
                AND c.c_region = 'AMERICA'
                AND s.s_nation = 'UNITED STATES'
                AND (d.d_year = 1997 OR d.d_year = 1998)
                AND p.p_category = 'MFGR#14'
                GROUP BY d_year, s_city, p_brand
                ORDER BY d_year, s_city, p_brand
            """, "All dimensions with finest granularity")
        ]
        
        self._run_query_group("Q4", q4_queries, expected_complexity="very_high")
    
    def _run_query_group(self, group_name: str, queries: List[Tuple], expected_complexity: str):
        """Run a group of queries and collect results."""
        
        for i, (query_name, query_sql, description) in enumerate(queries, 1):
            print(f"\n🔍 {query_name}: {description}")
            print(f"   Query: {query_sql.strip()[:100]}...")
            
            try:
                # Parse the query first
                parsed = self.parser.parse(query_sql)
                print(f"   📊 Parsed query:")
                print(f"      Tables: {len(parsed['tables'])} ({', '.join(parsed['tables'])})")
                print(f"      Joins: {len(parsed['join_conditions'])}")
                print(f"      Aggregates: {len(parsed['aggregates'])}")
                print(f"      Complexity: {parsed['complexity_score']}")
                print(f"      Type: {parsed['query_type']}")
                
                # Test with both exact and sampling execution
                results = {}
                
                # Exact execution
                print(f"   🎯 Testing EXACT execution...")
                exact_start = time.time()
                exact_result = self.interceptor.execute_query(query_sql, use_sampling=False)
                exact_time = time.time() - exact_start
                
                if exact_result['success']:
                    results['exact'] = {
                        'time': exact_time,
                        'result_count': exact_result['result_count'],
                        'success': True
                    }
                    print(f"      ✅ Exact: {exact_result['result_count']} rows in {exact_time:.4f}s")
                    
                    # Show sample results
                    if exact_result['results']:
                        print(f"      Sample results:")
                        for j, row in enumerate(exact_result['results'][:2]):
                            print(f"         {j+1}: {row}")
                        if len(exact_result['results']) > 2:
                            print(f"         ... and {len(exact_result['results']) - 2} more")
                else:
                    results['exact'] = {'success': False, 'error': exact_result.get('error')}
                    print(f"      ❌ Exact failed: {exact_result.get('error')}")
                
                # Sampling execution
                print(f"   🎲 Testing SAMPLING execution...")
                sample_start = time.time()
                sample_result = self.interceptor.execute_query(query_sql, use_sampling=True)
                sample_time = time.time() - sample_start
                
                if sample_result['success']:
                    results['sampling'] = {
                        'time': sample_time,
                        'result_count': sample_result['result_count'],
                        'is_approximate': sample_result['is_approximate'],
                        'success': True,
                        'strategy': sample_result.get('strategy_used', 'unknown')
                    }
                    
                    print(f"      ✅ Sampling: {sample_result['result_count']} rows in {sample_time:.4f}s")
                    print(f"         Strategy: {sample_result.get('strategy_used', 'unknown')}")
                    print(f"         Approximate: {sample_result['is_approximate']}")
                    
                    if sample_result.get('sample_info'):
                        si = sample_result['sample_info']
                        print(f"         Sample size: {si.get('sample_size', 'N/A')}")
                        if si.get('newly_created'):
                            print(f"         ✨ New sample created: {si.get('sample_id')}")
                        elif si.get('reused_existing'):
                            print(f"         ♻️  Reused sample: {si.get('sample_id')}")
                    
                    # Show confidence intervals if available
                    if sample_result.get('confidence_interval'):
                        print(f"         📊 Has confidence intervals")
                    
                    # Performance comparison
                    if results['exact']['success']:
                        speedup = exact_time / sample_time if sample_time > 0 else float('inf')
                        print(f"         ⚡ Speedup: {speedup:.2f}x")
                        results['speedup'] = speedup
                else:
                    results['sampling'] = {'success': False, 'error': sample_result.get('error')}
                    print(f"      ⚠️  Sampling failed: {sample_result.get('error')}")
                    print(f"         (This may be expected with complex queries or small data)")
                
                # Store results for analysis
                self.test_results.append({
                    'group': group_name,
                    'query_name': query_name,
                    'description': description,
                    'complexity_score': parsed['complexity_score'],
                    'table_count': len(parsed['tables']),
                    'join_count': len(parsed['join_conditions']),
                    'results': results
                })
                
                print(f"   ✅ {query_name} testing complete")
                
            except Exception as e:
                print(f"   ❌ {query_name} failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"✅ {group_name} query group complete")
    
    def analyze_performance(self):
        """Analyze performance across all queries."""
        print(f"\n" + "="*60)
        print(f"📈 PHASE 7: PERFORMANCE ANALYSIS")
        print(f"="*60)
        
        if not self.test_results:
            print(f"⚠️  No test results to analyze")
            return
        
        # Performance summary
        total_tests = len(self.test_results)
        exact_successes = sum(1 for r in self.test_results if r['results'].get('exact', {}).get('success', False))
        sampling_successes = sum(1 for r in self.test_results if r['results'].get('sampling', {}).get('success', False))
        
        print(f"📊 Overall Performance Summary:")
        print(f"   Total queries tested: {total_tests}")
        print(f"   Exact execution success rate: {exact_successes}/{total_tests} ({exact_successes/total_tests:.1%})")
        print(f"   Sampling execution success rate: {sampling_successes}/{total_tests} ({sampling_successes/total_tests:.1%})")
        
        # Speed analysis
        speedups = [r['results'].get('speedup', 0) for r in self.test_results if 'speedup' in r['results']]
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            max_speedup = max(speedups)
            print(f"   Average speedup: {avg_speedup:.2f}x")
            print(f"   Maximum speedup: {max_speedup:.2f}x")
            print(f"   Queries with speedup: {len(speedups)}/{total_tests}")
        
        # Complexity analysis
        print(f"\n📈 Performance by Query Complexity:")
        
        # Group by complexity ranges
        complexity_groups = {
            'Low (0-30)': [],
            'Medium (31-60)': [],
            'High (61-90)': [],
            'Very High (91+)': []
        }
        
        for result in self.test_results:
            complexity = result['complexity_score']
            if complexity <= 30:
                complexity_groups['Low (0-30)'].append(result)
            elif complexity <= 60:
                complexity_groups['Medium (31-60)'].append(result)
            elif complexity <= 90:
                complexity_groups['High (61-90)'].append(result)
            else:
                complexity_groups['Very High (91+)'].append(result)
        
        for group_name, group_results in complexity_groups.items():
            if group_results:
                group_sampling_success = sum(1 for r in group_results if r['results'].get('sampling', {}).get('success', False))
                group_speedups = [r['results'].get('speedup', 0) for r in group_results if 'speedup' in r['results']]
                avg_group_speedup = sum(group_speedups) / len(group_speedups) if group_speedups else 0
                
                print(f"   {group_name}: {len(group_results)} queries")
                print(f"      Sampling success: {group_sampling_success}/{len(group_results)}")
                if avg_group_speedup > 0:
                    print(f"      Average speedup: {avg_group_speedup:.2f}x")
        
        # Query group analysis
        print(f"\n📊 Performance by SSB Query Group:")
        query_groups = {}
        for result in self.test_results:
            group = result['group']
            if group not in query_groups:
                query_groups[group] = []
            query_groups[group].append(result)
        
        for group_name, group_results in query_groups.items():
            group_sampling_success = sum(1 for r in group_results if r['results'].get('sampling', {}).get('success', False))
            group_speedups = [r['results'].get('speedup', 0) for r in group_results if 'speedup' in r['results']]
            avg_group_speedup = sum(group_speedups) / len(group_speedups) if group_speedups else 0
            
            print(f"   {group_name} queries: {len(group_results)} queries")
            print(f"      Sampling success: {group_sampling_success}/{len(group_results)}")
            if avg_group_speedup > 0:
                print(f"      Average speedup: {avg_group_speedup:.2f}x")
        
        print(f"✅ Performance analysis complete")
    
    def show_final_statistics(self):
        """Show final system statistics."""
        print(f"\n" + "="*60)
        print(f"📊 PHASE 8: FINAL SYSTEM STATISTICS")
        print(f"="*60)
        
        try:
            # Get comprehensive system statistics
            stats = self.interceptor.get_execution_statistics()
            
            print(f"🎯 Query Execution Statistics:")
            query_stats = stats['query_execution']
            print(f"   Total queries processed: {query_stats['total_queries']}")
            print(f"   Total execution time: {query_stats['total_execution_time']:.3f}s")
            print(f"   Average execution time: {query_stats['average_execution_time']:.3f}s")
            print(f"   Sampling usage count: {query_stats['sampling_usage_count']}")
            print(f"   Sampling usage rate: {query_stats['sampling_usage_rate']:.1%}")
            
            print(f"\n💾 Sample Storage Statistics:")
            storage_stats = stats['sample_storage']
            print(f"   Materialized samples: {storage_stats['materialized_samples']}")
            print(f"   Total sample records: {storage_stats['total_sample_size']:,}")
            
            # List materialized samples created during test
            samples = self.interceptor.sample_storage.list_materialized_samples()
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
                    print(f"      Used: {query_count} times")
            
            print(f"\n⚙️  System Configuration:")
            config = stats['configuration']
            print(f"   Complexity threshold: {config['complexity_threshold']}")
            print(f"   Sample size range: {config['min_sample_size']}-{config['max_sample_size']}")
            print(f"   Confidence level: {config['confidence_level']}")
            
        except Exception as e:
            print(f"❌ Failed to get final statistics: {e}")
        
        print(f"✅ Final statistics complete")
    
    def cleanup(self):
        """Clean up test resources."""
        print(f"\n" + "="*60)
        print(f"🧹 CLEANUP & SUMMARY")
        print(f"="*60)
        
        try:
            # Final summary
            if self.test_results:
                successful_tests = sum(1 for r in self.test_results 
                                     if r['results'].get('exact', {}).get('success', False) or 
                                        r['results'].get('sampling', {}).get('success', False))
                
                print(f"📊 BENCHMARK SUMMARY:")
                print(f"   Total SSB queries tested: {len(self.test_results)}")
                print(f"   Successful executions: {successful_tests}/{len(self.test_results)}")
                
                # Show which queries worked best with sampling
                sampling_wins = [r for r in self.test_results if r['results'].get('speedup', 0) > 1.5]
                if sampling_wins:
                    print(f"   Queries with significant speedup (>1.5x): {len(sampling_wins)}")
                    for win in sampling_wins[:3]:  # Show top 3
                        speedup = win['results']['speedup']
                        print(f"      {win['query_name']}: {speedup:.2f}x speedup")
            
            # Clean up interceptor resources
            self.interceptor.cleanup_resources()
            
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")

def main():
    """Run the SSB benchmark test suite."""
    print("📊 STAR SCHEMA BENCHMARK (SSB) TEST SUITE")
    print("=" * 80)
    print("Testing Adaptive Sampling System with Standard SSB Queries")
    print("")
    print("This benchmark tests queries of increasing complexity:")
    print("• Q1.x: Single dimension queries (low-medium complexity)")
    print("• Q2.x: Two dimension queries (medium-high complexity)")  
    print("• Q3.x: Three dimension queries (high complexity)")
    print("• Q4.x: Four dimension queries (very high complexity)")
    print("")
    print("Prerequisites: SSB data must already exist in database")
    print("Run generate_data.py first if needed")
    print("=" * 80)
    
    # Run the benchmark
    benchmark = SSBBenchmarkTest()
    benchmark.run_benchmark_suite()

if __name__ == "__main__":
    main()