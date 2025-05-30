import time
import math
import re
import uuid
from typing import Dict, List, Tuple, Optional, Any, Union
from database.connector import DatabaseConnector
from database.schema import SSBSchema
from query.parser import SQLQueryParser
from query.adaptive_engine import AdaptiveDecisionEngine
from sampling.wander_join import WanderJoin
from sampling.sample_storage import SampleStorage

class QueryInterceptor:
    """
    Main query execution engine that intercepts queries and decides whether
    to execute them exactly or using samples. Handles complete query rewriting
    and result scaling with adaptive learning.
    """
    
    def __init__(self, db_connector=None):
        """Initialize the query interceptor with all necessary components."""
        self.db = db_connector if db_connector else DatabaseConnector()
        self.parser = SQLQueryParser()
        self.wander_join = WanderJoin(self.db)
        self.sample_storage = SampleStorage(self.db)
        
        # NEW: Adaptive decision engine replaces static thresholds
        self.decision_engine = AdaptiveDecisionEngine(self.db)
        
        # Keep these for backward compatibility and fallback
        self.min_sample_size = 100
        self.max_sample_size = 1000
        self.confidence_level = 0.95
        
        # Performance tracking
        self.query_count = 0
        self.total_execution_time = 0
        self.sampling_usage_count = 0
        
        # Store last decision for reference
        self._last_decision = None
        
    def execute_query(self, query: str, use_sampling: Optional[bool] = None, 
                     return_metadata: bool = True) -> Dict:
        """
        Main entry point for query execution with intelligent routing.
        
        Parameters:
        - query: SQL query string
        - use_sampling: Force sampling (True), exact (False), or auto-decide (None)
        - return_metadata: Include execution metadata in results
        
        Returns:
        - Dictionary with results, metadata, and performance statistics
        """
        start_time = time.time()
        self.query_count += 1
        
        try:
            print(f"\n🎯 INTERCEPTING QUERY #{self.query_count}")
            print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            # Step 1: Parse and analyze the query
            print(f"\n📋 PARSING & ANALYSIS...")
            parsed_query = self.parser.parse(query)
            self._log_query_analysis(parsed_query)
            
            # Step 2: Make intelligent execution decision
            execution_strategy = self._decide_execution_strategy(parsed_query, use_sampling)
            print(f"\n⚡ EXECUTION STRATEGY: {execution_strategy}")
            
            # Step 3: Execute based on strategy
            if execution_strategy == "EXACT":
                result = self._execute_exact_query(query, parsed_query)
            else:  # SAMPLING
                result = self._execute_sample_query(query, parsed_query)
            
            # Step 4: Add execution metadata
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            
            if return_metadata:
                result['execution_metadata'] = {
                    'query_number': self.query_count,
                    'strategy': execution_strategy,
                    'execution_time': execution_time,
                    'query_complexity': parsed_query['complexity_score'],
                    'query_type': parsed_query['query_type'],
                    'tables_involved': parsed_query['tables'],
                    'join_count': len(parsed_query['join_conditions'])
                }
            
            print(f"\n✅ QUERY COMPLETED in {execution_time:.3f} seconds")
            return result
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"\n❌ QUERY FAILED after {error_time:.3f} seconds: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': error_time,
                'results': [],
                'query_number': self.query_count
            }
    
    def _log_query_analysis(self, parsed_query: Dict):
        """Log detailed query analysis."""
        print(f"   📊 Query Analysis:")
        print(f"      Tables: {parsed_query['tables']}")
        print(f"      Query Type: {parsed_query['query_type']}")
        print(f"      Complexity Score: {parsed_query['complexity_score']}")
        print(f"      Join Conditions: {len(parsed_query['join_conditions'])}")
        print(f"      Aggregates: {len(parsed_query['aggregates'])}")
        print(f"      WHERE Conditions: {len(parsed_query['where_conditions'])}")
        if parsed_query['group_by_columns']:
            print(f"      GROUP BY: {parsed_query['group_by_columns']}")
    
    def _decide_execution_strategy(self, parsed_query: Dict, force_sampling: Optional[bool]) -> str:
        """
        Intelligent execution strategy decision using adaptive engine.
        """
        
        # Use the adaptive decision engine instead of static rules
        decision = self.decision_engine.should_use_sampling(parsed_query, force_sampling)
        
        # Store decision for later use
        self._last_decision = decision
        
        return "SAMPLING" if decision['use_sampling'] else "EXACT"
    
    def _execute_exact_query(self, query: str, parsed_query: Dict) -> Dict:
        """Execute query exactly on base tables."""
        print(f"   🎯 Executing EXACT query on base tables...")
        
        start_time = time.time()
        results = self.db.execute_query(query)
        execution_time = time.time() - start_time
        
        if results is None:
            results = []
        
        print(f"   ✅ Exact execution: {len(results)} rows in {execution_time:.3f}s")
        
        result = {
            'success': True,
            'results': results,
            'result_count': len(results),
            'is_approximate': False,
            'confidence_interval': None,
            'sample_info': None,
            'exact_execution_time': execution_time,
            'strategy_used': 'exact'
        }
        
        # Record for learning
        self._record_execution_for_learning(parsed_query, result)
        
        return result
    
    def _execute_sample_query(self, query: str, parsed_query: Dict) -> Dict:
        """Execute query using sampling with intelligent sample management."""
        print(f"   🎲 Executing with SAMPLING...")
        
        tables = parsed_query['tables']
        join_conditions = parsed_query.get('join_conditions', [])
        
        # Convert join conditions to WanderJoin format
        wj_joins = self.parser.format_joins_for_sampling()  # ✅ CORRECT        
        print(f"   📊 Query requirements:")
        print(f"      Tables: {tables}")
        print(f"      Joins: {len(wj_joins)} join conditions")
        
        # Step 1: Check for existing materialized samples
        sample_match = self.sample_storage.find_best_sample_match(tables, wj_joins)
        
        if sample_match:
            return self._execute_with_existing_sample(query, parsed_query, sample_match)
        else:
            return self._execute_with_new_sample(query, parsed_query, tables, wj_joins)
    
    def _execute_with_existing_sample(self, query: str, parsed_query: Dict, sample_match: Dict) -> Dict:
        """Execute query using an existing materialized sample."""
        
        if sample_match['exact_match']:
            # Perfect match - use the sample directly
            sample_id = sample_match['sample_id']
            print(f"   ✨ Using EXACT sample match: {sample_id}")
            
            # Rewrite query to use the sample table
            rewritten_query = self._rewrite_query_for_sample(query, parsed_query, sample_id)
            
            start_time = time.time()
            sample_results = self.db.execute_query(rewritten_query)
            sample_execution_time = time.time() - start_time
            
            if sample_results is None:
                sample_results = []
            
            print(f"   ✅ Sample query executed: {len(sample_results)} rows in {sample_execution_time:.3f}s")
            
        else:
            # Partial match - extend the sample
            print(f"   🔧 EXTENDING sample for additional tables...")
            print(f"      Base sample: {sample_match['sample_id']}")
            print(f"      Coverage: {sample_match['coverage']:.1%}")
            print(f"      Additional tables: {sample_match['additional_tables']}")
            
            try:
                extended_sample_id = self.sample_storage.extend_materialized_sample(
                    sample_match, 
                    target_sample_size=self.min_sample_size
                )
                
                print(f"   ✅ Sample extended: {extended_sample_id}")
                
                # Rewrite query to use extended sample
                rewritten_query = self._rewrite_query_for_sample(query, parsed_query, extended_sample_id)
                
                start_time = time.time()
                sample_results = self.db.execute_query(rewritten_query)
                sample_execution_time = time.time() - start_time
                
                if sample_results is None:
                    sample_results = []
                
            except Exception as e:
                print(f"   ⚠️  Sample extension failed: {e}")
                # Fallback to creating new sample
                return self._execute_with_new_sample(
                    query, parsed_query, 
                    parsed_query['tables'], 
                    self.parser.format_joins_for_sampling()
                )
        
        # Scale results and calculate confidence intervals
        scaled_results = self._scale_sample_results(
            sample_results, 
            parsed_query, 
            sample_match.get('sample_size', self.min_sample_size)
        )
        
        self.sampling_usage_count += 1
        
        result = {
            'success': True,
            'results': scaled_results['results'],
            'result_count': len(scaled_results['results']) if scaled_results['results'] else 0,
            'is_approximate': True,
            'confidence_interval': scaled_results['confidence_interval'],
            'sample_info': {
                'sample_id': sample_match['sample_id'],
                'sample_size': sample_match.get('sample_size', 'unknown'),
                'coverage': sample_match.get('coverage', 1.0),
                'exact_match': sample_match['exact_match'],
                'reused_existing': True
            },
            'sample_execution_time': sample_execution_time,
            'strategy_used': 'sampling_reuse'
        }
        
        # Record for learning
        self._record_execution_for_learning(parsed_query, result)
        
        return result
    
    def _execute_with_new_sample(self, query: str, parsed_query: Dict, tables: List[str], joins: List[Tuple]) -> Dict:
        """Execute query by creating a new sample with intelligent materialization."""
        
        if len(tables) < 2:
            print(f"   ⚠️  Single table query - falling back to exact execution")
            return self._execute_exact_query(query, parsed_query)
        
        if not joins:
            print(f"   ⚠️  No join conditions found - falling back to exact execution")
            return self._execute_exact_query(query, parsed_query)
        
        # Determine sample size based on adaptive complexity
        if self._last_decision and 'data_complexity' in self._last_decision:
            complexity = self._last_decision['data_complexity']
            sample_size = min(self.max_sample_size, max(self.min_sample_size, int(complexity * 50)))
        else:
            # Fallback to old method
            sample_size = min(self.max_sample_size, 
                             max(self.min_sample_size, parsed_query['complexity_score'] * 2))
        
        print(f"   🎯 Creating NEW sample of size {sample_size}")
        
        try:
            # Create sample using WanderJoin
            start_time = time.time()
            samples = self.wander_join.sample_join(tables, joins, sample_size=sample_size)
            sampling_time = time.time() - start_time
            
            if not samples:
                print(f"   ⚠️  Sampling failed - falling back to exact execution")
                return self._execute_exact_query(query, parsed_query)
            
            print(f"   ✅ Created {len(samples)} samples in {sampling_time:.3f}s")
            
            # INTELLIGENT MATERIALIZATION DECISION
            materialization_decision = self.decision_engine.should_materialize_sample(
                parsed_query, sampling_time, len(samples)
            )
            
            if materialization_decision['materialize']:
                # Materialize the sample for reuse
                print(f"   💾 MATERIALIZING sample: {materialization_decision['reasoning']}")
                
                sample_id = self.sample_storage.materialize_sample(samples, tables, joins, query)
                
                # Rewrite and execute query on sample
                rewritten_query = self._rewrite_query_for_sample(query, parsed_query, sample_id)
                
                start_time = time.time()
                sample_results = self.db.execute_query(rewritten_query)
                sample_execution_time = time.time() - start_time
                
                if sample_results is None:
                    sample_results = []
                
                sample_info = {
                    'sample_id': sample_id,
                    'sample_size': len(samples),
                    'coverage': 1.0,
                    'exact_match': False,
                    'newly_created': True,
                    'materialized': True,
                    'materialization_reasoning': materialization_decision['reasoning']
                }
                
            else:
                # DON'T MATERIALIZE: Use sample once and discard
                print(f"   🗑️  NOT MATERIALIZING: {materialization_decision['reasoning']}")
                
                # Execute directly on in-memory sample
                sample_results = self._execute_on_memory_sample(samples, parsed_query)
                sample_execution_time = 0.001  # Very fast for in-memory
                
                sample_info = {
                    'sample_id': None,
                    'sample_size': len(samples),
                    'coverage': 1.0,
                    'exact_match': False,
                    'newly_created': True,
                    'materialized': False,
                    'materialization_reasoning': materialization_decision['reasoning']
                }
            
            print(f"   ✅ Sample query executed: {len(sample_results) if sample_results else 0} rows in {sample_execution_time:.3f}s")
            
            # Scale results
            scaled_results = self._scale_sample_results(sample_results, parsed_query, len(samples))
            
            self.sampling_usage_count += 1
            
            result = {
                'success': True,
                'results': scaled_results['results'],
                'result_count': len(scaled_results['results']) if scaled_results['results'] else 0,
                'is_approximate': True,
                'confidence_interval': scaled_results['confidence_interval'],
                'sample_info': sample_info,
                'sampling_time': sampling_time,
                'sample_execution_time': sample_execution_time,
                'strategy_used': 'sampling_new',
                'decision_metadata': {
                    'sampling_decision': self._last_decision,
                    'materialization_decision': materialization_decision
                }
            }
            
            # Record for learning
            self._record_execution_for_learning(parsed_query, result)
            
            return result
            
        except Exception as e:
            print(f"   ❌ Sampling failed: {e}")
            print(f"   🔄 Falling back to exact execution")
            fallback_result = self._execute_exact_query(query, parsed_query)
            
            # Record the fallback for learning too
            self._record_execution_for_learning(parsed_query, fallback_result)
            
            return fallback_result
    
    def _execute_on_memory_sample(self, samples: List[Dict], parsed_query: Dict) -> List[Tuple]:
        """
        Execute query directly on in-memory sample data without materializing.
        This is used when materialization is not beneficial.
        """
        
        # For simple aggregations, we can compute directly on the sample data
        aggregates = parsed_query.get('aggregates', [])
        
        if not aggregates:
            # Non-aggregate query: return sample data as-is
            results = []
            for sample in samples:
                # Convert sample dict to tuple (order matters)
                row = tuple(sample.values())
                results.append(row)
            return results
        
        # Handle aggregate queries
        if len(aggregates) == 1 and aggregates[0]['function'] in ['COUNT', 'SUM']:
            agg = aggregates[0]
            
            if agg['function'] == 'COUNT':
                if agg['argument'] == '*':
                    return [(len(samples),)]
                else:
                    # Count non-null values for specific column
                    count = sum(1 for sample in samples if sample.get(agg['argument']) is not None)
                    return [(count,)]
            
            elif agg['function'] == 'SUM':
                # Sum a specific column
                total = sum(sample.get(agg['argument'], 0) for sample in samples)
                return [(total,)]
        
        # For complex aggregations, fall back to creating a temporary table
        print(f"   ⚠️  Complex aggregation on in-memory sample - using temporary table")
        
        # Create a temporary table just for this query
        temp_table_id = f"temp_sample_{uuid.uuid4().hex[:8]}"
        
        # Create temporary table with sample data
        if samples:
            columns = list(samples[0].keys())
            clean_columns = [col.replace('.', '_') for col in columns]
            schema = ", ".join([f"{col} VARCHAR" for col in clean_columns])
            
            self.db.create_table(temp_table_id, schema)
            
            # Insert sample data
            for sample in samples:
                values = [str(val) if val is not None else "NULL" for val in sample.values()]
                placeholders = ", ".join(["?"] * len(values))
                insert_query = f"INSERT INTO {temp_table_id} VALUES ({placeholders})"
                self.db.execute_query(insert_query, tuple(values))
            
            # Execute query on temporary table
            rewritten_query = self._rewrite_query_for_sample("SELECT * FROM temp", parsed_query, temp_table_id)
            results = self.db.execute_query(rewritten_query)
            
            # Clean up temporary table
            self.db.drop_table(temp_table_id)
            
            return results
        
        return []
    
    def _record_execution_for_learning(self, parsed_query: Dict, result: Dict):
        """Record execution results for the learning system."""
        
        if not result['success']:
            return
        
        # Extract timing information
        exact_time = None
        sampling_time = None
        sampling_success = False
        
        if result.get('strategy_used') == 'exact':
            exact_time = result.get('exact_execution_time')
        elif result.get('is_approximate'):
            # This was a sampling execution
            sampling_time = result.get('sampling_time', 0) + result.get('sample_execution_time', 0)
            sampling_success = True
        
        # Record for learning
        self.decision_engine.record_execution(
            parsed_query=parsed_query,
            exact_time=exact_time,
            sampling_time=sampling_time,
            sampling_success=sampling_success
        )
    
    def _rewrite_query_for_sample(self, original_query: str, parsed_query: Dict, sample_id: str) -> str:
        """
        Advanced query rewriting to execute on sample tables.
        
        This is the most complex part - transforms the original query to work
        with the materialized sample table while preserving query semantics.
        """
        
        print(f"   📝 REWRITING query for sample: {sample_id}")
        
        tables = parsed_query['tables']
        aggregates = parsed_query.get('aggregates', [])
        group_by = parsed_query.get('group_by_columns', [])
        where_conditions = parsed_query.get('where_conditions', [])
        order_by = parsed_query.get('order_by_columns', [])
        limit = parsed_query.get('limit')
        
        # Get sample table schema to understand available columns
        sample_columns_info = self.db.execute_query(f"PRAGMA table_info({sample_id})")
        available_columns = [col[1] for col in sample_columns_info]  # col[1] is column name
        
        print(f"      Sample columns: {len(available_columns)} columns available")
        
        # Build the rewritten query
        rewritten_parts = []
        
        # SELECT clause
        if aggregates:
            # Handle aggregate queries
            select_items = []
            for agg in aggregates:
                func = agg['function']
                arg = agg['argument']
                
                # Map original column names to sample column names
                if arg != '*':
                    # Convert table.column to table_column format used in sample
                    sample_arg = self._map_column_to_sample_format(arg, available_columns)
                    if sample_arg:
                        if agg.get('alias'):
                            select_items.append(f"{func}({sample_arg}) AS {agg['alias']}")
                        else:
                            select_items.append(f"{func}({sample_arg})")
                    else:
                        print(f"      ⚠️ Warning: Could not map aggregate argument: {arg}")
                else:
                    # COUNT(*) case
                    if agg.get('alias'):
                        select_items.append(f"COUNT(*) AS {agg['alias']}")
                    else:
                        select_items.append("COUNT(*)")
            
            # Add non-aggregate columns (for GROUP BY)
            for col in parsed_query.get('select_columns', []):
                if not any(col in agg.get('full_expression', '') for agg in aggregates):
                    sample_col = self._map_column_to_sample_format(col, available_columns)
                    if sample_col:
                        select_items.append(sample_col)
            
            select_clause = "SELECT " + ", ".join(select_items) if select_items else "SELECT *"
        else:
            # Non-aggregate query - select all or specific columns
            if parsed_query.get('select_columns'):
                # Map specific columns
                select_items = []
                for col in parsed_query['select_columns']:
                    sample_col = self._map_column_to_sample_format(col, available_columns)
                    if sample_col:
                        select_items.append(sample_col)
                    else:
                        print(f"      ⚠️ Warning: Could not map column: {col}")
            
                select_clause = "SELECT " + ", ".join(select_items) if select_items else "SELECT *"
            else:
                select_clause = "SELECT *"
        
        rewritten_parts.append(select_clause)
        
        # FROM clause - always the sample table
        rewritten_parts.append(f"FROM {sample_id}")
        
        # WHERE clause - convert column references
        if where_conditions:
            mapped_where_conditions = []
            for condition in where_conditions:
                mapped_condition = self._map_where_condition_to_sample(condition, available_columns)
                if mapped_condition:
                    mapped_where_conditions.append(mapped_condition)
        
            if mapped_where_conditions:
                rewritten_parts.append("WHERE " + " AND ".join(mapped_where_conditions))
        
        # GROUP BY clause
        if group_by:
            mapped_group_by = []
            for col in group_by:
                sample_col = self._map_column_to_sample_format(col, available_columns)
                if sample_col:
                    mapped_group_by.append(sample_col)
        
            if mapped_group_by:
                rewritten_parts.append("GROUP BY " + ", ".join(mapped_group_by))
        
        # ORDER BY clause
        if order_by:
            mapped_order_by = []
            for order_item in order_by:
                col = order_item['column']
                direction = order_item.get('direction', 'ASC')
                
                # First, check if this column is an aggregate alias
                is_aggregate_alias = False
                for agg in aggregates:
                    if agg.get('alias') and agg['alias'] == col:
                        # This is an aggregate alias - use it directly
                        mapped_order_by.append(f"{col} {direction}")
                        is_aggregate_alias = True
                        break
                
                if not is_aggregate_alias:
                    # Check if it's a GROUP BY column that needs mapping
                    sample_col = self._map_column_to_sample_format(col, available_columns)
                    if sample_col:
                        mapped_order_by.append(f"{sample_col} {direction}")
        
            if mapped_order_by:
                rewritten_parts.append("ORDER BY " + ", ".join(mapped_order_by))
        
        # LIMIT clause
        if limit:
            rewritten_parts.append(f"LIMIT {limit}")
        
        rewritten_query = " ".join(rewritten_parts)
        
        print(f"      Original: {original_query[:60]}...")
        print(f"      Rewritten: {rewritten_query[:60]}...")
        
        return rewritten_query
    
    def _map_column_to_sample_format(self, original_column: str, available_columns: List[str]) -> Optional[str]:
        """
        Map original column reference to sample table column format.
        
        Original format: 'table.column' or 'column'
        Sample format: 'table_column'
        """
        
        # Direct match (already in sample format)
        if original_column in available_columns:
            return original_column
        
        # Convert table.column to table_column
        if '.' in original_column:
            sample_format = original_column.replace('.', '_')
            if sample_format in available_columns:
                return sample_format
        
        # Try to find partial matches
        for available_col in available_columns:
            # Check if original column name appears in available column
            if original_column.lower() in available_col.lower():
                return available_col
            
            # Check if available column ends with the original column
            if available_col.endswith('_' + original_column):
                return available_col
        
        return None
    
    def _map_where_condition_to_sample(self, condition: str, available_columns: List[str]) -> Optional[str]:
        """
        Map WHERE condition column references to sample format.
        """
        
        # This is a simplified implementation
        # In production, you'd want more sophisticated SQL parsing
        
        mapped_condition = condition
        
        # Find column references in the condition and replace them
        # Look for patterns like table.column
        table_column_pattern = r'(\w+)\.(\w+)'
        matches = re.findall(table_column_pattern, condition)
        
        for table, column in matches:
            original_ref = f"{table}.{column}"
            sample_ref = self._map_column_to_sample_format(original_ref, available_columns)
            if sample_ref:
                mapped_condition = mapped_condition.replace(original_ref, sample_ref)
        
        return mapped_condition if mapped_condition != condition else None
    
    def _scale_sample_results(self, sample_results: List, parsed_query: Dict, sample_size: int) -> Dict:
        """
        Scale sample results back to population estimates and calculate confidence intervals.
        """
        
        if not sample_results or not parsed_query.get('aggregates'):
            # No scaling needed for non-aggregate queries
            return {
                'results': sample_results,
                'confidence_interval': None
            }
        
        print(f"   📊 SCALING results from {sample_size} samples to population estimates...")
        
        # Get rough population size estimate
        estimated_population_size = self._estimate_population_size(parsed_query['tables'])
        scaling_factor = estimated_population_size / sample_size
        
        print(f"      Population estimate: {estimated_population_size:,}")
        print(f"      Scaling factor: {scaling_factor:.2f}")
        
        scaled_results = []
        confidence_intervals = []
        
        for row in sample_results:
            scaled_row = []
            row_confidence = []
            
            for i, value in enumerate(row):
                if isinstance(value, (int, float)) and value is not None:
                    # Scale numeric values (assumes they're aggregates)
                    scaled_value = value * scaling_factor
                    
                    # Calculate confidence interval (simplified statistical approach)
                    if sample_size > 30:  # Use normal distribution
                        margin_of_error = 1.96 * math.sqrt(scaled_value) / math.sqrt(sample_size)
                    else:  # Use t-distribution approximation
                        margin_of_error = 2.0 * math.sqrt(scaled_value) / math.sqrt(sample_size)
                    
                    lower_bound = max(0, scaled_value - margin_of_error)
                    upper_bound = scaled_value + margin_of_error
                    
                    scaled_row.append(int(scaled_value))
                    row_confidence.append((int(lower_bound), int(upper_bound)))
                else:
                    # Keep non-numeric values as-is (like region names, etc.)
                    scaled_row.append(value)
                    row_confidence.append(None)
            
            scaled_results.append(tuple(scaled_row))
            confidence_intervals.append(row_confidence)
        
        print(f"      ✅ Scaled {len(scaled_results)} result rows")
        
        return {
            'results': scaled_results,
            'confidence_interval': confidence_intervals,
            'scaling_factor': scaling_factor,
            'population_estimate': estimated_population_size
        }
    
    def _estimate_population_size(self, tables: List[str]) -> int:
        """
        Estimate the size of the full join result.
        Uses table statistics and join selectivity estimates.
        """
        
        print(f"      📈 Estimating population size for {len(tables)} tables...")
        
        # Get row counts for each table
        table_sizes = {}
        for table in tables:
            try:
                result = self.db.execute_query(f"SELECT COUNT(*) FROM {table}")
                if result:
                    table_size = result[0][0]
                    table_sizes[table] = table_size
                    print(f"         {table}: {table_size:,} rows")
            except:
                # Fallback estimate based on SSB typical sizes
                if table == 'lineorder':
                    table_sizes[table] = 6000000
                elif table == 'customer':
                    table_sizes[table] = 30000
                elif table == 'supplier':
                    table_sizes[table] = 2000
                elif table == 'part':
                    table_sizes[table] = 200000
                elif table == 'date_dim':
                    table_sizes[table] = 2556
                else:
                    table_sizes[table] = 100000
        
        if len(tables) == 1:
            return table_sizes[tables[0]]
        
        # For joins, estimate result size using selectivity factors
        # This is a simplified heuristic - real systems use detailed statistics
        
        if len(tables) == 2:
            # Two-table join
            larger_table = max(table_sizes.values())
            selectivity = 0.1  # Assume 10% selectivity for foreign key joins
            estimated_size = int(larger_table * selectivity)
        else:
            # Multi-table join
            fact_table_size = table_sizes.get('lineorder', max(table_sizes.values()))
            # Multi-table joins typically have high selectivity
            selectivity = 0.01 * len(tables)  # Decreases with more tables
            estimated_size = int(fact_table_size * selectivity)
        
        # Ensure reasonable bounds
        estimated_size = max(1000, min(estimated_size, sum(table_sizes.values()) // 10))
        
        print(f"         Estimated result size: {estimated_size:,} rows")
        return estimated_size
    
    def get_execution_statistics(self) -> Dict:
        """Get comprehensive execution statistics including learning progress."""
        
        avg_execution_time = self.total_execution_time / max(1, self.query_count)
        sampling_usage_rate = self.sampling_usage_count / max(1, self.query_count)
        
        # Get sample storage statistics
        samples = self.sample_storage.list_materialized_samples()
        
        # Get learning statistics
        learning_stats = self.decision_engine.get_learning_statistics()
        
        return {
            'query_execution': {
                'total_queries': self.query_count,
                'total_execution_time': self.total_execution_time,
                'average_execution_time': avg_execution_time,
                'sampling_usage_count': self.sampling_usage_count,
                'sampling_usage_rate': sampling_usage_rate
            },
            'sample_storage': {
                'materialized_samples': len(samples),
                'total_sample_size': sum(sample[3] for sample in samples) if samples else 0
            },
            'learning_stats': learning_stats,
            'configuration': {
                'min_sample_size': self.min_sample_size,
                'max_sample_size': self.max_sample_size,
                'confidence_level': self.confidence_level
            }
        }
    
    def cleanup_resources(self):
        """Clean up database connections and temporary resources."""
        print(f"\n🧹 CLEANING UP RESOURCES...")
        
        # Get final statistics
        stats = self.get_execution_statistics()
        print(f"   📊 Final Statistics:")
        print(f"      Total queries processed: {stats['query_execution']['total_queries']}")
        print(f"      Sampling usage rate: {stats['query_execution']['sampling_usage_rate']:.1%}")
        print(f"      Average execution time: {stats['query_execution']['average_execution_time']:.3f}s")
        
        # Close database connection
        if self.db:
            self.db.close()
            print(f"   ✅ Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup_resources()