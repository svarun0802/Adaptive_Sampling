"""
Adaptive Decision Engine for Query Sampling and Materialization

This module provides intelligent decision-making for:
1. When to use sampling vs exact execution
2. When to materialize samples vs use in-memory
3. Learning from execution patterns over time
"""

import time
import json
import hashlib
import math
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
from database.connector import DatabaseConnector

class AdaptiveDecisionEngine:
    """
    Intelligent decision engine that learns from query execution patterns
    to make optimal sampling and materialization decisions.
    """
    
    def __init__(self, db_connector=None):
        self.db = db_connector if db_connector else DatabaseConnector()
        
        # Performance learning system
        self.execution_history = deque(maxlen=1000)  # Recent execution history
        self.query_patterns = defaultdict(list)      # Pattern -> execution times
        self.table_statistics = {}                   # Cached table statistics
        
        # Query frequency tracking
        self.query_fingerprints = defaultdict(int)   # fingerprint -> frequency
        self.query_recency = defaultdict(float)      # fingerprint -> last_seen_time
        self.similar_queries = defaultdict(list)     # fingerprint -> similar queries
        
        # Adaptive thresholds (these learn over time)
        self.sampling_threshold = 0.5                # Probability threshold for sampling
        self.materialization_threshold = 0.3         # Probability threshold for materialization
        self.min_speedup_required = 1.2             # Minimum speedup to justify sampling
        
        # System configuration
        self.learning_enabled = True
        self.history_window = 3600  # 1 hour window for pattern analysis
        
    def should_use_sampling(self, parsed_query: Dict, force_decision: Optional[bool] = None) -> Dict:
        """
        Intelligent sampling decision based on learned patterns and real-time analysis.
        
        Returns:
        - Dictionary with decision, confidence, and reasoning
        """
        if force_decision is not None:
            return {
                'use_sampling': force_decision,
                'confidence': 1.0,
                'reasoning': f"User forced {'sampling' if force_decision else 'exact'}",
                'decision_type': 'forced'
            }
        
        print(f"🤔 MAKING SAMPLING DECISION...")
        
        # Step 1: Calculate data-driven complexity
        data_complexity = self._calculate_data_driven_complexity(parsed_query)
        
        # Step 2: Estimate execution costs
        exact_cost_estimate = self._estimate_exact_execution_cost(parsed_query)
        sampling_cost_estimate = self._estimate_sampling_cost(parsed_query)
        
        # Step 3: Look for similar query patterns
        pattern_evidence = self._analyze_query_patterns(parsed_query)
        
        # Step 4: Make probabilistic decision
        sampling_probability = self._calculate_sampling_probability(
            data_complexity, exact_cost_estimate, sampling_cost_estimate, pattern_evidence
        )
        
        # Decision based on probability
        use_sampling = sampling_probability > self.sampling_threshold
        
        # Build reasoning
        reasoning_parts = []
        reasoning_parts.append(f"Data complexity: {data_complexity:.2f}")
        reasoning_parts.append(f"Exact cost estimate: {exact_cost_estimate:.3f}s")
        reasoning_parts.append(f"Sampling cost estimate: {sampling_cost_estimate:.3f}s")
        reasoning_parts.append(f"Sampling probability: {sampling_probability:.3f}")
        
        if pattern_evidence['similar_queries'] > 0:
            reasoning_parts.append(f"Found {pattern_evidence['similar_queries']} similar queries")
            reasoning_parts.append(f"Average speedup from sampling: {pattern_evidence['avg_speedup']:.2f}x")
        
        decision = {
            'use_sampling': use_sampling,
            'confidence': abs(sampling_probability - 0.5) * 2,  # 0 = uncertain, 1 = very confident
            'reasoning': '; '.join(reasoning_parts),
            'decision_type': 'learned',
            'sampling_probability': sampling_probability,
            'data_complexity': data_complexity,
            'cost_estimates': {
                'exact': exact_cost_estimate,
                'sampling': sampling_cost_estimate
            }
        }
        
        print(f"   Decision: {'SAMPLING' if use_sampling else 'EXACT'}")
        print(f"   Confidence: {decision['confidence']:.2f}")
        print(f"   Probability: {sampling_probability:.3f}")
        
        return decision
    
    def should_materialize_sample(self, parsed_query: Dict, sample_creation_time: float, 
                                sample_size: int) -> Dict:
        """
        Intelligent materialization decision based on query patterns and cost-benefit analysis.
        """
        print(f"🤔 MAKING MATERIALIZATION DECISION...")
        
        # Step 1: Query frequency analysis
        query_fingerprint = self._create_query_fingerprint(parsed_query)
        frequency_score = self._calculate_frequency_score(query_fingerprint)
        
        # Step 2: Pattern similarity analysis
        similarity_score = self._calculate_pattern_similarity_score(parsed_query)
        
        # Step 3: Cost-benefit analysis
        storage_cost = self._estimate_storage_cost(sample_size, parsed_query['tables'])
        reuse_benefit = self._estimate_reuse_benefit(sample_creation_time, frequency_score, similarity_score)
        
        # Step 4: Materialization probability
        materialization_probability = self._calculate_materialization_probability(
            frequency_score, similarity_score, storage_cost, reuse_benefit, sample_creation_time
        )
        
        # Decision
        should_materialize = materialization_probability > self.materialization_threshold
        
        # Reasoning
        reasoning_parts = []
        reasoning_parts.append(f"Frequency score: {frequency_score:.2f}")
        reasoning_parts.append(f"Similarity score: {similarity_score:.2f}")
        reasoning_parts.append(f"Storage cost: {storage_cost:.2f}MB")
        reasoning_parts.append(f"Reuse benefit: {reuse_benefit:.2f}")
        reasoning_parts.append(f"Materialization probability: {materialization_probability:.3f}")
        
        decision = {
            'materialize': should_materialize,
            'confidence': abs(materialization_probability - 0.5) * 2,
            'reasoning': '; '.join(reasoning_parts),
            'materialization_probability': materialization_probability,
            'frequency_score': frequency_score,
            'similarity_score': similarity_score,
            'expected_reuses': frequency_score * similarity_score * 5  # Heuristic
        }
        
        print(f"   Decision: {'MATERIALIZE' if should_materialize else 'IN-MEMORY ONLY'}")
        print(f"   Confidence: {decision['confidence']:.2f}")
        print(f"   Expected reuses: {decision['expected_reuses']:.1f}")
        
        return decision
    
    def record_execution(self, parsed_query: Dict, exact_time: Optional[float] = None,
                        sampling_time: Optional[float] = None, sampling_success: bool = True):
        """
        Record execution results for learning.
        """
        if not self.learning_enabled:
            return
        
        # Create execution record
        execution_record = {
            'timestamp': time.time(),
            'query_fingerprint': self._create_query_fingerprint(parsed_query),
            'tables': parsed_query['tables'],
            'join_count': len(parsed_query['join_conditions']),
            'complexity': self._calculate_data_driven_complexity(parsed_query),
            'exact_time': exact_time,
            'sampling_time': sampling_time,
            'sampling_success': sampling_success
        }
        
        # Add to execution history
        self.execution_history.append(execution_record)
        
        # Update query frequency tracking
        fingerprint = execution_record['query_fingerprint']
        self.query_fingerprints[fingerprint] += 1
        self.query_recency[fingerprint] = time.time()
        
        # Adaptive threshold adjustment
        if exact_time and sampling_time and sampling_success:
            speedup = exact_time / sampling_time
            if speedup > self.min_speedup_required:
                # Sampling was beneficial, slightly lower threshold
                self.sampling_threshold = max(0.1, self.sampling_threshold * 0.99)
            else:
                # Sampling wasn't beneficial enough, raise threshold
                self.sampling_threshold = min(0.9, self.sampling_threshold * 1.01)
        
        print(f"📚 Recorded execution - updated thresholds: sampling={self.sampling_threshold:.3f}")
    
    def _calculate_data_driven_complexity(self, parsed_query: Dict) -> float:
        """
        Calculate complexity based on actual data characteristics, not just query structure.
        """
        complexity = 0.0
        
        # Base complexity from query structure
        structure_complexity = (
            len(parsed_query['tables']) * 0.2 +
            len(parsed_query['join_conditions']) * 0.4 +
            len(parsed_query['aggregates']) * 0.1 +
            len(parsed_query['where_conditions']) * 0.05 +
            len(parsed_query['group_by_columns']) * 0.1
        )
        
        complexity += structure_complexity
        
        # Data size factor
        total_rows = 0
        for table in parsed_query['tables']:
            table_stats = self._get_table_statistics(table)
            total_rows += table_stats['row_count']
        
        # Logarithmic scaling for data size (large datasets don't linearly increase complexity)
        if total_rows > 0:
            data_size_factor = math.log10(total_rows) / 10.0  # Normalize to roughly 0-1
            complexity += data_size_factor
        
        # Join selectivity factor (if we can estimate it)
        if parsed_query['join_conditions']:
            join_complexity = self._estimate_join_complexity(parsed_query)
            complexity += join_complexity
        
        return min(complexity, 10.0)  # Cap at 10 for normalization
    
    def _estimate_exact_execution_cost(self, parsed_query: Dict) -> float:
        """
        Estimate the time cost of exact execution based on learned patterns.
        """
        # Look for similar queries in execution history
        similar_executions = self._find_similar_executions(parsed_query)
        
        if similar_executions:
            # Use median of similar executions
            times = [exec['exact_time'] for exec in similar_executions if exec.get('exact_time')]
            if times:
                return sorted(times)[len(times) // 2]  # Median
        
        # Fallback: estimate based on table sizes and joins
        base_cost = 0.001  # Base query cost
        
        for table in parsed_query['tables']:
            table_stats = self._get_table_statistics(table)
            # Assume 1 microsecond per 1000 rows
            base_cost += table_stats['row_count'] / 1000000.0
        
        # Join penalty
        join_penalty = len(parsed_query['join_conditions']) * 0.01
        
        # Aggregate penalty
        agg_penalty = len(parsed_query['aggregates']) * 0.005
        
        return base_cost + join_penalty + agg_penalty
    
    def _estimate_sampling_cost(self, parsed_query: Dict) -> float:
        """
        Estimate the cost of sampling execution.
        """
        # Base sampling overhead
        sampling_overhead = 0.01  # Fixed overhead for sampling setup
        
        # Cost depends on sample size needed
        complexity = self._calculate_data_driven_complexity(parsed_query)
        estimated_sample_size = min(1000, max(100, int(complexity * 50)))
        
        # WanderJoin cost is roughly proportional to sample size and join complexity
        wander_join_cost = estimated_sample_size * len(parsed_query['join_conditions']) * 0.0001
        
        # Query rewriting and result scaling cost
        processing_cost = 0.002
        
        return sampling_overhead + wander_join_cost + processing_cost
    
    def _analyze_query_patterns(self, parsed_query: Dict) -> Dict:
        """
        Analyze historical patterns for queries similar to this one.
        """
        fingerprint = self._create_query_fingerprint(parsed_query)
        
        # Find executions of similar queries
        similar_executions = self._find_similar_executions(parsed_query)
        
        if not similar_executions:
            return {
                'similar_queries': 0,
                'avg_speedup': 1.0,
                'success_rate': 0.5
            }
        
        # Calculate average speedup from sampling
        speedups = []
        successes = 0
        
        for execution in similar_executions:
            if execution.get('exact_time') and execution.get('sampling_time'):
                speedup = execution['exact_time'] / execution['sampling_time']
                speedups.append(speedup)
                if speedup > 1.0:
                    successes += 1
        
        avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0
        success_rate = successes / len(similar_executions) if similar_executions else 0.5
        
        return {
            'similar_queries': len(similar_executions),
            'avg_speedup': avg_speedup,
            'success_rate': success_rate
        }
    
    def _calculate_sampling_probability(self, data_complexity: float, exact_cost: float, 
                                      sampling_cost: float, pattern_evidence: Dict) -> float:
        """
        Calculate probability that sampling will be beneficial.
        """
        # Base probability from cost comparison
        if exact_cost > sampling_cost:
            cost_probability = min(0.9, (exact_cost - sampling_cost) / exact_cost)
        else:
            cost_probability = 0.1  # Low probability if sampling seems more expensive
        
        # Complexity factor (higher complexity favors sampling)
        complexity_probability = min(0.8, data_complexity / 5.0)
        
        # Historical evidence factor
        if pattern_evidence['similar_queries'] > 0:
            evidence_probability = min(0.9, pattern_evidence['success_rate'])
            # Weight by number of similar queries (more evidence = more confidence)
            evidence_weight = min(1.0, pattern_evidence['similar_queries'] / 5.0)
        else:
            evidence_probability = 0.5  # Neutral when no evidence
            evidence_weight = 0.0
        
        # Combine factors
        # Weighted average: cost (50%), complexity (30%), evidence (20% * evidence_weight)
        probability = (
            cost_probability * 0.5 +
            complexity_probability * 0.3 +
            evidence_probability * (0.2 * evidence_weight) +
            0.5 * (0.2 * (1 - evidence_weight))  # Default neutral for missing evidence
        )
        
        return min(0.95, max(0.05, probability))  # Clamp between 5% and 95%
    
    def _calculate_frequency_score(self, query_fingerprint: str) -> float:
        """
        Calculate how frequently this query pattern has been seen.
        """
        current_time = time.time()
        
        # Direct frequency
        direct_frequency = self.query_fingerprints[query_fingerprint]
        
        # Recency factor (recent queries are more likely to repeat)
        last_seen = self.query_recency.get(query_fingerprint, 0)
        time_since_last = current_time - last_seen
        recency_factor = math.exp(-time_since_last / 3600)  # Decay over 1 hour
        
        # Combine frequency and recency
        frequency_score = direct_frequency * recency_factor
        
        # Normalize to 0-1 range (log scale)
        if frequency_score > 0:
            return min(1.0, math.log10(frequency_score + 1) / 2.0)
        else:
            return 0.0
    
    def _calculate_pattern_similarity_score(self, parsed_query: Dict) -> float:
        """
        Calculate how similar this query is to other frequently-executed queries.
        """
        query_fingerprint = self._create_query_fingerprint(parsed_query)
        
        # Find similar query patterns
        similarity_scores = []
        
        for other_fingerprint, frequency in self.query_fingerprints.items():
            if frequency > 1:  # Only consider queries seen multiple times
                similarity = self._calculate_fingerprint_similarity(query_fingerprint, other_fingerprint)
                if similarity > 0.3:  # Only consider reasonably similar queries
                    # Weight by frequency of the other query
                    weighted_similarity = similarity * math.log10(frequency + 1)
                    similarity_scores.append(weighted_similarity)
        
        if similarity_scores:
            return min(1.0, sum(similarity_scores) / len(similarity_scores))
        else:
            return 0.0
    
    def _calculate_materialization_probability(self, frequency_score: float, similarity_score: float,
                                             storage_cost: float, reuse_benefit: float, 
                                             creation_time: float) -> float:
        """
        Calculate probability that materialization will be beneficial.
        """
        # Benefit factors
        reuse_probability = min(0.9, frequency_score + similarity_score * 0.5)
        time_saved_factor = min(0.8, creation_time / 5.0)  # Normalize creation time
        
        # Cost factors  
        storage_cost_factor = max(0.1, 1.0 - storage_cost / 100.0)  # Penalty for large storage
        
        # Benefit-cost ratio
        benefit = reuse_probability * time_saved_factor * reuse_benefit
        cost = storage_cost * 0.01  # Small cost factor for storage
        
        if cost > 0:
            benefit_cost_ratio = benefit / cost
            probability = min(0.9, benefit_cost_ratio / 5.0)  # Normalize
        else:
            probability = benefit
        
        return max(0.05, probability)
    
    def _get_table_statistics(self, table_name: str) -> Dict:
        """
        Get cached table statistics or compute them.
        """
        if table_name in self.table_statistics:
            stats = self.table_statistics[table_name]
            # Refresh if older than 1 hour
            if time.time() - stats['timestamp'] < 3600:
                return stats
        
        # Compute fresh statistics
        try:
            count_result = self.db.execute_query(f"SELECT COUNT(*) FROM {table_name}")
            row_count = count_result[0][0] if count_result else 0
            
            # Could add more statistics here (avg row size, key distributions, etc.)
            stats = {
                'row_count': row_count,
                'timestamp': time.time()
            }
            
            self.table_statistics[table_name] = stats
            return stats
            
        except Exception as e:
            print(f"⚠️ Could not get statistics for {table_name}: {e}")
            return {'row_count': 100000, 'timestamp': time.time()}  # Default estimate
    
    def _create_query_fingerprint(self, parsed_query: Dict) -> str:
        """
        Create a string fingerprint for query pattern matching.
        """
        # Create canonical representation
        fingerprint_data = {
            'tables': tuple(sorted(parsed_query['tables'])),
            'join_count': len(parsed_query['join_conditions']),
            'has_aggregates': bool(parsed_query['aggregates']),
            'has_group_by': bool(parsed_query['group_by_columns'])
        }
        
        # Create hash
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:16]
    
    def _find_similar_executions(self, parsed_query: Dict) -> List[Dict]:
        """
        Find similar executions in history.
        """
        query_fingerprint = self._create_query_fingerprint(parsed_query)
        
        similar_executions = []
        for execution in self.execution_history:
            if execution['query_fingerprint'] == query_fingerprint:
                similar_executions.append(execution)
            elif self._are_queries_similar(parsed_query, execution):
                similar_executions.append(execution)
        
        return similar_executions
    
    def _are_queries_similar(self, query1: Dict, execution_record: Dict) -> bool:
        """
        Check if two queries are similar enough to use for learning.
        """
        # Same tables
        if set(query1['tables']) != set(execution_record['tables']):
            return False
        
        # Similar join structure
        if abs(len(query1['join_conditions']) - execution_record['join_count']) > 1:
            return False
        
        # Similar complexity
        query1_complexity = self._calculate_data_driven_complexity(query1)
        if abs(query1_complexity - execution_record['complexity']) > 2.0:
            return False
        
        return True
    
    def _calculate_fingerprint_similarity(self, fp1: str, fp2: str) -> float:
        """
        Calculate similarity between two query fingerprints.
        Simple implementation - could be more sophisticated.
        """
        if fp1 == fp2:
            return 1.0
        
        # Hamming distance for simple similarity
        if len(fp1) == len(fp2):
            differences = sum(c1 != c2 for c1, c2 in zip(fp1, fp2))
            return 1.0 - (differences / len(fp1))
        
        return 0.0
    
    def _estimate_join_complexity(self, parsed_query: Dict) -> float:
        """
        Estimate complexity of join operations.
        """
        if not parsed_query['join_conditions']:
            return 0.0
        
        # Simple heuristic based on table sizes
        total_complexity = 0.0
        
        for join in parsed_query['join_conditions']:
            left_stats = self._get_table_statistics(join['left_table'])
            right_stats = self._get_table_statistics(join['right_table'])
            
            # Join complexity roughly proportional to product of table sizes
            join_complexity = math.log10(left_stats['row_count'] * right_stats['row_count']) / 10.0
            total_complexity += join_complexity
        
        return total_complexity
    
    def _estimate_storage_cost(self, sample_size: int, tables: List[str]) -> float:
        """
        Estimate storage cost in MB for materialized sample.
        """
        # Rough estimate: average row size * sample size
        estimated_row_size = len(tables) * 50  # Assume 50 bytes per table on average
        storage_mb = (sample_size * estimated_row_size) / (1024 * 1024)
        return storage_mb
    
    def _estimate_reuse_benefit(self, creation_time: float, frequency_score: float, 
                               similarity_score: float) -> float:
        """
        Estimate benefit from reusing this sample.
        """
        # Time saved per reuse
        time_saved_per_reuse = creation_time
        
        # Expected number of reuses
        expected_reuses = frequency_score * 5 + similarity_score * 3
        
        # Total expected benefit
        return time_saved_per_reuse * expected_reuses
    
    def get_learning_statistics(self) -> Dict:
        """
        Get statistics about the learning system.
        """
        return {
            'execution_history_size': len(self.execution_history),
            'unique_query_patterns': len(self.query_fingerprints),
            'total_query_executions': sum(self.query_fingerprints.values()),
            'current_sampling_threshold': self.sampling_threshold,
            'current_materialization_threshold': self.materialization_threshold,
            'table_statistics_cached': len(self.table_statistics)
        }