import random
import time
from database.connector import DatabaseConnector

class WanderJoin:
    def __init__(self, db_connector=None):
        """Initialize the Wander Join sampler."""
        self.db = db_connector if db_connector else DatabaseConnector()
    
    def sample_join(self, tables, join_conditions, sample_size=100, timeout=30):
        """
        Sample from a join using Wander Join algorithm.
        
        Parameters:
        - tables: List of table names in the join
        - join_conditions: List of tuples (table1, col1, table2, col2) 
        - sample_size: Number of samples to collect
        - timeout: Maximum time to spend sampling (seconds)
        
        Returns:
        - List of sample tuples from the join
        """
        if len(tables) < 2:
            raise ValueError("Need at least two tables to perform a join")
        
        samples = []
        successful_walks = 0
        failed_walks = 0
        start_time = time.time()
        
        # Build join graph
        join_graph = self._build_join_graph(tables, join_conditions)
        
        # Get column names for all tables
        column_names = {}
        for table in tables:
            query = f"PRAGMA table_info({table})"
            columns = self.db.execute_query(query)
            column_names[table] = [col[1] for col in columns]
        
        while successful_walks < sample_size and time.time() - start_time < timeout:
            # Start with a random tuple from the first table
            start_table = tables[0]
            query = f"SELECT * FROM {start_table} ORDER BY RANDOM() LIMIT 1"
            result = self.db.execute_query(query)
            
            if not result:
                continue  # Empty table
            
            start_tuple = result[0]
            
            # Try to perform a random walk
            sample = self._random_walk(start_tuple, start_table, join_graph, tables, column_names)
            
            if sample:
                samples.append(sample)
                successful_walks += 1
            else:
                failed_walks += 1
        
        sampling_time = time.time() - start_time
        
        print(f"Wander Join statistics:")
        print(f"- Successful walks: {successful_walks}")
        print(f"- Failed walks: {failed_walks}")
        print(f"- Success rate: {successful_walks / (successful_walks + failed_walks):.2%}")
        print(f"- Sampling time: {sampling_time:.2f} seconds")
        print(f"- Samples per second: {successful_walks / sampling_time:.2f}")
        
        return samples
    
    def extend_samples(self, base_samples, base_tables, additional_tables, 
                      additional_join_conditions, target_sample_size=None):
        """
        Extend existing samples to include additional tables.
        
        Parameters:
        - base_samples: List of existing sample dictionaries
        - base_tables: Tables already included in the base samples
        - additional_tables: New tables to join with
        - additional_join_conditions: Join conditions connecting to new tables
        - target_sample_size: Desired number of extended samples (defaults to len(base_samples))
        
        Returns:
        - List of extended sample dictionaries
        """
        if not base_samples:
            return []
        
        if target_sample_size is None:
            target_sample_size = len(base_samples)
        
        # Combine all tables and join conditions
        all_tables = base_tables + [t for t in additional_tables if t not in base_tables]
        all_join_conditions = additional_join_conditions  # These should connect base to additional tables
        
        # Build complete join graph
        join_graph = self._build_join_graph(all_tables, all_join_conditions)
        
        # Get column names for all tables
        column_names = {}
        for table in all_tables:
            query = f"PRAGMA table_info({table})"
            columns = self.db.execute_query(query)
            column_names[table] = [col[1] for col in columns]
        
        extended_samples = []
        successful_extensions = 0
        failed_extensions = 0
        start_time = time.time()
        
        # We might need to try multiple times to get enough samples
        max_attempts = len(base_samples) * 2  # Allow some failures
        
        for _ in range(max_attempts):
            if successful_extensions >= target_sample_size:
                break
                
            # Pick a random base sample
            base_sample = random.choice(base_samples)
            
            # Try to extend this sample
            extended_sample = self._extend_single_sample(
                base_sample, base_tables, additional_tables, 
                join_graph, column_names
            )
            
            if extended_sample:
                extended_samples.append(extended_sample)
                successful_extensions += 1
            else:
                failed_extensions += 1
        
        extension_time = time.time() - start_time
        
        print(f"Sample extension statistics:")
        print(f"- Successful extensions: {successful_extensions}")
        print(f"- Failed extensions: {failed_extensions}")
        print(f"- Success rate: {successful_extensions / (successful_extensions + failed_extensions):.2%}")
        print(f"- Extension time: {extension_time:.2f} seconds")
        
        return extended_samples
    
    def _extend_single_sample(self, base_sample, base_tables, additional_tables, 
                         join_graph, column_names):
        """Extend a single sample to include additional tables."""
        # Start with the base sample
        extended_sample = dict(base_sample)
        
        # Track which tables we've visited
        visited_tables = set(base_tables)
        remaining_tables = set(additional_tables)
        
        # Keep trying until we've visited all additional tables or fail
        while remaining_tables:
            # Flag to track if we made progress in this iteration
            progress_made = False
            
            # Create a list of visited tables to avoid modifying set during iteration
            current_visited = list(visited_tables)  # Convert to list to avoid iteration error
            
            # Try to find a connection from any visited table to any remaining table
            for current_table in current_visited:
                if progress_made:
                    break  # We found a connection, restart the outer loop
                
                # Check all neighbors of the current table
                neighbors = join_graph.get(current_table, [])
                for neighbor_info in neighbors:
                    neighbor_table, current_col, neighbor_col = neighbor_info
                    
                    if neighbor_table in remaining_tables:
                        # Found a connection to an unvisited table!
                        
                        # Get the join value from the current sample
                        join_key = self._find_join_key(extended_sample, current_table, current_col)
                        
                        if join_key is None:
                            continue  # Can't find the join value, try next neighbor
                        
                        join_value = extended_sample[join_key]
                        
                        # Find matching tuples in the next table
                        query = f"SELECT * FROM {neighbor_table} WHERE {neighbor_col} = ? LIMIT 1000"
                        matching_tuples = self.db.execute_query(query, (join_value,))
                        
                        if not matching_tuples:
                            continue  # No matching tuples, try next neighbor
                        
                        # Randomly select one matching tuple
                        next_tuple = random.choice(matching_tuples)
                        
                        # Add the new table's columns to our sample
                        for i, col in enumerate(column_names[neighbor_table]):
                            extended_sample[f"{neighbor_table}.{col}"] = next_tuple[i]
                        
                        # Update tracking
                        visited_tables.add(neighbor_table)
                        remaining_tables.remove(neighbor_table)
                        progress_made = True
                        break  # Found one connection, that's enough for this iteration
            
            if not progress_made:
                # No progress possible, extension fails
                return None
        
        # Successfully extended the sample
        return extended_sample

    def _find_join_key(self, sample, table, column):
        """Find the join key in the sample dictionary, handling different formats."""
        # Try different key formats
        possible_keys = [
            f"{table}.{column}",
            f"{table}_{column}",
            column
        ]
        
        for key in possible_keys:
            if key in sample:
                return key
        
        # If not found with exact table name, try with other table prefixes
        # This handles cases where the column might have come from a different table
        # but represents the same logical value
        for key in sample.keys():
            if key.endswith(f".{column}") or key.endswith(f"_{column}"):
                return key
        
        return None
    
    def _build_join_graph(self, tables, join_conditions):
        """Build a graph representation of the join."""
        join_graph = {table: [] for table in tables}
        
        for condition in join_conditions:
            table1, col1, table2, col2 = condition
            join_graph[table1].append((table2, col1, col2))
            join_graph[table2].append((table1, col2, col1))
        
        return join_graph
    
    def _random_walk(self, start_tuple, start_table, join_graph, tables, column_names):
        """Perform a random walk through the join graph."""
        visited = {start_table: start_tuple}
        current_table = start_table
        
        # Keep track of the path for debugging
        path = [start_table]
        
        # Try to visit all tables in the join
        remaining_tables = set(tables) - {start_table}
        
        while remaining_tables:
            # Find neighbors that we haven't visited yet
            valid_neighbors = []
            
            for neighbor_info in join_graph[current_table]:
                neighbor_table, current_col, neighbor_col = neighbor_info
                
                if neighbor_table in remaining_tables:
                    valid_neighbors.append((neighbor_table, current_col, neighbor_col))
            
            if not valid_neighbors:
                # No valid neighbors, walk fails
                return None
            
            # Choose a random neighbor
            next_table, current_col, next_col = random.choice(valid_neighbors)
            
            # Get the join value from current tuple
            current_col_idx = column_names[current_table].index(current_col)
            join_value = visited[current_table][current_col_idx]
            
            # Find matching tuples in the next table
            query = f"SELECT * FROM {next_table} WHERE {next_col} = ? LIMIT 1000"
            matching_tuples = self.db.execute_query(query, (join_value,))
            
            if not matching_tuples:
                # No matching tuples, walk fails
                return None
            
            # Randomly select one matching tuple
            next_tuple = random.choice(matching_tuples)
            visited[next_table] = next_tuple
            current_table = next_table
            path.append(current_table)
            
            # Remove this table from the remaining set
            remaining_tables.remove(next_table)
        
        # Successfully visited all tables, create result
        result = {}
        
        # Combine all the tuples into a single result
        for table in visited:
            tuple_values = visited[table]
            for i, col in enumerate(column_names[table]):
                result[f"{table}.{col}"] = tuple_values[i]
        
        return result