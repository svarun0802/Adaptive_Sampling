import json
import time
import uuid
from database.connector import DatabaseConnector
from database.schema import SSBSchema

class SampleStorage:
    def __init__(self, db_connector=None):
        """Initialize the sample storage system."""
        self.db = db_connector if db_connector else DatabaseConnector()
        self._ensure_metadata_table_exists()
    
    def _ensure_metadata_table_exists(self):
        """Create metadata table if it doesn't exist."""
        self.db.create_table("materialized_samples_metadata", SSBSchema.get_sample_metadata_schema())
    
    def materialize_sample(self, samples, tables, join_conditions, source_query=None):
        """
        Store samples as a materialized view with PROPER DATA TYPES.
        
        FIXED VERSION: Preserves original data types instead of converting everything to VARCHAR
        """
        if not samples:
            raise ValueError("Cannot materialize empty sample set")
        
        # Generate a unique ID for this sample
        sample_id = f"sample_{uuid.uuid4().hex[:8]}"
        
        # Create columns with proper data types
        sample_columns = []
        for key in samples[0].keys():
            clean_key = key.replace('.', '_')
            
            # Infer appropriate data type based on column name patterns
            if any(x in key.lower() for x in ['revenue', 'price', 'cost', 'discount', 'tax', 'ordtotalprice', 'extendedprice', 'supplycost']):
                column_type = "DECIMAL(15,2)"
            elif any(x in key.lower() for x in ['key', 'date', 'quantity', 'year', 'month', 'day', 'orderkey', 'linenumber', 'custkey', 'partkey', 'suppkey', 'orderdate', 'commitdate', 'yearmonthnum', 'daynuminweek', 'daynuminmonth', 'daynuminyear', 'monthnuminyear', 'weeknuminyear', 'lastdayinweekfl', 'lastdayinmonthfl', 'holidayfl', 'weekdayfl', 'size']):
                column_type = "INTEGER"
            else:
                column_type = "VARCHAR"
            
            sample_columns.append(f"{clean_key} {column_type}")
        
        schema = ", ".join(sample_columns)
        self.db.create_table(sample_id, schema)
        
        # Insert samples with proper type conversion
        for sample in samples:
            column_names = []
            column_values = []
            
            for key, value in sample.items():
                clean_key = key.replace('.', '_')
                column_names.append(clean_key)
                
                # Convert value to appropriate type
                if value is None:
                    column_values.append(None)
                elif any(x in key.lower() for x in ['revenue', 'price', 'cost', 'discount', 'tax', 'ordtotalprice', 'extendedprice', 'supplycost']):
                    try:
                        column_values.append(float(value))
                    except (ValueError, TypeError):
                        column_values.append(0.0)
                elif any(x in key.lower() for x in ['key', 'date', 'quantity', 'year', 'month', 'day', 'orderkey', 'linenumber', 'custkey', 'partkey', 'suppkey', 'orderdate', 'commitdate', 'yearmonthnum', 'daynuminweek', 'daynuminmonth', 'daynuminyear', 'monthnuminyear', 'weeknuminyear', 'lastdayinweekfl', 'lastdayinmonthfl', 'holidayfl', 'weekdayfl', 'size']):
                    try:
                        # Handle cases where value might be a float string like "1993.0"
                        column_values.append(int(float(str(value))))
                    except (ValueError, TypeError):
                        column_values.append(0)
                else:
                    column_values.append(str(value) if value is not None else "")
            
            columns = ", ".join(column_names)
            placeholders = ", ".join(["?"] * len(column_values))
            
            query = f"INSERT INTO {sample_id} ({columns}) VALUES ({placeholders})"
            self.db.execute_query(query, tuple(column_values))
        
        # Store metadata about this sample
        metadata = {
            "sample_id": sample_id,
            "source_tables": json.dumps(sorted(tables)),  # Sort for consistent matching
            "join_conditions": json.dumps(sorted(join_conditions)),  # Sort for consistent matching
            "sample_size": len(samples),
            "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_used": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query_count": 0,
            "error_stats": "{}"
        }
        
        # Create insert query
        columns = ", ".join(metadata.keys())
        placeholders = ", ".join(["?"] * len(metadata))
        query = f"INSERT INTO materialized_samples_metadata ({columns}) VALUES ({placeholders})"
        
        self.db.execute_query(query, tuple(metadata.values()))
        
        print(f"Materialized {len(samples)} samples as '{sample_id}'")
        return sample_id
    
    def find_best_sample_match(self, query_tables, query_join_conditions):
        """
        Find the best materialized sample to use for this query.
        
        Parameters:
        - query_tables: Tables in the query
        - query_join_conditions: Join conditions in the query
        
        Returns:
        - sample_info: Dictionary with sample details, or None
        """
        # First check for exact matches
        sorted_tables = sorted(query_tables)
        sorted_conditions = sorted(query_join_conditions)
        
        query = """
        SELECT sample_id, source_tables, join_conditions, sample_size 
        FROM materialized_samples_metadata 
        WHERE source_tables = ? AND join_conditions = ?
        """
        
        result = self.db.execute_query(query, (json.dumps(sorted_tables), json.dumps(sorted_conditions)))
        
        if result:
            # Perfect match found
            return {
                'sample_id': result[0][0],
                'exact_match': True,
                'needs_extension': False,
                'coverage': 1.0,
                'base_tables': query_tables,
                'additional_tables': [],
                'additional_joins': []
            }
        
        # No exact match, look for partial matches (subsets)
        all_samples = self.list_materialized_samples()
        
        best_match = None
        best_coverage = 0
        
        for sample_metadata in all_samples:
            sample_id = sample_metadata[0]
            sample_tables = set(json.loads(sample_metadata[1]))
            sample_joins = json.loads(sample_metadata[2])
            sample_size = sample_metadata[3]
            
            # Check if this sample is a subset of our query tables
            if sample_tables.issubset(set(query_tables)):
                coverage = len(sample_tables) / len(query_tables)
                
                if coverage > best_coverage:
                    additional_tables = [t for t in query_tables if t not in sample_tables]
                    additional_joins = self._find_additional_joins(
                        query_join_conditions, sample_tables, additional_tables
                    )
                    
                    best_match = {
                        'sample_id': sample_id,
                        'exact_match': False,
                        'needs_extension': True,
                        'coverage': coverage,
                        'base_tables': list(sample_tables),
                        'additional_tables': additional_tables,
                        'additional_joins': additional_joins,
                        'sample_size': sample_size
                    }
                    best_coverage = coverage
        
        return best_match
    
    def _find_additional_joins(self, all_join_conditions, base_tables, additional_tables):
        """Find join conditions that connect base tables to additional tables."""
        additional_joins = []
        
        for condition in all_join_conditions:
            table1, col1, table2, col2 = condition
            
            # Check if this join connects base to additional tables
            if ((table1 in base_tables and table2 in additional_tables) or
                (table1 in additional_tables and table2 in base_tables) or
                (table1 in additional_tables and table2 in additional_tables)):
                additional_joins.append(condition)
        
        return additional_joins
    
    def extend_materialized_sample(self, sample_info, target_sample_size=100):
        """
        Extend an existing materialized sample to include additional tables.
        
        Parameters:
        - sample_info: Information from find_best_sample_match
        - target_sample_size: Desired number of extended samples
        
        Returns:
        - extended_sample_id: ID of the new extended sample
        """
        if not sample_info['needs_extension']:
            raise ValueError("Sample doesn't need extension")
        
        # Get the base sample data
        base_sample_id = sample_info['sample_id']
        
        # Get column information correctly
        base_column_info = self.db.execute_query(f"PRAGMA table_info({base_sample_id})")
        base_column_names = [col[1] for col in base_column_info]  # col[1] is the column name
        
        # Get the actual data
        base_sample_data = self.db.execute_query(f"SELECT * FROM {base_sample_id}")
        
        # Convert base sample data to dictionaries
        base_samples = []
        for row in base_sample_data:
            sample_dict = {}
            for i, col in enumerate(base_column_names):
                # Convert back from sanitized format to table.column format
                if isinstance(col, str) and '_' in col:
                    # Split column name to get table and column parts
                    parts = col.split('_', 1)  # Split on first underscore
                    if len(parts) == 2:
                        # Skip certain prefixes that might exist
                        if parts[0].lower() in ['lineorder', 'customer', 'supplier', 'part', 'date_dim', 'date']:
                            reconstructed_key = f"{parts[0]}.{parts[1]}"
                        else:
                            reconstructed_key = col
                    else:
                        reconstructed_key = col
                else:
                    reconstructed_key = str(col) if col is not None else f"col_{i}"
                
                sample_dict[reconstructed_key] = row[i]
            base_samples.append(sample_dict)
        
        # Debug: Print sample structure
        if base_samples:
            print("Sample base_samples[0] keys:", list(base_samples[0].keys()))
        
        # Use WanderJoin to extend the samples
        from sampling.wander_join import WanderJoin
        wj = WanderJoin(self.db)
        
        extended_samples = wj.extend_samples(
            base_samples,
            sample_info['base_tables'],
            sample_info['additional_tables'],
            sample_info['additional_joins'],
            target_sample_size
        )
        
        # Materialize the extended samples
        all_tables = sample_info['base_tables'] + sample_info['additional_tables']
        all_joins = sample_info['additional_joins']
        
        extended_sample_id = self.materialize_sample(
            extended_samples,
            all_tables,
            all_joins
        )
        
        return extended_sample_id
    
    def get_materialized_sample(self, tables, join_conditions):
        """Find a materialized sample matching the given join pattern."""
        # Convert parameters to JSON for matching
        tables_json = json.dumps(sorted(tables))
        conditions_json = json.dumps(sorted(join_conditions))
        
        query = """
        SELECT sample_id, sample_size 
        FROM materialized_samples_metadata 
        WHERE source_tables = ? AND join_conditions = ?
        """
        
        result = self.db.execute_query(query, (tables_json, conditions_json))
        
        if result:
            sample_id, sample_size = result[0]
            
            # Update usage statistics
            update_query = """
            UPDATE materialized_samples_metadata 
            SET last_used = ?, query_count = query_count + 1
            WHERE sample_id = ?
            """
            
            self.db.execute_query(update_query, (time.strftime("%Y-%m-%d %H:%M:%S"), sample_id))
            
            return sample_id, sample_size
        
        return None, 0
    
    def list_materialized_samples(self):
        """List all materialized samples."""
        query = "SELECT * FROM materialized_samples_metadata"
        return self.db.execute_query(query)
    
    def drop_materialized_sample(self, sample_id):
        """Remove a materialized sample."""
        # Drop the sample table
        self.db.execute_query(f"DROP TABLE IF EXISTS {sample_id}")
        
        # Remove metadata
        query = "DELETE FROM materialized_samples_metadata WHERE sample_id = ?"
        self.db.execute_query(query, (sample_id,))
        
        print(f"Dropped materialized sample '{sample_id}'")

    def debug_sample_structure(self, sample_id):
        """Debug function to see the structure of a stored sample."""
        print(f"\nDebugging sample {sample_id}:")
        
        # Get column info
        column_info = self.db.execute_query(f"PRAGMA table_info({sample_id})")
        print("Column info:")
        for i, col_info in enumerate(column_info):
            print(f"  {i}: {col_info}")
        
        # Get actual column names
        column_names = [col[1] for col in column_info]
        print(f"Column names: {column_names}")
        
        # Get first row of data
        data = self.db.execute_query(f"SELECT * FROM {sample_id} LIMIT 1")
        if data:
            print(f"First row: {data[0]}")
            print(f"Data types: {[type(val) for val in data[0]]}")