import duckdb

class DatabaseConnector:
    def __init__(self, db_path='adaptive_sample.db'):
        """Initialize connection to DuckDB."""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)

    def execute_query(self, query, params=None):
        """Execute a query and return results."""
        try:
            if params:
                result = self.conn.execute(query, params)
            else:
                result = self.conn.execute(query)
        
            return result.fetchall()
        except Exception as e:
            print(f"Query execution error: {e}")
            return None
        
    def create_table(self, table_name, schema):
        """Create a table with given schema"""
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
        self.execute_query(query)
        print(f"Table {table_name} created successfully")
        
    def drop_table(self, table_name):
        """Drop a table if it exists"""
        query = f"DROP TABLE IF EXISTS {table_name}"
        self.execute_query(query)
        print(f"Table {table_name} dropped successfully")
        
    def get_table_columns(self, table_name):
        """Get column names of a table"""
        query = f"PRAGMA table_info({table_name})"
        return self.execute_query(query)
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")
        else:
            print("No database connection to close")
        
        
