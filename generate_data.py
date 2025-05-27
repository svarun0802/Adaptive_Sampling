from database.connector import DatabaseConnector
from testing.data_generator import SSBDataGenerator

def main():
    """Generate SSB data for testing."""
    
    # Initialize database connection
    db = DatabaseConnector()
    
    # Use scale_factor=0.1 for development (100MB of data)
    # or scale_factor=1.0 for more realistic testing (1GB of data)
    generator = SSBDataGenerator(scale_factor=0.1, db_connector=db)
    
    # Generate all data
    generator.generate_all_data()
    
    # Test a simple query to verify data
    result = db.execute_query("SELECT COUNT(*) FROM lineorder")
    print(f"Total lineorders: {result[0][0]}")
    
    result = db.execute_query("SELECT COUNT(*) FROM customer")
    print(f"Total customers: {result[0][0]}")
    
    result = db.execute_query("SELECT COUNT(*) FROM supplier")
    print(f"Total suppliers: {result[0][0]}")
    
    result = db.execute_query("SELECT COUNT(*) FROM part")
    print(f"Total parts: {result[0][0]}")
    
    # Close connection
    db.close()

if __name__ == "__main__":
    main()