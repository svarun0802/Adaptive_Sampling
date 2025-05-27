from database.connector import DatabaseConnector
from sampling.wander_join import WanderJoin
from sampling.sample_storage import SampleStorage

def main():
    """Test Wander Join sampling on SSB data."""
    db = DatabaseConnector()
    
    # Create Wander Join instance
    wj = WanderJoin(db_connector=db)
    
    # Define a simple join between lineorder and customer
    tables = ["lineorder", "customer"]
    join_conditions = [("lineorder", "lo_custkey", "customer", "c_custkey")]
    
    print("Sampling lineorder ⋈ customer join...")
    samples = wj.sample_join(tables, join_conditions, sample_size=100)
    
    # Print first few samples
    print("\nSample results:")
    for i, sample in enumerate(samples[:5]):  # Print first 5 samples
        print(f"\nSample {i+1}:")
        for key, value in sample.items():
            print(f"  {key}: {value}")
    
    # Test materialization
    print("\nTesting sample materialization...")
    storage = SampleStorage(db_connector=db)
    sample_id = storage.materialize_sample(samples, tables, join_conditions)
    
    # Verify storage
    print("\nVerifying stored samples:")
    result = db.execute_query(f"SELECT COUNT(*) FROM {sample_id}")
    print(f"Stored sample count: {result[0][0]}")
    
    # List first few materialized samples
    result = db.execute_query(f"SELECT * FROM {sample_id} LIMIT 3")
    for i, row in enumerate(result):
        print(f"\nStored sample {i+1}:")
        print(row)
    
    db.close()

if __name__ == "__main__":
    main()