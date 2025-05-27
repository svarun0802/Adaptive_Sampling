from database.connector import DatabaseConnector
from sampling.wander_join import WanderJoin
from sampling.sample_storage import SampleStorage

def main():
    """Test sample extension capability."""
    db = DatabaseConnector()
    wj = WanderJoin(db_connector=db)
    storage = SampleStorage(db_connector=db)
    
    # Step 1: Create a base sample with lineorder-customer join
    print("Step 1: Creating base sample (lineorder-customer)...")
    base_tables = ["lineorder", "customer"]
    base_joins = [("lineorder", "lo_custkey", "customer", "c_custkey")]
    
    base_samples = wj.sample_join(base_tables, base_joins, sample_size=50)
    base_sample_id = storage.materialize_sample(base_samples, base_tables, base_joins)
    
    # Debug the created sample
    storage.debug_sample_structure(base_sample_id)
    
    # Step 2: Create a query that extends the base sample
    print("\nStep 2: Testing extension to include supplier table...")
    extended_tables = ["lineorder", "customer", "supplier"]
    extended_joins = [
        ("lineorder", "lo_custkey", "customer", "c_custkey"),
        ("lineorder", "lo_suppkey", "supplier", "s_suppkey")
    ]
    
    # Step 3: Find if we can use the base sample
    match_info = storage.find_best_sample_match(extended_tables, extended_joins)
    
    if match_info:
        print(f"\nFound matching sample: {match_info['sample_id']}")
        print(f"Exact match: {match_info['exact_match']}")
        print(f"Coverage: {match_info['coverage']:.2%}")
        print(f"Needs extension: {match_info['needs_extension']}")
        
        if match_info['needs_extension']:
            print(f"Additional tables needed: {match_info['additional_tables']}")
            print(f"Additional joins needed: {match_info['additional_joins']}")
            
            # Debug the sample we found
            storage.debug_sample_structure(match_info['sample_id'])
            
            # Step 4: Extend the sample
            print("\nStep 4: Extending the sample...")
            try:
                extended_sample_id = storage.extend_materialized_sample(match_info, target_sample_size=30)
                
                # Step 5: Verify the extended sample
                print(f"\nStep 5: Verifying extended sample...")
                result = db.execute_query(f"SELECT COUNT(*) FROM {extended_sample_id}")
                print(f"Extended sample count: {result[0][0]}")
                
                # Show a few extended samples
                result = db.execute_query(f"SELECT * FROM {extended_sample_id} LIMIT 2")
                print("\nFirst two extended samples:")
                for i, row in enumerate(result):
                    print(f"Sample {i+1}: {row}")
            except Exception as e:
                print(f"Error during extension: {e}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()
    else:
        print("No suitable base sample found for extension")
    
    db.close()

if __name__ == "__main__":
    main()