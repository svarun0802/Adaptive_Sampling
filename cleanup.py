#!/usr/bin/env python3
"""
System Cleanup Script - Reset to Fresh Start

This script removes all materialized samples and learning state
while preserving the original SSB data (lineorder, customer, supplier, part, date_dim)
"""

import os
from database.connector import DatabaseConnector
from database.schema import SSBSchema

def cleanup_materialized_samples(db):
    """Remove all materialized sample tables and metadata"""
    
    print("🧹 CLEANING MATERIALIZED SAMPLES...")
    
    # Step 1: Get list of all materialized samples from metadata
    try:
        samples = db.execute_query("SELECT sample_id FROM materialized_samples_metadata")
        sample_ids = [sample[0] for sample in samples] if samples else []
    except:
        print("   ⚠️  No materialized_samples_metadata table found")
        sample_ids = []
    
    # Step 2: Drop all sample tables
    dropped_count = 0
    if sample_ids:
        print(f"   📋 Found {len(sample_ids)} materialized samples to remove")
        for sample_id in sample_ids:
            try:
                db.execute_query(f"DROP TABLE IF EXISTS {sample_id}")
                print(f"   ✅ Dropped table: {sample_id}")
                dropped_count += 1
            except Exception as e:
                print(f"   ⚠️  Failed to drop {sample_id}: {e}")
    
    # Step 3: Clear metadata table
    try:
        db.execute_query("DELETE FROM materialized_samples_metadata")
        print(f"   ✅ Cleared metadata table")
    except:
        print(f"   ⚠️  Could not clear metadata table")
    
    # Step 4: Find any orphaned sample tables (safety check)
    try:
        # Get all table names
        all_tables = db.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
        if all_tables:
            orphaned_samples = [table[0] for table in all_tables if table[0].startswith('sample_')]
            
            if orphaned_samples:
                print(f"   🔍 Found {len(orphaned_samples)} orphaned sample tables")
                for orphan in orphaned_samples:
                    try:
                        db.execute_query(f"DROP TABLE IF EXISTS {orphan}")
                        print(f"   ✅ Dropped orphaned table: {orphan}")
                        dropped_count += 1
                    except Exception as e:
                        print(f"   ⚠️  Failed to drop {orphan}: {e}")
    except:
        print("   ⚠️  Could not check for orphaned tables")
    
    print(f"   📊 Total sample tables removed: {dropped_count}")
    return dropped_count

def recreate_metadata_table(db):
    """Recreate clean metadata table"""
    
    print("\n🔧 RECREATING METADATA TABLE...")
    
    # Drop existing metadata table
    try:
        db.execute_query("DROP TABLE IF EXISTS materialized_samples_metadata")
        print("   ✅ Dropped old metadata table")
    except:
        pass
    
    # Create fresh metadata table
    try:
        db.create_table("materialized_samples_metadata", SSBSchema.get_sample_metadata_schema())
        print("   ✅ Created fresh metadata table")
    except Exception as e:
        print(f"   ❌ Failed to create metadata table: {e}")

def verify_core_data(db):
    """Verify that core SSB data is intact"""
    
    print("\n🔍 VERIFYING CORE DATA INTEGRITY...")
    
    core_tables = ['lineorder', 'customer', 'supplier', 'part', 'date_dim']
    verification_passed = True
    
    for table in core_tables:
        try:
            result = db.execute_query(f"SELECT COUNT(*) FROM {table}")
            if result and result[0][0] > 0:
                count = result[0][0]
                print(f"   ✅ {table}: {count:,} rows")
            else:
                print(f"   ❌ {table}: No data found!")
                verification_passed = False
        except Exception as e:
            print(f"   ❌ {table}: Error - {e}")
            verification_passed = False
    
    if verification_passed:
        print("   🎉 All core data intact!")
    else:
        print("   ⚠️  Some core data issues detected!")
    
    return verification_passed

def reset_learning_state():
    """Reset any persistent learning state (if applicable)"""
    
    print("\n🧠 RESETTING LEARNING STATE...")
    
    # Note: Current implementation stores learning state in memory only
    # If you ever add persistent learning state, reset it here
    
    print("   ✅ Learning state will be reset on next system startup")
    print("   📊 Thresholds will return to defaults")
    print("   🗄️  Execution history will be cleared")

def show_cleanup_summary(db, samples_removed):
    """Show final cleanup summary"""
    
    print(f"\n📋 CLEANUP SUMMARY")
    print("=" * 40)
    
    # Check remaining tables
    try:
        all_tables = db.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
        if all_tables:
            remaining_tables = [table[0] for table in all_tables]
            sample_tables = [t for t in remaining_tables if t.startswith('sample_')]
            
            print(f"🗑️  Samples removed: {samples_removed}")
            print(f"🏠 Core tables preserved: {len([t for t in remaining_tables if t in ['lineorder', 'customer', 'supplier', 'part', 'date_dim']])}")
            print(f"📊 Metadata table: Reset")
            print(f"🧠 Learning state: Will reset on startup")
            
            if sample_tables:
                print(f"⚠️  Remaining sample tables: {len(sample_tables)}")
                for st in sample_tables:
                    print(f"     • {st}")
            else:
                print(f"✅ No sample tables remaining")
                
    except Exception as e:
        print(f"⚠️  Could not verify final state: {e}")

def cleanup_system():
    """Main cleanup function"""
    
    print("🚨 ADAPTIVE SAMPLING SYSTEM CLEANUP")
    print("=" * 50)
    print("This will remove ALL materialized samples and reset learning state")
    print("Core SSB data (lineorder, customer, etc.) will be preserved")
    print("=" * 50)
    
    # Confirmation
    confirm = input("\nAre you sure you want to proceed? (yes/no): ").lower().strip()
    if confirm != 'yes':
        print("❌ Cleanup cancelled")
        return
    
    # Initialize database connection
    print(f"\n🔌 Connecting to database...")
    db = DatabaseConnector()
    
    try:
        # Step 1: Verify core data before cleanup
        core_intact = verify_core_data(db)
        if not core_intact:
            proceed = input("\n⚠️  Core data issues detected. Continue anyway? (yes/no): ").lower().strip()
            if proceed != 'yes':
                print("❌ Cleanup cancelled due to core data issues")
                return
        
        # Step 2: Clean up samples
        samples_removed = cleanup_materialized_samples(db)
        
        # Step 3: Recreate metadata table
        recreate_metadata_table(db)
        
        # Step 4: Reset learning state
        reset_learning_state()
        
        # Step 5: Final verification
        verify_core_data(db)
        
        # Step 6: Summary
        show_cleanup_summary(db, samples_removed)
        
        print(f"\n🎉 CLEANUP COMPLETE!")
        print("System is ready for fresh start")
        
    except Exception as e:
        print(f"\n❌ CLEANUP FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close database connection
        try:
            db.close()
            print("🔌 Database connection closed")
        except:
            pass

def quick_cleanup():
    """Quick cleanup without confirmations (for development)"""
    
    print("🧹 QUICK CLEANUP (No confirmations)")
    
    db = DatabaseConnector()
    
    try:
        samples_removed = cleanup_materialized_samples(db)
        recreate_metadata_table(db)
        print(f"\n✅ Quick cleanup complete - {samples_removed} samples removed")
    except Exception as e:
        print(f"❌ Quick cleanup failed: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_cleanup()
    else:
        cleanup_system()