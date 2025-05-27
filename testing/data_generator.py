import random
from datetime import datetime, timedelta
from database.connector import DatabaseConnector
from database.schema import SSBSchema
import time

class SSBDataGenerator:
    def __init__(self, scale_factor=0.1, db_connector=None):
        """Initialize the SSB data generator with scale factor."""
        self.scale_factor = scale_factor
        self.db = db_connector if db_connector else DatabaseConnector()
        self.lineorder_count = int(6000000 * scale_factor)
        self.customer_count = int(30000 * scale_factor)
        self.supplier_count = int(2000 * scale_factor)
        self.part_count = int(200000 * scale_factor)
        self.date_count = 2556  # Fixed number of dates (7 years)
        
    def create_schema(self):
        """Create all SSB tables."""
        self.db.create_table("date_dim", SSBSchema.get_date_dim_schema())
        self.db.create_table("customer", SSBSchema.get_customer_schema())
        self.db.create_table("supplier", SSBSchema.get_supplier_schema())
        self.db.create_table("part", SSBSchema.get_part_schema())
        self.db.create_table("lineorder", SSBSchema.get_lineorder_schema())
        
    def generate_date_dim(self):
        """Generate date dimension data."""
        start_date = datetime(1992, 1, 1)
        
        for i in range(self.date_count):
            date = start_date + timedelta(days=i)
            date_key = int(date.strftime('%Y%m%d'))
            
            values = (
                date_key,
                date.strftime('%Y-%m-%d'),
                date.strftime('%A'),
                date.strftime('%B'),
                date.year,
                int(date.strftime('%Y%m')),
                date.strftime('%Y-%m'),
                date.weekday() + 1,
                date.day,
                date.timetuple().tm_yday,
                date.month,
                date.isocalendar()[1],
                'Winter' if date.month < 3 else 'Spring' if date.month < 6 else 'Summer' if date.month < 9 else 'Fall',
                1 if date.weekday() == 6 else 0,
                1 if date.day == [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][date.month-1] else 0,
                0,  # holiday flag (simplified)
                0 if date.weekday() >= 5 else 1
            )
            
            self.db.execute_query(
                "INSERT INTO date_dim VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values
            )
        
        print(f"Generated {self.date_count} date records")
    
    def generate_customers(self):
        """Generate customer data."""
        nations = ["UNITED STATES", "CHINA", "INDIA", "GERMANY", "BRAZIL", "JAPAN", "RUSSIA", "FRANCE", "CANADA", "MEXICO"]
        regions = ["AMERICA", "ASIA", "EUROPE"]
        segments = ["AUTOMOBILE", "BUILDING", "HOUSEHOLD", "MACHINERY", "FURNITURE"]
        
        for i in range(self.customer_count):
            nation = random.choice(nations)
            region = random.choice(regions)
            segment = random.choice(segments)
            
            values = (
                i,
                f"Customer#{i}",
                f"Address #{i}",
                f"City#{i % 250}",
                nation,
                region,
                f"Phone#{i}",
                segment
            )
            
            self.db.execute_query(
                "INSERT INTO customer VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                values
            )
        
        print(f"Generated {self.customer_count} customer records")
    
    def generate_suppliers(self):
        """Generate supplier data."""
        nations = ["UNITED STATES", "CHINA", "INDIA", "GERMANY", "BRAZIL", "JAPAN", "RUSSIA", "FRANCE", "CANADA", "MEXICO"]
        regions = ["AMERICA", "ASIA", "EUROPE"]
        
        for i in range(self.supplier_count):
            nation = random.choice(nations)
            region = random.choice(regions)
            
            values = (
                i,
                f"Supplier#{i}",
                f"Address #{i}",
                f"City#{i % 100}",
                nation,
                region,
                f"Phone#{i}"
            )
            
            self.db.execute_query(
                "INSERT INTO supplier VALUES (?, ?, ?, ?, ?, ?, ?)",
                values
            )
        
        print(f"Generated {self.supplier_count} supplier records")
    
    def generate_parts(self):
        """Generate part data."""
        categories = ["MFGR#1", "MFGR#2", "MFGR#3", "MFGR#4", "MFGR#5"]
        brands = ["BRAND#11", "BRAND#12", "BRAND#13", "BRAND#14", "BRAND#15"]
        colors = ["RED", "GREEN", "BLUE", "YELLOW", "BLACK", "WHITE", "PURPLE", "ORANGE"]
        containers = ["SM CASE", "LG CASE", "SM BOX", "LG BOX", "SM PACK", "LG PACK", "SM BAG", "LG BAG"]
        
        for i in range(self.part_count):
            category = random.choice(categories)
            brand = random.choice(brands)
            color = random.choice(colors)
            container = random.choice(containers)
            
            values = (
                i,
                f"Part#{i}",
                category,
                f"Category#{i % 5}",
                brand,
                color,
                f"Type#{i % 4}",
                random.randint(1, 50),
                container
            )
            
            self.db.execute_query(
                "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values
            )
        
        print(f"Generated {self.part_count} part records")
    
    def generate_lineorders(self):
        """Generate lineorder data with progress tracking."""
        priorities = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-LOW", "5-NOT URGENT"]
        ship_modes = ["RAIL", "TRUCK", "AIR", "MAIL", "SHIP", "FOX"]
        
        start_time = time.time()
        total_rows = 0
        batch_size = 1000
        
        print(f"Starting lineorder generation - target: {self.lineorder_count} orders")
        
        try:
            for i in range(self.lineorder_count):
                order_key = i
                num_lines = random.randint(1, 7)
                
                for line_num in range(1, num_lines + 1):
                    date_key = 19920101 + random.randint(0, self.date_count - 1)
                    cust_key = random.randint(0, self.customer_count - 1)
                    part_key = random.randint(0, self.part_count - 1)
                    supp_key = random.randint(0, self.supplier_count - 1)
                    
                    quantity = random.randint(1, 50)
                    price = random.randint(1000, 10000)
                    discount = random.randint(0, 10)
                    tax = random.randint(1, 8)
                    
                    values = (
                        order_key,
                        line_num,
                        cust_key,
                        part_key,
                        supp_key,
                        date_key,
                        random.choice(priorities),
                        str(random.randint(0, 1)),
                        quantity,
                        price,
                        price * quantity,
                        discount,
                        int(price * quantity * (1 - discount / 100)),
                        int(price * 0.8),
                        tax,
                        date_key + random.randint(1, 30),
                        random.choice(ship_modes)
                    )
                    
                    self.db.execute_query(
                        "INSERT INTO lineorder VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        values
                    )
                    total_rows += 1
                    
                    # Print progress periodically
                    if total_rows % batch_size == 0:
                        elapsed = time.time() - start_time
                        progress = (i + 1) / self.lineorder_count * 100
                        rate = total_rows / elapsed if elapsed > 0 else 0
                        print(f"Progress: {i+1}/{self.lineorder_count} orders ({progress:.1f}%) - "
                              f"{total_rows} rows at {rate:.0f} rows/sec")
            
            elapsed = time.time() - start_time
            print(f"Generated {total_rows} lineorder records in {elapsed:.2f} seconds ({total_rows/elapsed:.0f} rows/sec)")
        
        except Exception as e:
            print(f"Error generating lineorder data: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_all_data(self):
        """Generate all SSB data."""
        print(f"Generating SSB data with scale factor {self.scale_factor}")
        
        self.create_schema()
        self.generate_date_dim()
        self.generate_customers()
        self.generate_suppliers()
        self.generate_parts()
        self.generate_lineorders()
        
        print("Data generation complete!")
    
    def clear_all_data(self):
        """Clear all data from SSB tables."""
        for table in SSBSchema.get_all_tables():
            self.db.execute_query(f"DELETE FROM {table}")
        print("All data cleared")