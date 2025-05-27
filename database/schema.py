class SSBSchema:
    """Schema definitions for Star Schema Benchmark."""
    
    @staticmethod
    def get_date_dim_schema():
        return """
            d_datekey INTEGER PRIMARY KEY,
            d_date VARCHAR,
            d_dayofweek VARCHAR,
            d_month VARCHAR,
            d_year INTEGER,
            d_yearmonthnum INTEGER,
            d_yearmonth VARCHAR,
            d_daynuminweek INTEGER,
            d_daynuminmonth INTEGER,
            d_daynuminyear INTEGER,
            d_monthnuminyear INTEGER,
            d_weeknuminyear INTEGER,
            d_sellingseason VARCHAR,
            d_lastdayinweekfl INTEGER,
            d_lastdayinmonthfl INTEGER,
            d_holidayfl INTEGER,
            d_weekdayfl INTEGER
        """
    
    @staticmethod
    def get_customer_schema():
        return """
            c_custkey INTEGER PRIMARY KEY,
            c_name VARCHAR,
            c_address VARCHAR,
            c_city VARCHAR,
            c_nation VARCHAR,
            c_region VARCHAR,
            c_phone VARCHAR,
            c_mktsegment VARCHAR
        """
    
    @staticmethod
    def get_supplier_schema():
        return """
            s_suppkey INTEGER PRIMARY KEY,
            s_name VARCHAR,
            s_address VARCHAR,
            s_city VARCHAR,
            s_nation VARCHAR,
            s_region VARCHAR,
            s_phone VARCHAR
        """
    
    @staticmethod
    def get_part_schema():
        return """
            p_partkey INTEGER PRIMARY KEY,
            p_name VARCHAR,
            p_mfgr VARCHAR,
            p_category VARCHAR,
            p_brand VARCHAR,
            p_color VARCHAR,
            p_type VARCHAR,
            p_size INTEGER,
            p_container VARCHAR
        """
    
    @staticmethod
    def get_lineorder_schema():
        return """
            lo_orderkey INTEGER,
            lo_linenumber INTEGER,
            lo_custkey INTEGER,
            lo_partkey INTEGER,
            lo_suppkey INTEGER,
            lo_orderdate INTEGER,
            lo_orderpriority VARCHAR,
            lo_shippriority VARCHAR,
            lo_quantity INTEGER,
            lo_extendedprice INTEGER,
            lo_ordtotalprice INTEGER,
            lo_discount INTEGER,
            lo_revenue INTEGER,
            lo_supplycost INTEGER,
            lo_tax INTEGER,
            lo_commitdate INTEGER,
            lo_shipmode VARCHAR,
            PRIMARY KEY (lo_orderkey, lo_linenumber)
        """
    
    @staticmethod
    def get_sample_metadata_schema():
        """Schema for storing information about materialized samples."""
        return """
            sample_id VARCHAR PRIMARY KEY,
            source_tables VARCHAR,
            join_conditions VARCHAR,
            sample_size INTEGER,
            creation_time TIMESTAMP,
            last_used TIMESTAMP,
            query_count INTEGER DEFAULT 0,
            error_stats VARCHAR
        """
    
    @staticmethod
    def get_all_tables():
        """Get a list of all SSB tables."""
        return ['date_dim', 'customer', 'supplier', 'part', 'lineorder']