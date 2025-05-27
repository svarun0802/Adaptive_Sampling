import re
import sqlparse
from sqlparse.sql import Statement, IdentifierList, Identifier, Function, Token
from sqlparse.tokens import Token as TokenType

class SQLQueryParser:
    def __init__(self):
        """Initialize the SQL query parser."""
        self.reset()
    
    def reset(self):
        """Reset parser state for a new query."""
        self.tables = []
        self.table_aliases = {}  # alias -> actual_table_name mapping
        self.join_conditions = []
        self.where_conditions = []
        self.group_by_columns = []
        self.select_columns = []
        self.aggregates = []
        self.order_by_columns = []
        self.limit = None
        self.query_type = None
    
    def parse(self, query):
        """
        Parse a SQL query and extract relevant components.
        
        Parameters:
        - query: SQL query string
        
        Returns:
        - Dictionary with parsed components
        """
        # Reset state for new query
        self.reset()
        
        # Normalize query
        query = self._normalize_query(query)
        
        # Parse using sqlparse
        try:
            parsed = sqlparse.parse(query)[0]
        except Exception as e:
            raise ValueError(f"Failed to parse query: {e}")
        
        # Extract components in order
        self._extract_select_clause(parsed)
        self._extract_from_clause(parsed)
        self._extract_join_clauses(parsed)
        self._extract_where_clause(parsed)
        self._extract_group_by_clause(parsed)
        self._extract_order_by_clause(parsed)
        self._extract_limit_clause(parsed)
        
        # Handle comma-separated joins (implicit joins)
        self._handle_implicit_joins()
        
        # Determine query type
        self.query_type = self._determine_query_type()
        
        # Build fingerprint for sample matching
        fingerprint = self._create_query_fingerprint()
        
        return {
            'tables': self.tables,
            'table_aliases': self.table_aliases,
            'join_conditions': self.join_conditions,
            'where_conditions': self.where_conditions,
            'group_by_columns': self.group_by_columns,
            'select_columns': self.select_columns,
            'aggregates': self.aggregates,
            'order_by_columns': self.order_by_columns,
            'limit': self.limit,
            'query_type': self.query_type,
            'fingerprint': fingerprint,
            'complexity_score': self._calculate_complexity_score()
        }
    
    def _normalize_query(self, query):
        """Normalize query for consistent parsing."""
        # Remove extra whitespace and ensure single spaces
        query = re.sub(r'\s+', ' ', query.strip())
        return query
    
    def _extract_select_clause(self, parsed):
        """Extract SELECT clause information including aggregates."""
        # Convert to string and extract SELECT clause
        query_str = str(parsed)
        
        # Find SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query_str, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return
        
        select_text = select_match.group(1).strip()
        
        # Split by comma outside of parentheses
        select_items = self._split_by_comma_outside_parens(select_text)
        
        for item in select_items:
            item = item.strip()
            if item:
                self._process_select_item(item)
    
    def _split_by_comma_outside_parens(self, text):
        """Split text by commas that are outside parentheses."""
        items = []
        current_item = ""
        paren_level = 0
        
        for char in text:
            if char == '(':
                paren_level += 1
                current_item += char
            elif char == ')':
                paren_level -= 1
                current_item += char
            elif char == ',' and paren_level == 0:
                if current_item.strip():
                    items.append(current_item.strip())
                current_item = ""
            else:
                current_item += char
        
        if current_item.strip():
            items.append(current_item.strip())
        
        return items
    
    def _process_select_item(self, item_text):
        """Process individual SELECT items to identify aggregates."""
        item_text = item_text.strip()
        
        # Check for aggregate functions - comprehensive regex
        agg_pattern = r'(COUNT|SUM|AVG|MIN|MAX|STDDEV|VARIANCE)\s*\(\s*([^)]*)\s*\)'
        agg_match = re.search(agg_pattern, item_text, re.IGNORECASE)
        
        if agg_match:
            func_name = agg_match.group(1).upper()
            argument = agg_match.group(2).strip()
            
            # Handle COUNT(*) special case
            if func_name == 'COUNT' and '*' in argument:
                argument = '*'
            
            # Extract alias - look for AS keyword or just trailing word
            alias_pattern = r'.*\)\s*(?:AS\s+)?(\w+)\s*$'
            alias_match = re.search(alias_pattern, item_text, re.IGNORECASE)
            alias = alias_match.group(1) if alias_match else None
            
            # Get the full expression without alias
            expr_pattern = r'(.+?)\s*(?:AS\s+\w+)?$'
            expr_match = re.search(expr_pattern, item_text, re.IGNORECASE)
            expression = expr_match.group(1).strip() if expr_match else item_text
            
            self.aggregates.append({
                'function': func_name,
                'argument': argument,
                'expression': expression,
                'alias': alias,
                'full_expression': item_text
            })
        else:
            # Regular column or expression
            self.select_columns.append(item_text)
    
    def _extract_from_clause(self, parsed):
        """Extract table names and aliases from FROM clause."""
        # Convert to string and extract FROM clause
        query_str = str(parsed)
        
        # Find FROM clause (until JOIN, WHERE, GROUP, ORDER, or LIMIT)
        from_pattern = r'FROM\s+(.*?)(?:\s+(?:INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|JOIN|WHERE|GROUP\s+BY|ORDER\s+BY|LIMIT)|\s*$)'
        from_match = re.search(from_pattern, query_str, re.IGNORECASE)
        
        if from_match:
            from_text = from_match.group(1).strip()
            
            # Split by comma to handle multiple tables
            table_refs = [ref.strip() for ref in from_text.split(',') if ref.strip()]
            
            for ref in table_refs:
                self._process_table_reference(ref)
    
    def _process_table_reference(self, reference):
        """Process table reference which may include alias."""
        reference = reference.strip()
        
        # Match patterns: "table", "table alias", "table AS alias"
        patterns = [
            r'^(\w+)\s+AS\s+(\w+)$',  # table AS alias
            r'^(\w+)\s+(\w+)$',       # table alias
            r'^(\w+)$'                # just table
        ]
        
        table_name = None
        alias = None
        
        for pattern in patterns:
            match = re.match(pattern, reference, re.IGNORECASE)
            if match:
                table_name = match.group(1)
                alias = match.group(2) if len(match.groups()) > 1 else None
                break
        
        if table_name:
            # Add to tables if not already present
            if table_name not in self.tables:
                self.tables.append(table_name)
            
            # Store alias mapping
            if alias:
                self.table_aliases[alias] = table_name
    
    def _extract_join_clauses(self, parsed):
        """Extract JOIN conditions and table references."""
        query_str = str(parsed)
        
        # Find all JOIN clauses
        join_pattern = r'((?:INNER\s+)?(?:LEFT\s+|RIGHT\s+|FULL\s+)?JOIN)\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+ON\s+([^)]+?)(?=\s+(?:JOIN|WHERE|GROUP|ORDER|LIMIT)|$)'
        
        matches = re.finditer(join_pattern, query_str, re.IGNORECASE)
        
        for match in matches:
            join_type = match.group(1).upper()
            table_name = match.group(2)
            alias = match.group(3)
            condition = match.group(4).strip()
            
            # Process table reference
            ref = f"{table_name} {alias}" if alias else table_name
            self._process_table_reference(ref)
            
            # Parse join condition
            join_condition = self._parse_join_condition_text(condition)
            if join_condition:
                join_condition['join_type'] = join_type
                self.join_conditions.append(join_condition)
    
    def _parse_join_condition_text(self, condition_text):
        """Parse join condition from text."""
        # Handle format: table1.col1 = table2.col2
        condition_text = condition_text.strip()
        
        # Match table.column = table.column pattern
        match = re.match(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', condition_text)
        if match:
            left_table, left_col, right_table, right_col = match.groups()
            
            # Resolve aliases to actual table names
            left_table = self.table_aliases.get(left_table, left_table)
            right_table = self.table_aliases.get(right_table, right_table)
            
            return {
                'left_table': left_table,
                'left_column': left_col,
                'right_table': right_table,
                'right_column': right_col,
                'condition_text': condition_text
            }
        
        return None
    
    def _parse_join_condition(self, tokens, start_index):
        """Parse a single join condition from tokens."""
        condition_text = ""
        i = start_index
        
        # Collect tokens until we hit a keyword
        while i < len(tokens):
            token = tokens[i]
            if token.ttype is TokenType.Keyword and token.value.upper() in [
                'JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'WHERE', 'GROUP', 'ORDER', 'LIMIT'
            ]:
                break
            condition_text += token.value
            i += 1
        
        return self._parse_join_condition_text(condition_text)
    
    def _extract_where_clause(self, parsed):
        """Extract WHERE conditions."""
        query_str = str(parsed)
        
        # Find WHERE clause
        where_pattern = r'WHERE\s+(.*?)(?:\s+(?:GROUP\s+BY|ORDER\s+BY|LIMIT)|\s*$)'
        where_match = re.search(where_pattern, query_str, re.IGNORECASE | re.DOTALL)
        
        if where_match:
            where_text = where_match.group(1).strip()
            if where_text:
                self.where_conditions.append(where_text)
    
    def _handle_implicit_joins(self):
        """Handle comma-separated tables as implicit joins by analyzing WHERE clause."""
        if len(self.tables) > 1 and not self.join_conditions and self.where_conditions:
            # Look for join conditions in WHERE clause
            where_text = ' '.join(self.where_conditions)
            
            # Find patterns like table1.col = table2.col
            join_pattern = r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
            matches = re.findall(join_pattern, where_text, re.IGNORECASE)
            
            for match in matches:
                left_table, left_col, right_table, right_col = match
                
                # Resolve aliases
                left_table = self.table_aliases.get(left_table, left_table)
                right_table = self.table_aliases.get(right_table, right_table)
                
                # Check if both tables exist
                if left_table in self.tables and right_table in self.tables:
                    self.join_conditions.append({
                        'left_table': left_table,
                        'left_column': left_col,
                        'right_table': right_table,
                        'right_column': right_col,
                        'condition_text': f"{left_table}.{left_col} = {right_table}.{right_col}",
                        'join_type': 'IMPLICIT'
                    })
    
    def _extract_group_by_clause(self, parsed):
        """Extract GROUP BY columns."""
        query_str = str(parsed)
        
        # Find GROUP BY clause
        group_by_pattern = r'GROUP\s+BY\s+(.*?)(?:\s+(?:HAVING|ORDER\s+BY|LIMIT)|\s*$)'
        group_by_match = re.search(group_by_pattern, query_str, re.IGNORECASE)
        
        if group_by_match:
            group_by_text = group_by_match.group(1).strip()
            if group_by_text:
                # Split by comma
                columns = [col.strip() for col in group_by_text.split(',') if col.strip()]
                self.group_by_columns.extend(columns)
    
    def _extract_order_by_clause(self, parsed):
        """Extract ORDER BY columns."""
        query_str = str(parsed)
        
        # Find ORDER BY clause
        order_by_pattern = r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT|\s*$)'
        order_by_match = re.search(order_by_pattern, query_str, re.IGNORECASE)
        
        if order_by_match:
            order_by_text = order_by_match.group(1).strip()
            if order_by_text:
                # Split by comma and parse each column
                columns = [col.strip() for col in order_by_text.split(',') if col.strip()]
                for col in columns:
                    parts = col.split()
                    if parts:
                        column_info = {
                            'column': parts[0],
                            'direction': parts[1].upper() if len(parts) > 1 and parts[1].upper() in ['ASC', 'DESC'] else 'ASC'
                        }
                        self.order_by_columns.append(column_info)
    
    def _extract_limit_clause(self, parsed):
        """Extract LIMIT value."""
        query_str = str(parsed)
        
        # Find LIMIT clause
        limit_pattern = r'LIMIT\s+(\d+)'
        limit_match = re.search(limit_pattern, query_str, re.IGNORECASE)
        
        if limit_match:
            try:
                self.limit = int(limit_match.group(1))
            except ValueError:
                pass
    
    def _determine_query_type(self):
        """Classify the query type based on its characteristics."""
        if self.aggregates or self.group_by_columns:
            return 'analytical'
        elif len(self.tables) > 1 or self.join_conditions:
            return 'join'
        else:
            return 'simple'
    
    def _create_query_fingerprint(self):
        """Create a unique fingerprint for the query structure."""
        # Sort tables for consistent fingerprinting
        sorted_tables = sorted(self.tables)
        
        # Create join pattern signature
        join_signature = []
        for join in self.join_conditions:
            # Create a canonical representation of the join
            tables_in_join = sorted([join['left_table'], join['right_table']])
            join_sig = f"{tables_in_join[0]}⋈{tables_in_join[1]}"
            join_signature.append(join_sig)
        
        join_signature = sorted(join_signature)
        
        # Create fingerprint
        fingerprint = {
            'tables': tuple(sorted_tables),
            'joins': tuple(join_signature),
            'has_aggregates': bool(self.aggregates),
            'has_group_by': bool(self.group_by_columns)
        }
        
        return fingerprint
    
    def _calculate_complexity_score(self):
        """Calculate a complexity score for the query."""
        score = 0
        
        # Base score for number of tables
        score += len(self.tables) * 10
        
        # Add for joins
        score += len(self.join_conditions) * 15
        
        # Add for aggregates
        score += len(self.aggregates) * 5
        
        # Add for WHERE conditions
        score += len(self.where_conditions) * 3
        
        # Add for GROUP BY
        score += len(self.group_by_columns) * 5
        
        return score
    
    def format_joins_for_sampling(self):
        """
        Format join conditions in the format expected by WanderJoin.
        
        Returns:
        - List of tuples: [(table1, col1, table2, col2), ...]
        """
        formatted_joins = []
        
        for join in self.join_conditions:
            formatted_joins.append((
                join['left_table'],
                join['left_column'],
                join['right_table'],
                join['right_column']
            ))
        
        return formatted_joins
    
    def __str__(self):
        """String representation for debugging."""
        return f"""
        Query Type: {self.query_type}
        Tables: {self.tables}
        Aliases: {self.table_aliases}
        Joins: {self.join_conditions}
        Aggregates: {self.aggregates}
        Group By: {self.group_by_columns}
        Order By: {self.order_by_columns}
        Select Columns: {self.select_columns}
        Complexity: {self._calculate_complexity_score()}
        """