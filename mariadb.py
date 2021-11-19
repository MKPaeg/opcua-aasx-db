import pymysql

class MariaDB:
    def __init__(self, host, port, user, password, db, db_schema, args):
        # Declare global variable
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = db
        self.db_schema = db_schema
        self.args = args
        
    def connect(self):
        self.conn = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password, db=self.db)    # Connection information
        self.cur = self.conn.cursor()
    
    def execute_sql(self, sql:str):
        self.cur.execute(sql)
        self.conn.commit()
        
    def create_table(self):
        date_time_sql = "ts DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
        value_type_sql = "DECIMAL(10,2)"
        temp_sql = f""
        
        for table_name, column_names in self.db_schema.items():
            if table_name not in self.args.table_list:
                continue
            
            for cl_sql in column_names:
                temp_sql += f"{cl_sql} {value_type_sql}, "
                
            sql = f"CREATE TABLE IF NOT EXISTS {table_name}" + f"({date_time_sql}, {temp_sql})"
            sql = sql[:-3] + ")"        # postprocess for last column
            print(f"{table_name} created..")
            self.execute_sql(sql)
        
    def delete_table(self):
        for table_name, column_names in self.db_schema.items():
            if table_name not in self.args.table_list:
                continue
            
            sql = f"DROP TABLE {table_name}"
            self.execute_sql(sql)
            print(f"{table_name} deleted..")
        
    def insert_value(self, table_name:str, column_name:str, val:float):
        sql = f"INSERT INTO {table_name}({column_name}) VALUES ({val})"
        self.execute_sql(sql)
        
    def insert_predicted(self, val:int):
        sql = f"INSERT INTO RESULT(RESULT_TOMATO_RIPENESS) VALUES ({val})"
        self.execute_sql(sql)
        