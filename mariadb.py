import pymysql

class MariaDB:
    def __init__(self, host, port, user, password, db, table_dict, args):
        # Declare global variable
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = db
        self.table_dict = table_dict
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
        
        for table_name, column_names in self.table_dict['db_info'].items():
            if table_name not in self.args.table_list:
            # if table_name in "DEPTHIMAGE" or table_name == "HSIIMAGE":
                continue
            
            for cl_sql in column_names:
                temp_sql += f"{cl_sql} {value_type_sql}, "
                
            sql = f"CREATE TABLE IF NOT EXISTS {table_name}" + f"({date_time_sql}, {temp_sql})"
            sql = sql[:-3] + ")"        # postprocess for last column
            print(sql)
            self.execute_sql(sql)
        
        # sql = f"CREATE TABLE IF NOT EXISTS DEPTHIMAGE(ts DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, DEPTH_WIDTH DECIMAL(10,2), DEPTH_HEIGHT DECIMAL(10,2), DEPTH_AVG_WIDTH DECIMAL(10,2))"
        # self.execute_sql(sql)
        
        # sql = f"CREATE TABLE IF NOT EXISTS HSIIMAGE(ts DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, HSI_CROP_WIDTH DECIMAL(10,2), HSI_CROP_HEIGHT DECIMAL(10,2), HSI_LABEL_BAND_NUM DECIMAL(10,2), HSI_INTERLEAVE DECIMAL(10,2), HSI_DATA_TYPE DECIMAL(10,2), HSI_BANDS DECIMAL(10,2), HSI_DATASIZE DECIMAL(10,2), HSI_BAND_STRIDE DECIMAL(10,2))"
        # self.execute_sql(sql)
        
        # sql = f"CREATE TABLE IF NOT EXISTS RESULT(ts DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, RESULT_TOMATO_RIPENESS DECIMAL(10,2), RESULT_TOMATO_SP DECIMAL(10,2))"
        # self.execute_sql(sql)
        
    def delete_table(self):
        for table_name in self.table_dict['db_info'].keys():
            if table_name == "DEPTHIMAGE" or table_name == "HSIIMAGE":
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
        