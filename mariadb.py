import pymysql

class MariaDB:
    def __init__(self, host, port, user, password, db, table_dict):
        # Declare global variable
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = db
        self.table_dict = table_dict
        
    def connect(self):
        self.conn = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password, db=self.db)    # Connection information
        self.cur = self.conn.cursor()
    
    def execute_sql(self, sql:str):
        self.cur.execute(sql)
        self.conn.commit()
        
    def create_table(self):
        # sql = [f"IF NOT EXISTS RGBIMAGE CREATE TABLE RGBIMAGE(RGB_WIDTH DECIMAL(10,2), RGB_HEIGHT DECIMAL(10,2), RGB_CHANNEL DECIMAL(10,2))"]
        sql = f"CREATE TABLE IF NOT EXISTS RGBIMAGE(RGB_WIDTH DECIMAL(10,2), RGB_HEIGHT DECIMAL(10,2), RGB_CHANNEL DECIMAL(10,2))"
        self.execute_sql(sql)
        
        sql = f"CREATE TABLE IF NOT EXISTS DEPTHIMAGE(DEPTH_WIDTH DECIMAL(10,2), DEPTH_HEIGHT DECIMAL(10,2), DEPTH_AVG_WIDTH DECIMAL(10,2))"
        self.execute_sql(sql)
        
        sql = f"CREATE TABLE IF NOT EXISTS HSIIMAGE(HSI_CROP_WIDTH DECIMAL(10,2), HSI_CROP_HEIGHT DECIMAL(10,2), HSI_LABEL_BAND_NUM DECIMAL(10,2), HSI_INTERLEAVE DECIMAL(10,2), HSI_DATA_TYPE DECIMAL(10,2), HSI_BANDS DECIMAL(10,2), HSI_DATASIZE DECIMAL(10,2), HSI_BAND_STRIDE DECIMAL(10,2))"
        self.execute_sql(sql)
        
        sql = f"CREATE TABLE IF NOT EXISTS RESULT(ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, RESULT_TOMATO_RIPENESS DECIMAL(10,2), RESULT_TOMATO_SP DECIMAL(10,2))"
        # sql = f"CREATE TABLE IF NOT EXISTS RESULT(ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP)"
        self.execute_sql(sql)
        
    def insert_value(self, table_name:str, column_name:str, val:float):
        sql = f"INSERT INTO {table_name}({column_name}) VALUES ({val})"
        self.execute_sql(sql)
        
    def insert_predicted(self, val:int):
        sql = f"INSERT INTO RESULT(RESULT_TOMATO_RIPENESS) VALUES ({val})"
        self.execute_sql(sql)
        