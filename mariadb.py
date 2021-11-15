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
        # sql = [f"IF NOT EXISTS RGBIMAGE CREATE TABLE RGBIMAGE(RGB_WIDTH VARCHAR(10), RGB_HEIGHT VARCHAR(10), RGB_CHANNEL VARCHAR(10))"]
        sql = f"CREATE TABLE IF NOT EXISTS RGBIMAGE(RGB_WIDTH VARCHAR(10), RGB_HEIGHT VARCHAR(10), RGB_CHANNEL VARCHAR(10))"
        self.execute_sql(sql)
        
        sql = f"CREATE TABLE IF NOT EXISTS DEPTHIMAGE(DEPTH_WIDTH VARCHAR(10), DEPTH_HEIGHT VARCHAR(10), DEPTH_AVG_WIDTH VARCHAR(10))"
        self.execute_sql(sql)
        
        sql = f"CREATE TABLE IF NOT EXISTS HSIIMAGE(HSI_CROP_WIDTH VARCHAR(10), HSI_CROP_HEIGHT VARCHAR(10), HSI_LABEL_BAND_NUM VARCHAR(10), HSI_INTERLEAVE VARCHAR(10), HSI_DATA_TYPE VARCHAR(10), HSI_BANDS VARCHAR(10), HSI_DATASIZE VARCHAR(10), HSI_BAND_STRIDE VARCHAR(10))"
        self.execute_sql(sql)
        
        sql = f"CREATE TABLE IF NOT EXISTS RESULT(RESULT_TOMATO_RIPENESS VARCHAR(10), RESULT_TOMATO_SP VARCHAR(10))"
        self.execute_sql(sql)
        
    def insert_predicted(self, val:int):
        sql = f"INSERT INTO RESULT(RESULT_TOMATO_RIPENESS) VALUES ({val})"
        self.execute_sql(sql)
        
            