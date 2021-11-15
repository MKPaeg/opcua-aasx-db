from client import OPCUA_client
from mariadb import MariaDB
from time import sleep
import time

def json_load():
    import json
    
    with open('./initialization.json', 'r', encoding='utf-8') as json_data:
        table_dict = json.load(json_data)
    
    return table_dict

if __name__ == '__main__':
    table_dict = json_load()

    # cl = OPCUA_client("opc.tcp://localhost:53530/OPCUA/SimulationServer")
    cl = OPCUA_client("opc.tcp://localhost:51210/UA/SampleServer", table_dict['db_info'])
    # cl.connect()
    
    db = MariaDB(host='127.0.0.1', port=3306, user='root', password='13130132', db='opcua', table_dict=table_dict)
    db.connect()
    
    db.create_table()  
    exit()
    # while True:
    #     now = time.strftime('%Y-%m-%d %H:%M:%S')
    #     value = cl.get_value("Random")
        
        
    #     sql = "INSERT INTO random VALUES (%s, %s)"   # 걍 정수든, 실수든, 문자열이든 다 %s로 지정
    #     val = (now, value)
    #     db.execute_sql(sql,val)
    #     sleep(1)