from opcua import Client, ua
from opcua.common.ua_utils import get_node_children, get_node_supertype
from opcua.ua import ua_binary as uabin
from opcua.common.methods import call_method
from time import sleep

def get_aasx_db_schema(client:Client, name_space:str): # 이 함수를 재귀적으로 호출하면서 schema 구조 생성하려 했으나, 기타 잡다한 정보들이 opcua 서버에 너무 많음
    
    def get_property_schema(table_list:list):
        property_list = []
        
        for table in table_list:
            tp_list = table.get_children(refs=47)
            tp_list = list(set(tp_list))
            tp_list = [node for node in tp_list if node.get_display_name().Text != "SemanticId" and node.get_display_name().Text != "Identification"]
            property_list.append(tp_list)
            
        return property_list
    
    def create_schema_dict(property_list:list):
        schema_dict = {}
        print(f"property list -> {property_list}")
        for property in property_list:
            table_name = property[0].get_parent().get_display_name().Text  # Every property has the same parents in the lists.
            tp = [_pro_perty.get_display_name().Text for _pro_perty in property]
            schema_dict[table_name] = tp
        
        print(schema_dict)
        return schema_dict
        
    DB_TABLE = client.get_node(name_space)
    tp_list = DB_TABLE.get_children(refs=47)  # choose node that has property
    tp_list = list(set(tp_list))    # delete duplicated
    table_list = [node for node in tp_list if node.get_display_name().Text != "SemanticId" and node.get_display_name().Text != "Identification"]
    
    property_list = get_property_schema(table_list)
    schema_dict = create_schema_dict(property_list)
    
    return schema_dict
    
class OPCUA_client:
    
    # I don't use with statement to use method with opcua server communication
    def __init__(self, IP:str):
        # self.table_list = table_dict
        client = Client(IP)
        client.connect()
        
        BaseNameSpace = "ns=3;s=AASROOT.DATA.DB_TABLE"
        
        print("\nConnection Success..")
        print(f"IP: {IP}")
        
        db_schema = get_aasx_db_schema(client, BaseNameSpace)
    
        self.db_schema = db_schema
    
    def get_db_schema(self):
        return self.db_schema

if __name__ == '__main__':
    cl = OPCUA_client("opc.tcp://localhost:51210/UA/SampleServer")
    
    # with Client("opc.tcp://localhost:51210/UA/SampleServer") as client:
    #     root = client.get_root_node()
    #     print("Root node is: ", root)
    #     objects = client.get_objects_node()
    #     print("Objects node is: ", objects)
        
    #     Simulation_folder = objects.get_children()[2]
    #     Random_value = client.get_node("ns=3;i=1002")
    #     print(Random_value.get_browse_name())
        
    #     while(True):
    #         print(Random_value.get_value())
    #         sleep(1)