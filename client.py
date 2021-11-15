from opcua import Client, ua
from opcua.ua import ua_binary as uabin
from opcua.common.methods import call_method
from time import sleep

class OPCUA_client:
    
    # I don't use with statement to use method with opcua server communication
    def __init__(self, IP:str, table_dict:dict):
        self.table_list = table_dict
        client = Client(IP)
        client.connect()
        
        print("\n Connection Success..")
        print(f"IP: {IP}")
        
        # root = client.get_root_node()
        print("Information of OPCUA-Server node")
        # print(client.get_node('ns=3'))
        # exit()
        for table_name, columns_name in table_dict.items():
            print(f'table_name: {table_name}')
            for column in columns_name:
                print('ns=3;' + 's=AASROOT.DATA.DB_TABLE.' + table_name + '.' + column)
                print(client.get_node('ns=3;' + 's=AASROOT.DATA.DB_TABLE.' + table_name + '.' + column + '.Value').get_value())
                # node = client.get_node('ns=3;' + 's=AASROOT.DATA.DB_TABLE.' + table_name + '.' + column)
                print()
        
        # objects = client.get_objects_node()        
        # AASROOT = objects.get_child("AASROOT")
        
    def connect(self):
        while(True):
            print(Random_value.get_value())
            sleep(1)
    
    def get_value(self, display_name:str):
        result = 0
        if display_name == "Random":
            result = self.random_value.get_value()
            
        elif display_name == "Counter":
            pass
        else:
            pass
        
        return result

if __name__ == '__main__':
    
    
    with Client("opc.tcp://localhost:53530/OPCUA/SimulationServer") as client:
        root = client.get_root_node()
        print("Root node is: ", root)
        objects = client.get_objects_node()
        print("Objects node is: ", objects)
        
        Simulation_folder = objects.get_children()[2]
        Random_value = client.get_node("ns=3;i=1002")
        print(Random_value.get_browse_name())
        
        while(True):
            print(Random_value.get_value())
            sleep(1)