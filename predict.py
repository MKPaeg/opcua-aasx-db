import os
import glob
import pandas as pd
import torch
from torchvision.io import read_image
import torch.utils.data as data

from torchvision import transforms
from PIL import Image
import numpy as np
from client import OPCUA_client
from mariadb import MariaDB
from time import sleep
import time
from utils.io_preprocess import preprocess_swir
from utils.mobilenet import MobileNet
# from utils.volumerender_sim import  volumerender_sim


def json_load():
    import json
    
    with open('./initialization.json', 'r', encoding='utf-8') as json_data:
        table_dict = json.load(json_data)
    
    return table_dict

def make_data_path(train_path):
    file = glob.glob(train_path + "\*.raw")
    train_img_list, train_label = list(), list()
    for item in file:
        img_path = item
        train_img_list.append(img_path)

        img_label = int(str(item.split("\\")[-1].split(".")[0])[0])
        train_label.append(img_label)

    return train_img_list, train_label


class ImageTransform():

    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),

        ])

    def __call__(self, img):
        return self.data_transform(img)


class Sup_Img_Dataset(data.Dataset):

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]

        img = preprocess_swir(img_path, 256)

        img_resize = img.reshape(256, 256, 256)

        img_transformed = self.transform(img_resize)

        label = str(img_path.split("\\")[-1].split(".")[0])[0]

        img_transformed = img_transformed
        label = int(label)
        return img_transformed, label

def load_swir(path):
    img = preprocess_swir(path, 256)
    img_resize = img.reshape(256, 256, 256)
    img_transformed = ImageTransform(img_resize)
    return img_transformed

if __name__ =="__main__":
    table_dict = json_load()

    # cl = OPCUA_client("opc.tcp://localhost:53530/OPCUA/SimulationServer")
    cl = OPCUA_client("opc.tcp://localhost:51210/UA/SampleServer", table_dict['db_info'])
    # cl.connect()
    
    db = MariaDB(host='127.0.0.1', port=3306, user='root', password='13130132', db='opcua', table_dict=table_dict)
    db.connect()
    
    # train_path = "F:\_data\_Food\pytorc/h_tomato_v2"
    # train_path = "C:\Users\hojun_window\Desktop\Work\오즈레이AAS\_tomato"
    train_path = "C:/Users/hojun_window/Desktop/Work/오즈레이AAS/_tomato"
    file = glob.glob(train_path + "/**/*.raw", recursive=True)
    # print(file)
    test_num = 78
    model_path = "./model_v3"
    epoch = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PATH = os.path.join(model_path, f"tomato_ripeness_{epoch}_" + ".pt")

    with torch.no_grad():
         test_dataloader = Sup_Img_Dataset(file, ImageTransform())
         
         print(f"Total test length: {len(file)}")
         for num in range(len(file)):
            transformed_img, label = test_dataloader[num]
            # volr.volumerender_sim(transformed_img.numpy(), 256, 256, 256)

            model = MobileNet().to(device)
            input_expands_tensor = transformed_img.unsqueeze(0) #[ch x with x height]-> [1 xch x with x height]
            model(input_expands_tensor.cuda())

            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint["model_state_dict"])

            output = model(input_expands_tensor.cuda())
            _, predicted = torch.max(output, 1)
            pred = np.squeeze(predicted.cpu().numpy())
            print(f"predicted :{pred}, Ground Truth :{label}")
            # db.insert_predicted(pred)
            db.insert_value(table_name="RESULT", column_name="RESULT_TOMATO_RIPENESS", val=pred)
            print(f"Saved to RESULT table")


    """`
    
    
    for i in range(50):
        with torch.no_grad():
            test_dataloader = Sup_Img_Dataset(file, ImageTransform())
            transformed_img, label = test_dataloader[i]
            # volr.volumerender_sim(transformed_img.numpy(), 256, 256, 256)

            model = MobileNet().to(device)
            input_expands_tensor = transformed_img.unsqueeze(0) # [ch x with x height]-> [1 x ch x with x height]
            model(input_expands_tensor.cuda())

            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint["model_state_dict"])

            output = model(input_expands_tensor.cuda())

            _, predicted = torch.max(output, 1)
            pred = np.squeeze(predicted.cpu().numpy())
            print(f"predicted :{pred}, Ground Truth :{label}")
    """
