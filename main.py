from client import OPCUA_client
from mariadb import MariaDB
from time import sleep
import sys
import os
import time
import argparse
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
import onnx
import onnxruntime

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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def predict(args):
    train_path = "D:/Work/오즈레이AAS/_tomato"
    file = glob.glob(train_path + "/**/*.raw", recursive=True)
    print(f"Total test length: {len(file)}")

    test_num = 78
    model_path = "./model_v3"
    epoch = 30 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PATH = os.path.join(model_path, f"tomato_ripeness_{epoch}_" + ".pt")

    with torch.no_grad():
        test_dataloader = Sup_Img_Dataset(file, ImageTransform())
        model = MobileNet().to(device)
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        if args.onnx :
            if not os.path.isfile(model_path + '/weight.onnx'):
                x = torch.randn(1, 256, 256, 256, requires_grad=True).cuda() # 배치사이즈는 1로 생각
                torch_out = model(x)
                torch.onnx.export(
                    model,               # 실행될 모델
                    x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능, 또한 랜덤이여도 상관 없음)
                    "weight.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                    export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                    opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                    do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
                    input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                    output_names = ['output'], # 모델의 출력값을 가리키는 이름
                    dynamic_axes={
                        'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                        'output' : {0 : 'batch_size'}})
        
            ort_session = onnxruntime.InferenceSession("weight.onnx")

        # ONNX 런타임에서 계산된 결과값
        # 여기 코드가 핵심임. 실질적으로 onnx 모델을 통한 inference가 이루어지는 부분
        # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_expands_tensor.cuda())}
        # ort_outs = ort_session.run(None, ort_inputs)[0]        # 이게 그냥 pytorch 모델의 인자에 값 넣는것과 같음
        # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        # print("Exported model has been tested with ONNXRuntime, and the result looks good!")

        for num in range(len(file)):
            transformed_img, label = test_dataloader[num]
            # volr.volumerender_sim(transformed_img.numpy(), 256, 256, 256)

            input_expands_tensor = transformed_img.unsqueeze(0) #[ch x with x height]-> [1 xch x with x height]
            # model(input_expands_tensor.cuda())

            if args.onnx:
                # ONNX 런타임에서 계산된 결과값
                # 여기 코드가 핵심임. 실질적으로 onnx 모델을 통한 inference가 이루어지는 부분

                ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_expands_tensor.cuda())}
                ort_outs = ort_session.run(None, ort_inputs)[0]        # 이게 그냥 pytorch 모델의 인자에 값 넣는것과 같음
                # print(f"ort_outs type: {type(ort_outs)} \n ort_outs: {ort_outs}")
                predicted = np.argmax(ort_outs)
                # _, predicted = torch.max(ort_outs, 1)
                # pred = np.squeeze(predicted.cpu().numpy())
                print(f"####### ONNX Inference ###### \n predicted :{predicted}, Ground Truth :{label}")
                # db.insert_predicted(pred)
                db.insert_value(table_name="RESULT", column_name="RESULT_TOMATO_RIPENESS", val=predicted)
                print(f"Saved to RESULT table")

            else: # onnx로 추론하는것이 아니라면
                output = model(input_expands_tensor.cuda())
                _, predicted = torch.max(output, 1)
                pred = np.squeeze(predicted.cpu().numpy())
                print(f"predicted :{pred}, Ground Truth :{label}")
                # db.insert_predicted(pred)
                db.insert_value(table_name="RESULT", column_name="RESULT_TOMATO_RIPENESS", val=pred)
                print(f"Saved to RESULT table")
    
if __name__ == '__main__':
    # 인자값 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='Set option')
    
    # 입력받을 인자값 등록
    parser.add_argument('--refresh', required=False, default=False, action='store_true', help="Choose whether refresh or net")
    parser.add_argument('--predict', required=False, default=False, action='store_true', help="Predict and insert to mariaDB")
    parser.add_argument('--table_list', required=False, nargs='+', type=str, help="Choose table to use / RGBIMAGE, DEPTHIMAGE, HSIIMAGE, RESULT")
    parser.add_argument('--onnx', required=False, default=False, action='store_true', help="Choose whether use onnx or not")
    
    args = parser.parse_args()
    
    # cl = OPCUA_client("opc.tcp://localhost:53530/OPCUA/SimulationServer")
    cl = OPCUA_client("opc.tcp://localhost:51210/UA/SampleServer")
    
    db_schema = cl.get_db_schema()            
    
    if args.table_list == None:     # if table_list empty
        args.table_list = list(db_schema.keys())
    
    print(f"Chosen table list => {args.table_list}")
    
    db = MariaDB(host='127.0.0.1', port=3306, user='root', password='13130132', db='opcua', db_schema=db_schema, args = args)       # MariaDB 연결
    db.connect()
    
    if args.refresh:
        db.delete_table()
        db.create_table()
        
    else:
        db.create_table()
    
    if args.predict:            
        print(f"Start predict..")
        predict(args)           # predict 함수 실행하여 실시간 inference 및 DB에 값 저장 진행
        
    db.close()
    cl.close()
    print(f"Server closed..")