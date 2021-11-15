import os
import glob
import pandas as pd
import torch
from torchvision.io import read_image
import torch.utils.data as data

from torchvision import transforms
from PIL import Image
import numpy as np

from utils.io_preprocess import preprocess_swir
from utils.mobilenet import MobileNet

def make_data_path(train_path):
    file = glob.glob(train_path + "\*.raw")
    train_img_list, train_label = list(), list()
    for item in file:
        img_path = item
        train_img_list.append(img_path)

        img_label = int(str(item.split("\\")[-1].split(".")[0])[0])
        train_label.append(img_label)

    return train_img_list, train_label


# train_list, train_label = make_data_path(train_path)

# print(train_list)
# print(train_label)


## 이미지 전처리 클래스
class ImageTransform():
    '''
    __init__은 객체 생성될 때 불러와짐 / __call__은 인스턴스 생성될 때 불러와짐
    __call__함수는 이 클래스의 객체가 함수처럼 호출되면 실행되는 함수임...
    '''
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
        img_path = self.file_list[index]  # 데이터셋에서 파일 하나를 특정
        # img = np.load(img_path)
        # print(f"img_path :{img_path}")
        img = preprocess_swir(img_path, 256)
        # print(f"img shape :{img.shape}")
        img_resize = img.reshape(256, 256, 256)
        # print(f"img_resize shape :{img_resize.shape}")
        img_transformed = self.transform(img_resize)

        # label = img_path.split("\\")[-2:-1][0]
        label = str(img_path.split("\\")[-1].split(".")[0])[0] # 파일명으로부터 라벨명 추출

        img_transformed = img_transformed
        label = int(label)
        return img_transformed, label

def load_swir(path):
    img = preprocess_swir(path, 256)
    img_resize = img.reshape(256, 256, 256)
    img_transformed = ImageTransform(img_resize)

    return img_transformed

if __name__ =="__main__":
    import volumerender_sim as volr
    train_path = "F:\_data\_Food\pytorch_tomato_v2"
    file = glob.glob(train_path + "\*.raw")
    # print(file)
    train_dataloader = Sup_Img_Dataset(file, ImageTransform())

    transformed_img, label = train_dataloader[130]
    print(f"label:{label}")
    print(f"len of transformed_img :{len(train_dataloader)}")
    print(f"type of transformed_img :{type(transformed_img)}")
    print(f"size of tranformed_img tensor :{transformed_img.size()}")
    print(f"size of tranformed_img tensor :{transformed_img.unsqueeze(0).shape}")
    # volr.volumerender_sim(transformed_img.numpy(), 256, 256, 256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNet().to(device)
    input_expands_tensor = transformed_img.unsqueeze(0)
    model(input_expands_tensor.cuda())
    print(model)

