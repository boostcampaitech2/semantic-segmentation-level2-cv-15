import os
import time
import json
import warnings

from eda import EDA_S

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from setting import *
from utils import random_set
from dataset import *
from train import *
from inference import *
#from inference import infer
import cv2

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.patches import Patch
import webcolors

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchsummary

import sys


if __name__ == '__main__':
    plt.rcParams['axes.grid'] = False

    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())

# GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda" if torch.cuda.is_available() else "cpu"

    conf = JsonConfigFileManager("./config.json")
    config = conf.values

    ex_id = config["name"]["id"]
    model_name = config["name"]["model"]

    batch_size = config["param"]["batch_size"]
    num_epochs = config["param"]["num_epochs"]
    learning_rate = config["param"]["learning_rate"]

    dataset_path = config["path"]["dataset_path"]
    train_path = config["path"]["train_path"]
    val_path = config["path"]["val_path"]
    test_path = config["path"]["test_path"]
    anns_file_path = config["path"]["anns_file_path"]
    save_dir = config["path"]["save_model_path"]
    model_path = config["path"]["save_model_path"] + model_name + "_" + str(ex_id) + ".pt"
    conf.update({"path" : {"best_model_path" : model_path}})

# seed 고정
    random_seed = 21
    random_set(random_seed)

    sorted_df = EDA_S(anns_file_path)

# Dataset
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2()
        ])

    val_transform = A.Compose([
        ToTensorV2()
        ])

    test_transform = A.Compose([
        ToTensorV2()
        ])

    # train dataset
    train_dataset = CustomDataLoader(data_dir=train_path, data_path=dataset_path, sorted_df= sorted_df,mode='train', transform=train_transform)

    # validation dataset
    val_dataset = CustomDataLoader(data_dir=val_path, data_path=dataset_path, sorted_df= sorted_df,mode='val', transform=val_transform)

    # test dataset
    test_dataset = CustomDataLoader(data_dir=test_path, data_path=dataset_path, sorted_df= sorted_df,mode='test', transform=test_transform)


    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=4,
                                         collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          num_workers=4,
                                          collate_fn=collate_fn)

    # model 설정
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    # output class를 data set에 맞도록 수정
    model.classifier[4] = nn.Conv2d(256, 11, kernel_size=1)

    # 구현된 model에 임의의 input을 넣어 output이 잘 나오는지 test
    x = torch.randn([2, 3, 512, 512])
    print(f"input shape : {x.shape}")
    out = model(x)['out']
    #out = model(x).to(device)
    print(f"output shape : {out.size()}")

    # Loss function 정의
    criterion = nn.CrossEntropyLoss()

    # Optimizer 정의
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)

    #train
    train(num_epochs, model, train_loader, val_loader, criterion, optimizer, model_path, val_every, device)

    #sys.exit() -> wandb부분부터 다시 해결
    #make submission file 

    # best model 저장된 경로 : model_path

    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    model = model.to(device)

    infer(model, test_loader, device)
