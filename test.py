# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
import cv2
# This is for the progress bar.
from tqdm.auto import tqdm
import random
import torch.nn.functional as F
from dataset import SpineDataset_Test
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#from model import unet_model , 
from models.U2net import U2NET
from models.Unet import unet_model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--checkpoint", help = "the checkpoint")
parser.add_argument("-d", "--directory", help = "the directory to store result")
parser.add_argument("-c", "--channel", type = int, help = "the unet channel for different dataset")
args = parser.parse_args()


cls_color = {
    0:  [255, 255, 255],
    1:  [0 , 0 , 255],
    2:  [0, 153, 0],
    3:  [57, 217, 249],
    4:  [255, 128, 0],
    5:  [204, 153, 255],
    6: [255, 255, 0],
}

if __name__ == "__main__":
    if not os.path.exists(f"./result/{args.directory}"):
        os.makedirs(f"./result/{args.directory}")
    t1 = A.Compose([
        A.augmentations.transforms.CLAHE(),
        A.Resize(1024,512),
        ToTensorV2()
    ])
    data = []
    with open("testing.txt")as f:
        d  = f.read().splitlines()
    for i in d:
        tmp = "./testing_dataset/" + i
        data.append(tmp)
    test_dataset = SpineDataset_Test(data,t1)
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    model = unet_model(args.channel).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    with torch.no_grad():
        for x,path_name in tqdm(test_batch):
            write_name = path_name[0].split("/")[-1]
            softmax = nn.Softmax(dim=1)
            x = x.float().to(DEVICE)
            preds = torch.argmax(softmax(model(x)),axis=1).to('cpu')
            uni_data = np.unique(preds)
            img1 = np.transpose(np.array(x[0,:,:,:].to('cpu')),(1,2,0))
            preds1 = np.array(preds[0,:,:])
            
            img1 = cv2.cvtColor(img1 , cv2.COLOR_GRAY2BGR)
            img1 = np.array(img1).transpose(2,0,1)
            for i in range(len(preds1)):
                for j in range(len(preds1[0])):
                    if(preds1[i][j]==0):
                        continue
                    #print(preds1[i][j])
                    cmap = cls_color[preds1[i][j]]
                    for index in range(3):
                        img1[index][i][j] = cmap[index]
            img1 = img1.transpose(1,2,0)
            print(f"./{args.directory}/{write_name}")
            cv2.imwrite(f"./result/{args.directory}/{write_name}",img1)

