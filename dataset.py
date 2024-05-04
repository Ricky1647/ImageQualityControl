from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import torchvision.transforms as transforms
import numpy as np 
import pandas as pd
import json


class SpineDataset(Dataset):
    def __init__(self,img_path,transform = None):
        self.transforms = transform
        self.img_path = img_path
    def __len__(self):
        return len(self.img_path)
    def __getitem__(self,index):
        img = np.array(Image.open(f"./dataset/{self.img_path[index]}/img.png"))
        mask = np.array(Image.open(f"./dataset/{self.img_path[index]}/label.png"))
        # with open("qualified.json") as f :
        #     qualified = json.load(f)
        if self.transforms is not None:
            aug = self.transforms(image=img,mask=mask)
            c_weights = np.zeros(16)
            for i in range(16):
                if (mask==i).sum()==0:
                    c_weights[i] = 0.0
                else:
                    c_weights[i] = 1.0/ ((mask==i).sum())
            c_weights /= c_weights.max()
            img = aug['image']
            mask = aug['mask']
            weight = np.copy(mask).astype(float)
            for i in range(16):
                weight[weight==i]=c_weights[i]
        # image = self.img_path[index].replace("_json",".png")
        #return img,mask,weight, int(qualified[image])
        return img,mask,weight
    
class SpineDataset_Test(Dataset):
    def __init__(self,img_path,transform = None):
        self.transforms = transform
        self.img_path = img_path
    def __len__(self):
        return len(self.img_path)
    def __getitem__(self,index):
        img = np.array(Image.open(f"{self.img_path[index]}"))
        # img = np.array(Image.open(f"./dataset/{self.img_path[index]}/img.png"))
        if self.transforms is not None:
            aug = self.transforms(image=img)
            img = aug['image']
        return img,self.img_path[index]
    

class SpineDataset_Denoise(Dataset):
    def __init__(self,img_path,transform = None):
        self.transforms = transform
        self.img_path = img_path
    def __len__(self):
        return len(self.img_path)
    def __getitem__(self,index):
        img = (Image.open(f"./dataset/{self.img_path[index]}_json/label.png"))
        denoise_img = np.asarray(np.loadtxt(f"./denoise_dataset/{self.img_path[index]}.txt"), dtype=np.uint8)
        # print(f"origin image label {np.array(img).max()}")
        # print(f"denoise image label {denoise_img.max()}")
        denoise_img = Image.fromarray(denoise_img)
        # img = np.array(Image.open(f"./dataset/{self.img_path[index]}/img.png"))
        img = self.transforms(img)
        denoise_img = self.transforms(denoise_img)
        return img,denoise_img

class SpineDataset_Denoise_Test(Dataset):
    def __init__(self,img_path,transform = None):
        self.transforms = transform
        self.img_path = img_path
    def __len__(self):
        return len(self.img_path)
    def __getitem__(self,index):
        img = (Image.open(f"./testing/{self.img_path[index]}.png"))
        #denoise_img = (Image.open(f"./denoise_test_dataset/{self.img_path[index]}.png"))
        denoise_img = np.asarray(np.loadtxt(f"./denoise_test_dataset/{self.img_path[index]}.txt"), dtype=np.uint8)
        #print(denoise_img.max())
        denoise_img = np.array(denoise_img)
        #print(f"www {denoise_img.max()}")
        denoise_img = Image.fromarray(denoise_img)

        # img = np.array(Image.open(f"./dataset/{self.img_path[index]}/img.png"))
        img = self.transforms(img)
        denoise_img = self.transforms(denoise_img)
        return img, denoise_img, f"{self.img_path[index]}.png"
    