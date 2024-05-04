import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import torch
from PIL import Image
#import matplotlib.pyplot as plt
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
#from tqdm import tqdm


class encoding_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(encoding_block,self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        return self.conv(x)    
    
class unet_model(nn.Module):
    def __init__(self,out_channels=9,features=[64, 128, 256, 512]):
        super(unet_model,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = encoding_block(1,features[0])
        self.conv2 = encoding_block(features[0],features[1])
        self.conv3 = encoding_block(features[1],features[2])
        self.conv4 = encoding_block(features[2],features[3])
        self.conv5 = encoding_block(features[3]*2,features[3])
        self.conv6 = encoding_block(features[3],features[2])
        self.conv7 = encoding_block(features[2],features[1])
        self.conv8 = encoding_block(features[1],features[0])        
        self.tconv1 = nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)        
        self.bottleneck = encoding_block(features[3],features[3]*2)
        self.final_layer = nn.Conv2d(features[0],out_channels,kernel_size=1)
        self.cnn1 =  nn.Conv2d(out_channels, 16,kernel_size=5,padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 =  nn.Conv2d(16, 32,kernel_size=5,padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.cnn3 =  nn.Conv2d(32, 64,kernel_size=5,padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.cnn4 =  nn.Conv2d(64, 128,kernel_size=5,padding=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.cnn5 =  nn.Conv2d(128, 256,kernel_size=5,padding=2)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(256*32*16,2)
    def forward(self,x):
        skip_connections = []
        # print(x.shape)
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)        
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        # print(x.shape)
        y =  self.cnn1(x)
        # print(y.shape)
        y = self.maxpool1(y)
        layer1 = y
        # print(y.shape)
        y = self.cnn2(y)
        # print(y.shape)
        y = self.maxpool2(y)
        # print(y.shape)
        layer2 = y
        y = self.cnn3(y)
        # print(y.shape)
        y = self.maxpool3(y)

        layer3 = y
        y = self.cnn4(y)
        # print(y.shape)
        y = self.maxpool4(y)
        layer4 = y
        y = self.cnn5(y)
        # print(y.shape)
        y = self.maxpool5(y)

        layer5 = y
        # print(y.shape)
        y = y.view(y.size(0),-1)
        y = self.fc1(y)
        # print(y.shape)
        # print(y )
        return x,y
    

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(64*128*64, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
    def forward(self, inputs):
        codes = self.encoder(inputs)
        return codes
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 64*128*64),
            nn.Sigmoid()
        )
    def forward(self, inputs):
        outputs = self.decoder(inputs)
        return outputs
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = Encoder()
        # Decoder
        self.decoder = Decoder()
    def forward(self, inputs):
            codes = self.encoder(inputs)
            decoded = self.decoder(codes)
            return codes, decoded