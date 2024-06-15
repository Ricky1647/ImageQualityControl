# 這是原始

import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import torch
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import os
from dataset import SpineDataset
from models.Unet import unet_model
from models.Unet import AutoEncoder
from models.anxialnet import axial50l
from models.U2net import U2NET
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--train", help = "the input file for training")
parser.add_argument("-t", "--test", help = "the input  file for testing")
parser.add_argument("-d", "--directory", help = "the directory to store checkpoint")
parser.add_argument("-c", "--channel", type = int, help = "the unet channel for different dataset")
parser.add_argument("-n", "--gpu", type = int, help = "the gpu index")

args = parser.parse_args()
gpu_index = args.gpu
DEVICE = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
def test(model, test_loader, device=DEVICE):
    model.eval()
    iou_score = 0.0
    with torch.no_grad(): 
        outputs = np.zeros((len(test_loader), 1024, 512))
        masks = np.zeros((len(test_loader), 1024, 512))
        idx = 0
        for data, target ,weight in test_loader:
            softmax = nn.Softmax(dim=1)
            data, target = data.float().to(device), target.to(device)
            target = target.type(torch.long)
            x = model(data)
            output = torch.argmax(softmax(x),axis=1)
            target = target.squeeze(1)
            output = output.cpu()
            target = target.cpu()
            outputs[idx, :, :] = output[0]
            masks[idx, :, :] = target[0]
            idx += 1
        iou_score = mean_iou_score(outputs, masks)

    print("\n[Testing] mIoU:{:.4f}".format(iou_score))
    return iou_score
def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    unique_data = np.unique(labels)
    for i in unique_data:
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        # print(tp_fp, tp_fn, tp)
        if ((tp_fp + tp_fn - tp)==0):
            _divide = 1
        else:
            _divide = tp_fp + tp_fn - tp
        iou = tp / (_divide)
        mean_iou += iou / len(unique_data)
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

if __name__ == "__main__":
    if not os.path.exists(f"./checkpoint/{args.directory}"):
        os.makedirs(f"./checkpoint/{args.directory}")
    t1 = A.Compose([
        A.Resize(1024,512),
        ToTensorV2()
    ])
    t2 = A.Compose([
            A.Resize(1024,512),
            A.HorizontalFlip(p=1),
            ToTensorV2()
    ])
    t3 = A.Compose([
        A.CenterCrop(1024,512),
        ToTensorV2(),
    ])
    t4 = A.Compose([
        A.Resize(1024,512),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        ToTensorV2()
    ]
    )
    t5 = A.Compose(
        [
            A.augmentations.transforms.CLAHE(),
            A.Resize(1024,512),
            ToTensorV2()
        ]
    )
    t6 = A.Compose(
        [
            A.augmentations.transforms.InvertImg(),
            A.Resize(1024,512),
            ToTensorV2()
        ]
    )

    t7 = A.Compose(
        [
            A.augmentations.transforms.Equalize(),
            A.Resize(1024,512),
            ToTensorV2()
        ]
    )
    t8 = A.Compose(
        [
            A.augmentations.transforms.ColorJitter(),
            A.Resize(1024,512),
            ToTensorV2()
        ]
    )
    with open(args.train)as f:
        train_path = f.read().splitlines()
    with open(args.test)as f:
        test_path = f.read().splitlines()
    #train_batch,test_batch = get_images(img_path,transform=t1,batch_size=4)
    training_data = SpineDataset("./dataset",train_path,t1)
    training_data_flip = SpineDataset("./dataset",train_path,t2)
    # training_data_center = SpineDataset("./dataset",train_path,t3)
    training_data_rotate = SpineDataset("./dataset",train_path,t4)
    training_data_a1 = SpineDataset("./dataset",train_path,t5)
    training_data_a2 = SpineDataset("./dataset",train_path,t6)
    training_data_a3 = SpineDataset("./dataset",train_path,t7)
    training_data_a4 = SpineDataset("./dataset",train_path,t8)


    train_set = ConcatDataset([training_data,training_data_flip,training_data_rotate, training_data_a1, training_data_a2, training_data_a3, training_data_a4])

    testing_data = SpineDataset("./dataset",test_path,t1)

    train_batch = DataLoader(train_set, batch_size=10, shuffle=True,num_workers=32)
    test_batch = DataLoader(testing_data,batch_size=10,shuffle=True,num_workers=32)

    model = unet_model(out_channels = args.channel).to(DEVICE)
    #model = axial50l().to(DEVICE)
    # model  = U2NET().to(DEVICE)
    LEARNING_RATE = 1e-4
    num_epochs = 9000
    alpha = 0.90
    # model.load_state_dict(torch.load("checkpoint/unet_b_226/best_resnetlarge.ckpt"))
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer.load_state_dict(torch.load("./checkpoint/unet/optimizer.ckpt"))
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0
    


    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        loop = tqdm(enumerate(train_batch),total=len(train_batch))
        for batch_idx, (data, targets,weighted_maps) in loop:
            # print(f"reference {targets.shape}")
            #print(data.shape)
            # print(qualified)
            batch_size = (data.shape[0])

            data = data.float().to(DEVICE)
            targets = targets.to(DEVICE)
            targets = targets.type(torch.long)
            weighted_maps =weighted_maps.to(DEVICE).type(torch.float64)
            #print(weighted_maps)
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                logp = F.log_softmax(predictions,dim=1)
                logp = logp.gather(1, targets.view(batch_size,1,1024,512))
                weighted_logp = (logp * weighted_maps).view(batch_size,-1)
                try2 = weighted_maps.view(batch_size,-1).sum(1)
                #try2[try2==0] = 1
                #print(try2)
                weighted_loss = weighted_logp.sum(1) / try2
                loss = -1*weighted_loss
                loss =loss.mean()

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
        acc = test(model, test_batch) # Evaluate at the end of each epoch
        print(best_acc)
        if acc > best_acc:
            best_acc = acc
            print("Saving best model... Best Acc is: {:.4f}".format(best_acc))
            torch.save(model.state_dict(), f"./checkpoint/{args.directory}/best_resnetlarge.ckpt")
            if acc>0.95:
                torch.save(model.state_dict(), f"./checkpoint/{args.directory}/{acc}_best_resnetlarge_{acc}.ckpt")
        torch.save(optimizer.state_dict(), f"./checkpoint/{args.directory}/optimizer.ckpt")
