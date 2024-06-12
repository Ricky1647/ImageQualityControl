# 這是拿來結合segmentation model 以及 image quality classify 的 train.py
# image quality classify 分開來

import numpy as np
import os
import sys
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

from dataset import SpineDataset_Pipeline,SpineDataset
from models.Unet_joint import ImageQualityClassifier
from models.Unet import unet_model
from models.anxialnet import axial50l
from models.U2net import U2NET
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
import torchvision.transforms as transforms





def test(model, moded_classify,test_loader, device=DEVICE):
    model.eval()
    moded_classify.eval()
    iou_score = 0.0
    acc = 0
    with torch.no_grad(): 
        outputs = np.zeros((len(test_loader), 1024, 512))
        masks = np.zeros((len(test_loader), 1024, 512))
        idx = 0
        total = 0
        for data, target ,weight, qualified in test_loader:
            total += len(qualified)
            softmax = nn.Softmax(dim=1)
            data, target = data.float().to(device), target.to(device)
            target = target.type(torch.long)
            x = model(data)
            y = x.detach()
            y = moded_classify(y)
            output = torch.argmax(softmax(x),axis=1)
            target = target.squeeze(1)
            output = output.cpu()
            target = target.cpu()
            outputs[idx, :, :] = output[0]
            masks[idx, :, :] = target[0]
            idx += 1
            _, test_pred = torch.max(y, 1)
            test_pred = test_pred.cpu().detach().numpy()
            qualified = qualified.detach().numpy()
            acc += (sum(test_pred==qualified))
        acc = acc/total
        iou_score = mean_iou_score(outputs, masks)

    print("\n[Testing] mIoU:{:.4f}".format(iou_score))
    print("\n[Testing] Acc :{:.4f}".format(acc))
    return iou_score,acc
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
    t1 = A.Compose([
        A.Resize(1024,512),
        ToTensorV2()
    ])
    t2 = A.Compose([
            A.Resize(1024,512),
            A.HorizontalFlip(p=1),
            ToTensorV2()
    ])
    t5 = A.Compose(
        [
            A.augmentations.transforms.CLAHE(),
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
    with open("./train_data.txt")as f:
        train_path = f.read().splitlines()
    with open("./unlabeled_data.txt")as f:
        train_un_path = f.read().splitlines()
    with open("./test_data.txt")as f:
        test_path = f.read().splitlines()
    #train_batch,test_batch = get_images(img_path,transform=t1,batch_size=4)
    training_data = SpineDataset_Pipeline(train_path,t1)
    training_data_flip = SpineDataset_Pipeline(train_path,t2)
    # training_data_center = SpineDataset_Pipeline(train_path,t3)
    training_data_a1 = SpineDataset_Pipeline(train_path,t5)
    training_data_a3 = SpineDataset_Pipeline(train_path,t7)

    training_un_data = SpineDataset(train_un_path,t1)
    training_un_data_flip = SpineDataset(train_un_path,t2)
    training_un_data_a1 = SpineDataset(train_un_path,t5)
    training_un_data_a3 = SpineDataset(train_un_path,t7)


    train_set = ConcatDataset([training_data,training_data_flip, training_data_a1, training_data_a3])
    train_un_set = ConcatDataset([training_un_data,training_un_data_flip, training_un_data_a1, training_un_data_a3])
    testing_data = SpineDataset_Pipeline(test_path,t1)

    train_batch = DataLoader(train_set, batch_size=8, shuffle=True,num_workers=8)
    train_un_batch = DataLoader(train_un_set, batch_size=8, shuffle=True,num_workers=8)

    test_batch = DataLoader(testing_data,batch_size=8,shuffle=True,num_workers=8)


    model = unet_model().to(DEVICE)
    moded_classify = ImageQualityClassifier().to(DEVICE) 
    #model = axial50l().to(DEVICE)
    # model  = U2NET().to(DEVICE)
    LEARNING_RATE = 1e-4
    num_epochs = 9000
    alpha = 0.95
    loss_fn = nn.CrossEntropyLoss()
    # model.load_state_dict(torch.load("checkpoint/unet_b_226/best_resnetlarge.ckpt"))
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer_classify = Adam(moded_classify.parameters(), lr=LEARNING_RATE)
    # optimizer.load_state_dict(torch.load("./checkpoint/unet/optimizer.ckpt"))
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0
    best_acc_classify = 0.0


    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        moded_classify.train()



        loop = tqdm(enumerate(train_un_batch),total=len(train_un_batch))
        for batch_idx, (data, targets,weighted_maps) in loop:
            batch_size = (data.shape[0])
            data = data.float().to(DEVICE)
            targets = targets.to(DEVICE)
            targets = targets.type(torch.long)
            weighted_maps =weighted_maps.to(DEVICE).type(torch.float64)

            with torch.cuda.amp.autocast():
                predictions = model(data)
                logp = F.log_softmax(predictions,dim=1)
                logp = logp.gather(1, targets.view(batch_size,1,1024,512))
                weighted_logp = (logp * weighted_maps).view(batch_size,-1)
                try2 = weighted_maps.view(batch_size,-1).sum(1)
                weighted_loss = weighted_logp.sum(1) / try2
                loss = -1*weighted_loss
                loss =loss.mean()
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix_str(f"unlabled loss_segment: {loss.item()}")
        
        
        loop = tqdm(enumerate(train_batch),total=len(train_batch))
        for batch_idx, (data, targets,weighted_maps, qualified) in loop:
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
                classify_input = predictions.detach()
                predict_qualified = moded_classify(classify_input)

                logp = F.log_softmax(predictions,dim=1)
                logp = logp.gather(1, targets.view(batch_size,1,1024,512))
                weighted_logp = (logp * weighted_maps).view(batch_size,-1)
                try2 = weighted_maps.view(batch_size,-1).sum(1)
                #try2[try2==0] = 1
                #print(try2)
                weighted_loss = weighted_logp.sum(1) / try2
                loss = -1*weighted_loss
                loss =loss.mean()
                
            ## classify backward
            
            loss_q = loss_fn(predict_qualified, qualified.to(DEVICE))
            # print(f"qualified loss: {loss_q}")
            loss_q.backward()
            optimizer_classify.step()
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            # update tqdm loop
            loop.set_postfix_str(f"loss_segment: {loss.item()} loss_qualified: {loss_q.item()}")
        acc, qualified_acc = test(model, moded_classify, test_batch) # Evaluate at the end of each epoch
        print("---segmentation---")
        print(best_acc)
        if acc > best_acc:
            best_acc = acc
            print("Saving best model... Best Acc is: {:.4f}".format(best_acc))
            torch.save(model.state_dict(), f"./checkpoint/{sys.argv[1]}/best_resnetlarge.ckpt")
            if acc>0.95:
                torch.save(model.state_dict(), f"./checkpoint/{sys.argv[1]}/{acc}_best_resnetlarge_{acc}.ckpt")
        torch.save(optimizer.state_dict(), f"./checkpoint/{sys.argv[1]}/optimizer.ckpt")
        
        print("---classify---")
        print(best_acc_classify)
        if qualified_acc > best_acc_classify:
            best_acc_classify = qualified_acc
            print("Saving best model... Best Acc is: {:.4f}".format(best_acc_classify))
            torch.save(moded_classify.state_dict(), f"./checkpoint/classify/{sys.argv[1]}/best_resnetlarge.ckpt")
            if qualified_acc>0.95:
                torch.save(moded_classify.state_dict(), f"./checkpoint/classify/{sys.argv[1]}/{acc}_best_resnetlarge_{acc}.ckpt")
        torch.save(optimizer_classify.state_dict(), f"./checkpoint/classify/{sys.argv[1]}/optimizer.ckpt")