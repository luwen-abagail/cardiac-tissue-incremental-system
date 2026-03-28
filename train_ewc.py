from Unetmodel import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pseudo import pseudo
from look import newnii, newii
from loss import PCGJCL, DiceLoss
from torch.utils.data import Dataset, DataLoader
from metrics import dice_single_class
import torch.nn.functional as F  
import numpy as np
import time
from dataset import load_data
import copy
from torchvision import transforms
from PIL import Image
import os
from patch_utils import _get_patches, get_embeddings
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

class EWC:
    def __init__(self, model,lossfunction, dataloader, dataloader1,num_classes,device='cuda'):
        self.device = device
        self.param_old = {}
        self.fisher_matrix = {}
        
        # 存储旧参数
        for name, param in model.named_parameters():
            self.param_old[name] = param.clone().detach().to(device)
        for epoch in range(100):
            running_loss = 0.0
            progress = 0
            # 计算 Fisher 信息矩阵
            model.train()  # 确保模型在评估模式（关闭 Dropout/BatchNorm 扰动）
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                _,output = model(data)
                loss = lossfunction(output, target)
                loss.backward()
            
            # 累加梯度平方（Fisher 信息矩阵的近似）
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self.fisher_matrix:
                    self.fisher_matrix[name] = param.grad.clone().detach() ** 2
                else:
                    self.fisher_matrix[name] += param.grad.clone().detach() ** 2
        model.eval()
        diceimages = torch.empty((0, 256, 256)).to(device)
        dicepredicted = torch.empty((0, 256, 256)).to(device)
        alldice = 0
            
        with torch.no_grad():
            for images, labels in dataloader1:
                if (labels == 0).sum().item() == 65536:
                    continue
                images, labels = images.to(device), labels.to(device)
                _, outputs = model(images)
                predicted = torch.argmax(outputs, dim=1)
                diceimages = torch.cat((diceimages, labels), 0)
                dicepredicted = torch.cat((dicepredicted, predicted), 0)
                
            # 计算 Dice 系数
            for k in range(1, num_classes):
                dice = dice_single_class(diceimages, dicepredicted, k)
                print("dice", k, "=", dice)
                alldice += dice
        
        # 平均 Fisher 信息矩阵
        for name in self.fisher_matrix:
            self.fisher_matrix[name] /= len(dataloader)

    def penalty(self, model):
        loss = 0
        for name, param in model.named_parameters():
            if name in self.param_old and name in self.fisher_matrix:
                loss += torch.sum(self.fisher_matrix[name] * (param - self.param_old[name]) ** 2)
        return loss

def train_unet(teachermodel, image_dataset, label_dataset, jin_images, jin_label, num_epochs=10, learning_rate=0.001, pseud=None, distillation_weight=0.5, i="0", criterion_old=0, num_classes=1, easy_h=0, eh=0, method="mine"):  
    # 初始化背景掩码（保持原有逻辑）
    mask_new = torch.zeros_like(label_dataset).int()
    mask_new[label_dataset != 0] = 1
    
    # 分割数据集（保持原有逻辑）
    train_images = image_dataset
    train_labels = label_dataset
    test_images = jin_images
    jin_test = jin_label
    
    # 初始化模型和优化器（保持原有逻辑）
    studentmodel = UNet(1, num_classes)
    dataset = CustomDataset(train_images, train_labels)
    dataset1 = CustomDataset(test_images, jin_test)
    diceloss = DiceLoss(n_classes=num_classes)
    optimizer = optim.Adam(studentmodel.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    studentmodel.to(device)
    num_samples = 2
    
    if num_classes==3:
        dataset_dir_1 = "./class1data/train"
        dataset_dir_2 = "./class1data/test"
    else:
        dataset_dir_1 = "./class2data/train"
        dataset_dir_2 = "./class1data/test"
    image_data_1, label_data_1 = load_data(dataset_dir_1, 256, 256)
    datasetold = CustomDataset(image_data_1, label_data_1)
    dataloaderold = DataLoader(datasetold, batch_size=2, shuffle=False)
    image_data_2, label_data_2 = load_data(dataset_dir_2, 256, 256)
    datasetold2 = CustomDataset(image_data_2, label_data_2)
    dataloaderold2 = DataLoader(datasetold2, batch_size=1, shuffle=False)
    
    # ------------------------- 初始化 EWC -------------------------
    ewc = EWC(studentmodel, diceloss,dataloaderold,dataloaderold2,num_classes,device=device)
    
    # ------------------------- 训练循环 -------------------------
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    dataloader1 = DataLoader(dataset1, batch_size=1, shuffle=False)
    maxdice = 0
    best_model = None
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress = 0
        studentmodel.train()
        
        for images, labels in dataloader:
            if (labels == num_classes-1).sum().item() == 0:
                continue
            
            progress += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # 前向传播
            _, outputs = studentmodel(images)
            
            # 基础损失（如 Dice Loss）
            loss = diceloss(outputs, labels)
            
            # ------------------------- EWC 正则项 -------------------------
            if teachermodel is not None:  # 从第二个 epoch 开始施加 EWC
                ewc_loss = ewc.penalty(studentmodel)
                loss +=50 * ewc_loss  
            
            # 反向传播
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if progress > 0:
            print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss/progress))
            print(loss,ewc_loss)
        # ------------------------- 验证逻辑（保持原有） -------------------------
        if (epoch + 1) % 1 == 0:
            studentmodel.eval()
            diceimages = torch.empty((0, 256, 256)).to(device)
            dicepredicted = torch.empty((0, 256, 256)).to(device)
            alldice = 0
            
            with torch.no_grad():
                for images, labels in dataloader1:
                    if (labels == 0).sum().item() == 65536:
                        continue
                    images, labels = images.to(device), labels.to(device)
                    _, outputs = studentmodel(images)
                    predicted = torch.argmax(outputs, dim=1)
                    diceimages = torch.cat((diceimages, labels), 0)
                    dicepredicted = torch.cat((dicepredicted, predicted), 0)
                
                # 计算 Dice 系数
                for k in range(1, num_classes):
                    dice = dice_single_class(diceimages, dicepredicted, k)
                    print("dice", k, "=", dice)
                    alldice += dice
                
                if alldice >= maxdice or epoch + 1 == 5:
                    best_model = copy.deepcopy(studentmodel.state_dict())
                    maxdice = alldice
    
    # ------------------------- 返回最佳模型 -------------------------
    temp_model = copy.deepcopy(studentmodel)  # 复制模型结构
    if best_model is not None:
        temp_model.load_state_dict(best_model)    # 加载最佳状态
    temp_model.eval()     
    miou = 0
    j = 0
    diceimages = torch.empty((0, 256, 256))
    dicepredicted = torch.empty((0, 256, 256))
    diceimages = diceimages.to(device)
    dicepredicted = dicepredicted.to(device)
    feat_2t = []
    label_t = []
    
    with torch.no_grad():
        correct = 0
        total = 0
        accuracy = 0
        progress = 0
        criterion1 = 0  # 伪标签要用的交叉熵
        for images, labels in dataloader1:
            fea_t = []
            progress += 1
            if ((labels == 0).sum().item() == 65536):
                continue
            i = str(int(i) + 1) 
            images = images.to(device)
            labels = labels.to(device)
            fea, outputs = temp_model(images)
            predicted = torch.argmax(outputs, dim=1)
            diceimages = torch.cat((diceimages, labels), 0)
            dicepredicted = torch.cat((dicepredicted, predicted), 0)
            criterion = F.cross_entropy(outputs, labels.long(), reduction="none")
            criterion1 = criterion1 + criterion
            total += 65536
            correct += (predicted == labels).sum().item()
            newnii(predicted, labels, images, i)
            j += 1
        
        if total != 0:
            accuracy = 100 * correct / total
        
        print(i)
        print('Accuracy on validation set: %.2f %%' % accuracy)
        criterion_m = criterion1 / progress  # 中值熵
        for k in range(1, num_classes):
            dice = dice_single_class(diceimages, dicepredicted, k)
            print(k, 'dice on validation set: ', dice)
    
    return temp_model, i, criterion_m