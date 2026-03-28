from Unetmodel import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pseudo import pseudo
from look import newnii, newii
from loss import PCGJCL, DiceLoss, icarl_loss
from torch.utils.data import Dataset, DataLoader
from metrics import dice_single_class
import torch.nn.functional as F  

from dataset import load_data
import numpy as np
import time
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

# 添加旧知识存储类

def train_unet(teachermodel, image_dataset, label_dataset, jin_images, jin_label, 
               num_epochs=10, learning_rate=0.001, pseud=None, distillation_weight=0.5,
               i="0", criterion_old=0, num_classes=1, easy_h=0, eh=0, method="mine", 
               old_knowledge=None):  
    # 初始化知识库
    
    mask_new = torch.zeros_like(label_dataset).int()
    mask_new[label_dataset != 0] = 1
         
    # 分割数据集
    print(label_dataset.shape, image_dataset.shape)
    train_images = image_dataset
    train_labels = label_dataset
    test_images = jin_images
    jin_test = jin_label
    
    # 创建模型实例
    studentmodel = UNet(1, num_classes)
    dataset = CustomDataset(train_images, train_labels)
    dataset1 = CustomDataset(test_images, jin_test)
    diceloss = DiceLoss(n_classes=num_classes)
    
    # 定义优化器
    optimizer = optim.Adam(studentmodel.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    studentmodel.to(device)
    
    # 创建数据加载器
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader1 = DataLoader(dataset1, batch_size=1, shuffle=False)
    
    start_time = time.time()
    maxdice = 0
    best_model = None
    if num_classes==3:
        dataset_dir_1 = "./class1data/train"
    else:
        dataset_dir_1 = "./classd2ata/train"
    image_data_1, label_data_1 = load_data(dataset_dir_1, 256, 256)
    
    datasetold = CustomDataset(image_data_1, label_data_1)
    
    dataloaderold = DataLoader(datasetold, batch_size=2, shuffle=False)
    # 获取旧知识（如果有）
    teachermodel.eval()
    with torch.no_grad():
        # 从训练数据中选取一些样本存储
        sample_images = []
        sample_features = []
        sample_labels = []
        
        # 只存储当前类别的样本（避免存储太多背景）
        for images, labels in dataloaderold:
            images = images.to(device)
            labels = labels.to(device)
            
            # 只选择包含当前类别的样本
            mask = (labels == num_classes-1)
            if mask.sum() > 0:
                fea,_= teachermodel(images)
                sample_images.append(images.cpu())
                sample_features.append(fea.cpu())
                sample_labels.append(labels.cpu())
        
        if len(sample_images) > 0:
            old_images = torch.cat(sample_images, dim=0)
            old_features = torch.cat(sample_features, dim=0)
            old_labels = torch.cat(sample_labels, dim=0)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress = 0
        studentmodel.train()

        # 训练当前任务数据
        for images, labels in dataloader:
            # 跳过没有当前类标签的样本
            if ((labels == num_classes-1).sum().item() == 0):
                continue
                
            progress += 1  
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 获取当前模型输出和特征
            fea, out = studentmodel(images)
            
            # 计算当前任务的损失
            supervised_loss = diceloss(out, labels)
            
            # 如果有教师模型（旧模型）和旧知识，计算蒸馏损失
            kd_loss = torch.tensor(0.0).to(device)
            print(old_images)
            if teachermodel is not None and old_images is not None and len(old_images) > 0:
                # 确保旧数据在正确的设备上
                old_images = old_images.to(device)
                old_labels = old_labels.to(device)
                print("1")
                # 计算旧数据的特征（使用当前模型）
                with torch.no_grad():
                    old_fea_teacher,_ = teachermodel(old_images)
                
                
                # 计算特征蒸馏损失
                temperature = 2.0
                kd_loss = icarl_loss(fea, old_fea_teacher.detach(), temperature)
                
                # 计算旧类别的分类损失（如果旧标签中有当前类别）
                if num_classes > 1:
                    old_out = studentmodel.classify(old_images)  # 假设模型有classify方法
                    old_pred = torch.argmax(old_out, dim=1)
                    # 只计算旧类别的损失
                    old_class_loss = F.cross_entropy(old_out, old_labels.long())
                    kd_loss += old_class_loss
            
            # 组合损失
            total_loss = supervised_loss + 10 * kd_loss
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        
        if progress > 0:
            print('Epoch [%d/%d], Loss: %.4f (Sup: %.4f, KD: %.4f)' % 
                  (epoch+1, num_epochs, running_loss/progress, 
                   supervised_loss.item(), kd_loss.item()))
        
        # 验证阶段
        if (epoch + 1) % 1 == 0:
            studentmodel.eval()
            diceimages = torch.empty((0, 256, 256)).to(device)
            dicepredicted = torch.empty((0, 256, 256)).to(device)
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in dataloader1:
                    if ((labels == 0).sum().item() == 65536):
                        continue
                    
                    images = images.to(device)
                    labels = labels.to(device)
                    fea, outputs = studentmodel(images)
                    predicted = torch.argmax(outputs, dim=1)
                    
                    diceimages = torch.cat((diceimages, labels), 0)
                    dicepredicted = torch.cat((dicepredicted, predicted), 0)
                    
                    total += 65536
                    correct += (predicted == labels).sum().item()
            
            if total != 0:
                accuracy = 100 * correct / total
                print('Accuracy on validation set: %.2f %%' % accuracy)
                
            alldice = 0
            for k in range(1, num_classes):
                dice = dice_single_class(diceimages, dicepredicted, k)
                alldice += dice
                print(k, 'dice on validation set: ', dice)
            
            # 保存最佳模型
            if alldice >= maxdice or epoch + 1 == 5:
                best_model = copy.deepcopy(studentmodel.state_dict())
                print("Saved best model at epoch", epoch + 1)
                maxdice = alldice
    
    # 训练结束后，存储当前任务的知识用于未来任务

    
    end_time = time.time()  
    training_time = end_time - start_time  
    print(f"训练时长: {training_time:.2f} 秒")
    
    # 加载最佳模型进行评估
    temp_model = copy.deepcopy(studentmodel)
    if best_model is not None:
        temp_model.load_state_dict(best_model)
    temp_model.eval()
    
    # 最终评估
    diceimages = torch.empty((0, 256, 256)).to(device)
    dicepredicted = torch.empty((0, 256, 256)).to(device)
    correct = 0
    total = 0
    criterion1 = 0
    
    with torch.no_grad():
        for images, labels in dataloader1:
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
            criterion1 += criterion.sum().item()
            
            total += 65536
            correct += (predicted == labels).sum().item()
            newnii(predicted, labels, images, i)
    
    if total != 0:
        accuracy = 100 * correct / total
        print('Final Accuracy on validation set: %.2f %%' % accuracy)
    
    criterion_m = criterion1 / total if total > 0 else 0
    for k in range(1, num_classes):
        dice = dice_single_class(diceimages, dicepredicted, k)
        print(k, 'Final dice on validation set: ', dice)
    
    return temp_model, i, criterion_m