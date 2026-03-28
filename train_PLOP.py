from Unetmodel import UNet
#from unet3 import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pseudo import pseudo
from look import newnii,newii
from loss import PCGJCL,DiceLoss,LocalPOD
from torch.utils.data import Dataset, DataLoader
from metrics import dice_single_class
import torch.nn.functional as F  
import numpy as np
import time
import copy
from torchvision import transforms
from PIL import Image
import os
from patch_utils import _get_patches,get_embeddings

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
def train_unet(teachermodel,image_dataset, label_dataset,jin_images,jin_label, num_epochs=10, learning_rate=0.001,pseud=None,distillation_weight=0.5,i="0",criterion_old=0,num_classes=1,easy_h=0,eh=0,method="mine"):  
    #label_dataset[label_dataset!=0]=1
    mask_new= torch.zeros_like(label_dataset).int()
    mask_new[label_dataset!=0]=1
    # 备训练数据
    if teachermodel !=None:
        label_dataset=pseudo(teachermodel,image_dataset, label_dataset,criterion_old)
                
    #分割数据集
    print(label_dataset.shape, image_dataset.shape)
    # 将数据集分割为训练集和测试集
    train_images=image_dataset
    train_labels=label_dataset
    test_images=jin_images
    jin_test=jin_label
    #train_images, train_labels = mosaic_data_augmentation(train_images, train_labels)
    # 创建模型实例
    studentmodel = UNet(1,num_classes)
    dataset = CustomDataset(train_images, train_labels)
    dataset1 = CustomDataset(test_images, jin_test)
    diceloss=DiceLoss(n_classes=num_classes)
    #loss=dice_loss
    # 定义损失函数和优化器
    optimizer = optim.Adam(studentmodel.parameters(), lr=learning_rate)
    #ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    # 训练循环
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    studentmodel.to(device)
    
    #loss_on =ExcludedKnowledgeDistillationLoss(reduction='mean', alpha=0.5)
    # 创建数据加载器
    #loss3=nn.MSELoss()
    #losss=dice_loss()
    batch_size = 2
    #harddata
    
    pseloss=LocalPOD(scales=[1, 1/2, 1/4])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader1 = DataLoader(dataset1, batch_size=1, shuffle=False)
    start_time = time.time()
    patch = []
    patchnum = 64
    patch_size = 16
    maxdice=0
    t=1
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress = 0
        # 训练数据迭代
        j=0
        studentmodel.train()

        for images, labels in  dataloader:
            
            if ((labels ==num_classes-1).sum().item()==0):
                #print("out1")
                continue
            # 更新进度  
            progress += 1  
            # 将数据移动到设备上
            
            #images=torch.unsqueeze(images, 0)
            images = images.to(device)
            labels = labels.to(device)
            #print(images.shape, labels.shape)
            # 清除梯度
            optimizer.zero_grad()
            
            if teachermodel == None:
                
                # 前向传播
                fea,out = studentmodel(images)
                # 计算损失
                loss = diceloss(out, labels)
                
            else:
                #calculate supervied lose
                fea,out = studentmodel(images)
                feaold,outold = teachermodel(images)
                supervised_loss = diceloss(out,labels)
                pse_loss=pseloss(fea,feaold)
                j+=1
    	        #total lossl  supervised_loss+0.95*supervised_loss+
                loss =0.9*supervised_loss+0.1*pse_loss 
                #print(PCGJCL_loss)
                #print(loss)
            
            #print(loss)   
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            #ExpLR.step()
            running_loss += loss.item()
        # 打印训练过程中的损失

        if progress>0:
            print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss/progress))
        
       
        if (epoch + 1) % 1== 0 :
            #print(supervised_loss,PCGJCL_loss)
            studentmodel.eval()  # 切换到评估模式
            j=0
            diceimages=torch.empty((0,256, 256))
            dicepredicted=torch.empty((0, 256, 256))
            alldice=0
            diceimages=diceimages.to(device)
            dicepredicted=dicepredicted.to(device)
            with torch.no_grad():  # 禁用梯度计算
                correct = 0
                total = 0
                accuracy=0
                progress = 0
                criterion1=0#伪标签要用的交叉熵
                for images, labels in dataloader1:
                    progress+=1
                    if ((labels ==0).sum().item()==65536):
                        continue
                    #i = str(int(i) + 1) 
                    # 前向传播
                    # 将数据移动到设备上
                    #images=torch.unsqueeze(images, 0)
                    images = images.to(device)
                    labels = labels.to(device)
                    fea,outputs = studentmodel(images)
                    predicted=torch.argmax(outputs, dim=1)
                    diceimages=torch.cat((diceimages,labels),0)
                    dicepredicted=torch.cat((dicepredicted,predicted),0)
                    # 计算准确率
                    total += 65536
                    correct += (predicted == labels).sum().item()
                   #newnii(predicted,labels,images,i)
            
                    j+=1
                if total!=0:
                    accuracy = 100 * correct / total
                print(i)
                print('Accuracy on validation set: %.2f %%' % accuracy)
                for k in range(1,num_classes):
                    dice=dice_single_class(diceimages,dicepredicted,k)
                    alldice+=dice
                    print(k,'dice on validation set: ', dice)
            if alldice>=maxdice or epoch+1==5:
                best_model = copy.deepcopy(studentmodel.state_dict())
                print("copy",epoch+1)
                maxdice=alldice
    
        
    end_time = time.time()  
    # 计算并打印训练时长  
    training_time = end_time - start_time  
    print(f"训练时长: {training_time:.2f} 秒")
    # 训练数据迭代  
    # 在验证集上评估模型
    temp_model = copy.deepcopy(studentmodel)  # 复制模型结构
    if best_model is not None:
        
        temp_model.load_state_dict(best_model)    # 加载最佳状态
    temp_model.eval()     
    miou=0
    j=0
    diceimages=torch.empty((0,256, 256))
    dicepredicted=torch.empty((0, 256, 256))
    
    diceimages=diceimages.to(device)
    dicepredicted=dicepredicted.to(device)

    with torch.no_grad():
        correct = 0
        total = 0
        accuracy=0
        progress = 0
        criterion1=0#伪标签要用的交叉熵
        for images, labels in dataloader1:
            fea_t=[]
            progress+=1
            if ((labels ==0).sum().item()==65536):
                continue
            i = str(int(i) + 1) 
            # 前向传播
            # 将数据移动到设备上
            #images=torch.unsqueeze(images, 0)
            images = images.to(device)
            labels = labels.to(device)
            fea,outputs = temp_model(images)
            predicted=torch.argmax(outputs, dim=1)
            diceimages=torch.cat((diceimages,labels),0)
            dicepredicted=torch.cat((dicepredicted,predicted),0)
            criterion =  F.cross_entropy(outputs, labels.long(), reduction="none")
            criterion1=criterion1+criterion
            # 计算准确率
            # 计算张量中1的数量   
            #print(labels)
            #print(labels)
            #print(predicted)
            total += 65536
            correct += (predicted == labels).sum().item()
            newnii(predicted,labels,images,i)
            
            j+=1
        if total!=0:
            accuracy = 100 * correct / total
        
        print(i)
        print('Accuracy on validation set: %.2f %%' % accuracy)
        criterion_m=criterion1/progress#中值熵
        for k in range(1,num_classes):
            dice=dice_single_class(diceimages,dicepredicted,k)
            print(k,'dice on validation set: ', dice)
    return temp_model,i,criterion_m