import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from look import newii

from skimage.transform import resize, rotate
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

def pseudo(model,image_dataset, label_dataset,logits_old):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建自定义数据集实例
    dataset = CustomDataset(image_dataset, label_dataset)

    # 创建数据加载器
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    i="0"
    # 创建一个空张量，用于存储所有较小的张量,和mask
    all_tensors = torch.empty((0, 256, 256))
    all_tensors=all_tensors.to(device)
    #阈值
    #k=torch.mean(logits_old)
    #print(k)
    k=logits_old
    #torch.median(logits_old)
    # 在验证集上评估模型
    model.eval()
    with torch.no_grad():
        num=0
        for images, labels in dataloader:
            # 前向传播
            # 将数据移动到设备上
            
            #images=torch.unsqueeze(images, 0)
            images = images.to(device)
            labels = labels.to(device)

            fea,outputs = model(images)
            pseudo_labels = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)
            max_probs= probs.max(dim=1)[0]
            #pseudo_labels[max_probs< k] = 0
            pseudo_labels = torch.where(labels != 0, labels, pseudo_labels)
            newii(pseudo_labels,i,"./d")
            i = str(int(i) + 1) 
            all_tensors = torch.cat([all_tensors, pseudo_labels], dim=0)
        all_tensors=all_tensors.long()
            


    return all_tensors
