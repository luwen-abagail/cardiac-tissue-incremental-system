
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
#216*256，12
from dataset import load_data
#from train_PLOP import train_unet
#from lwf.train_lwf import train_unet
from train_mine import train_unet

from metrics import dice_single_class
import torch
import numpy as np
from Unetmodel import UNet
from look import newnii,newii

from loss import PCGJCL,DiceLoss
from torch.utils.data import Dataset, DataLoader
# 1. 加载预训练模型
def load_model(model_path):
    model = UNet(1,2)
    
    # 加载预训练权重
    model.load_state_dict(torch.load(model_path))
    
    # 设置评估模式
    model.eval()
    return model

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# 自定义数据集类
class AlbumentationsSegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].squeeze(0).numpy()  # Convert to numpy array
        mask = (self.masks[idx].cpu().numpy()).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image_aug = augmented['image']
            mask_aug = augmented['mask']
        else:
            image_aug = image
            mask_aug = mask

        # No need to manually convert to tensor if using ToTensorV2 in the transform
        image_tensor = augmented['image']  # This is already a tensor due to ToTensorV2
        mask_tensor = augmented['mask'].long()  # Ensure mask is long tensor for segmentation

        return image_tensor, mask_tensor

# Example data (replace with your actual data loading logic)
# Define transforms
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    
    # A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0, always_apply=True),  # Do not normalize masks
    A.pytorch.transforms.ToTensorV2()  # Convert to tensor (note: this is from albumentations.pytorch)
], 
additional_targets={'mask': 'mask'},  # Specify that we also want to transform the mask
bbox_params=None,  # Not needed for segmentation
keypoint_params=None  # Not needed for segmentation
)

# 使用示例
if __name__ == "__main__":
    # 参数设置
    model_path = "./model/unetplop_model1.pth"  # 替换为你的模型路径

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model =  load_model(model_path).to(device)  
    # 预处理数据
    dataset_dir_5 = "./b"
    image_data, label_data = load_data(dataset_dir_5, 256, 256)
    
    dataset1 = AlbumentationsSegmentationDataset(image_data, label_data,transform)
    dataloader1 = DataLoader(dataset1,batch_size=1, shuffle=False)
    num_classes=2
    miou=0
    j=0
    diceimages=torch.empty((0,256, 256))
    dicepredicted=torch.empty((0, 256, 256))
    i="100"
    diceimages=diceimages.to(device)
    dicepredicted=dicepredicted.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        accuracy=0
        progress = 0
        criterion1=0#伪标签要用的交叉熵
        for images, labels in dataloader1:
            progress+=1
            if ((labels ==0).sum().item()==65536):
                continue
            i = str(int(i) + 1) 
            # 前向传播
            # 将数据移动到设备上
            #images=torch.unsqueeze(images, 0)
            images = images.to(device)
            labels = labels.to(device)
            fea,outputs = model(images)
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