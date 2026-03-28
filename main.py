#216*256，12
from dataset import load_data
#from train_PLOP import train_unet
#from lwf.train_lwf import train_unet
#from train_ewc import train_unet
#from train_mine import train_unet
from train_icarl import train_unet
#from train_mbi import train_unet
import torch
import numpy as np
from Unetmodel import UNet

#from unet3 import UNetpip install numpy
torch.set_printoptions(threshold=np.inf)
#数据处理
#读取数据集
dataset_dir = "./class1data/train"
dataset_dir0 = "./class1data/test"
dataset_dir_1 = "./class2data/train"
dataset_dir_2 = "./class2data/test"#金标准
dataset_dir_3 = "./class3data/train"
dataset_dir_4 = "./class3data/test"
input_size_1 = 256
input_size_2 = 256
print("开始数据处理")
'''#image_data, label_data = load_data(dataset_dir, input_size_1, input_size_2)
#image_data0, label_data0 = load_data(dataset_dir0, input_size_1, input_size_2)'''
image_data_1, label_data_1 = load_data(dataset_dir_1, input_size_1, input_size_2)
image_data_2, label_data_2 = load_data(dataset_dir_2, input_size_1, input_size_2)
image_data_3, label_data_3 = load_data(dataset_dir_3, input_size_1, input_size_2)
image_data_4, label_data_4 = load_data(dataset_dir_4, input_size_1, input_size_2)
print("结束数据处理")
#acdc训练网络

'''image_dataset = image_data  # 替换为图像数据张量
label_dataset = label_data  # 替换为标签数据张
image_dataset0 = image_data0  # 替换为图像数据张量
label_dataset0 = label_data0  # 替换为标签数据张'''

image_dataset_1 = image_data_1  # 替换为图像数据张量
label_dataset_1 = label_data_1  # 替换为标签数据张量
image_dataset_2 = image_data_2
label_dataset_2 = label_data_2  # 替换为标签数据张量


image_dataset_3 = image_data_3  # 替换为图像数据张量
label_dataset_3 = label_data_3  # 替换为标签数据张量
image_dataset_4 = image_data_4
label_dataset_4 = label_data_4  # 替换为标签数据张量
num_epochs  = 10
learning_rate = 0.001
distillation_weight=20#pod长期记忆权重
method="mine"
i="0"
#train
easy_h=1
print("开始训练")
#modelclass1,i2,criterion_m1= train_unet(teachermodel=None,image_dataset=image_dataset, label_dataset=label_dataset, num_epochs=num_epochs,learning_rate=learning_rate,jin_images=image_dataset0,jin_label=label_dataset0,pseud=None,distillation_weight=distillation_weight,i=i,criterion_old=0,num_classes=2,easy_h=easy_h,eh=0,method=method)
print("结束训练")
#print(i2,criterion_m1)
i2="80"
criterion_m1=0.3148
easy_h+=1
#torch.save(modelclass1.state_dict(), 'unetplop_model1-all.pth')
model_path = "./unetplop_model1-all.pth" 
# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(model_path):
    model = UNet(1,2)
    
    # 加载预训练权重
    model.load_state_dict(torch.load(model_path))
    
    # 设置评估模式
    model.eval()
    return model
model1 =  load_model(model_path).to(device)
print("持续学习1")
model2,i3,criterion_m2= train_unet(teachermodel=model1,image_dataset=image_dataset_1, label_dataset=label_dataset_1,jin_images=image_dataset_2,jin_label=label_dataset_2, num_epochs=num_epochs,learning_rate=learning_rate,pseud=1,distillation_weight=distillation_weight,i=i2,criterion_old=criterion_m1,num_classes=3,easy_h=easy_h,eh=3,method="ewc")
easy_h+=1
print("结束训练")

torch.save(model2.state_dict(), 'unetplop_model2-all.pth')
# 加载模型

print("持续学习2")
easy_h=2
model3,i4,criterion_m3= train_unet(teachermodel=model2,image_dataset=image_dataset_3, label_dataset=label_dataset_3,jin_images=image_dataset_4,jin_label=label_dataset_4, num_epochs=num_epochs,learning_rate=learning_rate,pseud=1,distillation_weight=distillation_weight,i=i3,criterion_old=criterion_m2,num_classes=4,easy_h=easy_h,eh=3,method="ewc")

torch.save(model3.state_dict(), 'unetplop_model3.pth')


