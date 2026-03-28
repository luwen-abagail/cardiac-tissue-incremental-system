#216*256，12
from dataset import load_data
#from train_PLOP import train_unet
#from lwf.train_lwf import train_unet
from train_mine import train_unet
import torch
import numpy as np
from Unetmodel import UNet


#from unet3 import UNetpip install numpy
torch.set_printoptions(threshold=np.inf)
#数据处理
#读取数据集
dataset_dir = "./ACDC3"
dataset_dir_1 = "./ACDC4"
dataset_dir_2 = "./ACDC5"#金标准
dataset_dir_3 = "./ACDC1"
dataset_dir_4 = "./ACDC2"
input_size_1 = 256
input_size_2 = 256
print("开始数据处理")
image_data, label_data = load_data(dataset_dir, input_size_1, input_size_2)

image_data_1, label_data_1 = load_data(dataset_dir_1, input_size_1, input_size_2)
image_data_2, label_data_2 = load_data(dataset_dir_2, input_size_1, input_size_2)
'''image_data_3, label_data_3 = load_data(dataset_dir_3, input_size_1, input_size_2)
image_data_4, label_data_4 = load_data(dataset_dir_4, input_size_1, input_size_2)'''
print("结束数据处理")
#acdc训练网络

image_dataset = image_data  # 替换为图像数据张量
label_dataset = label_data  # 替换为标签数据张

image_dataset_1 = image_data_1  # 替换为图像数据张量
label_dataset_1 = label_data_1  # 替换为标签数据张量
label_dataset_2 = label_data_2  # 替换为标签数据张量

'''image_dataset_3 = image_data_3  # 替换为图像数据张量
label_dataset_3 = label_data_3  # 替换为标签数据张量
label_dataset_4 = label_data_4  # 替换为标签数据张量'''
num_epochs  = 40
learning_rate = 0.0001
distillation_weight=20#pod长期记忆权重
method="mine"
i="0"
#train
easy_h=1
print("开始训练")
model1,i2,criterion_m1,_,_,_ = train_unet(teachermodel=None,image_dataset=image_dataset, label_dataset=label_dataset, num_epochs=num_epochs,learning_rate=learning_rate,jin_label=label_dataset,pseud=None,distillation_weight=distillation_weight,i=i,criterion_old=0,num_classes=2,easy_h=easy_h,eh=0,method=method)
print("结束训练")
print(i2,criterion_m1)
easy_h+=1
torch.save(model1.state_dict(), 'unetplop_model1.pth')
#model1 = torch.load('unetplop_model1.pth')
#model1 = UNet(3,3)	# 导入网络结构
#model1.load_state_dict(torch.load("unetplop_model1.pth")) # 导入网络的参数
num_epochs=num_epochs*3
print("持续学习1")
model2,i3,criterion_m2,hard_images,hard_labels,hard_jin = train_unet(teachermodel=model1,image_dataset=image_dataset_1, label_dataset=label_dataset_1,jin_label=label_dataset_2, num_epochs=num_epochs,learning_rate=learning_rate,pseud=1,distillation_weight=distillation_weight,i=i2,criterion_old=criterion_m1,num_classes=3,easy_h=easy_h,eh=3,method=method)
easy_h+=1
print("结束训练")
if hard_images.shape[0]>2:
    print("you1")
    model2,i3,criterion_m2,hard_images,hard_labels,hard_jin = train_unet(teachermodel=model2,image_dataset=hard_images, label_dataset=hard_labels, jin_label=hard_jin,num_epochs=num_epochs,learning_rate=learning_rate,pseud=1,distillation_weight=distillation_weight,i=i3,criterion_old=criterion_m2,num_classes=3,easy_h=easy_h,eh=3,method=method)

torch.save(model2.state_dict(), 'unetplop_model2.pth')

'''print("持续学习2")
easy_h=2
model3,i4,criterion_m3,hard_images,hard_labels,hard_jin = train_unet(teachermodel=model2,image_dataset=image_dataset_3, label_dataset=label_dataset_3,jin_label=label_dataset_4, num_epochs=num_epochs,learning_rate=learning_rate,pseud=1,distillation_weight=distillation_weight,i=i3,criterion_old=criterion_m2,num_classes=4,easy_h=easy_h,eh=3,method=method)
easy_h+=1
if hard_images.shape[0]>2:
    print("you2")
    model3,i4,criterion_m3,_,_,_ = train_unet(teachermodel=model3,image_dataset=hard_images, label_dataset=hard_labels, jin_label=hard_jin,num_epochs=num_epochs,learning_rate=learning_rate,pseud=1,distillation_weight=distillation_weight,i=i4,criterion_old=criterion_m3,num_classes=4,easy_h=easy_h,eh=3,method=method)

torch.save(model3.state_dict(), 'unetplop_model3.pth')'''


