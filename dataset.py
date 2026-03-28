import torch
import nibabel as nib
import numpy as np
from skimage.transform import rotate, resize
import random
import torch.nn.functional as F
torch.set_printoptions(threshold=np.inf)
import os

import torchvision.transforms as T
from skimage.transform import resize, rotate



def load_data(dataset_dir, input_size_1, input_size_2):
    folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    image_paths = []
    label_paths = []
    # 遍历每个文件夹
    for folder in folders:
        # 构建原始文件夹中图像和标签文件夹的路径
        image_folder = os.path.join(dataset_dir, folder)
        # 获取原始文件夹中的所有文件列表
        files = os.listdir(image_folder)
        
        for file1 in files:
            # 检查文件名是否以 "image" 开头并以 ".nii" 结尾
            if file1.startswith("image") and file1.endswith(".nii"):
                image_paths.append(os.path.join(image_folder, file1))
            
            # 获取原始文件夹中的所有标签文件列表
            if file1.startswith("label") and file1.endswith(".nii"):
                label_paths.append(os.path.join(image_folder, file1))
    
    image_data = []
    label_data = []

    # 定义旋转角度的范围（例如，-30到30度）
    angle_range = [-10, 10]

    # 遍历每个图像和标签路径
    for image_path, label_path in zip(image_paths, label_paths):
    # 加载NIfTI文件
        nii_image = nib.load(image_path)
        nii_label = nib.load(label_path)
        data_image = nii_image.get_fdata()
        data_label = nii_label.get_fdata()
 
        # 调整大小
        data_image = resize(data_image, (input_size_1, input_size_2, 1), preserve_range=True)
        data_label = resize(data_label, (input_size_1, input_size_2), preserve_range=True)
 
        # 归一化到 [0, 1]
        '''data_image = (data_image - np.min(data_image)) / (np.max(data_image) - np.min(data_image) + 1e-8)
 
        # 标准化
        mean = np.mean(data_image)
        std = np.std(data_image)
        data_image = (data_image - mean) / (std + 1e-8)
         # 随机旋转（保持尺寸不变，使用零填充）
        angle = random.uniform(angle_range[0], angle_range[1])
        data_image = rotate(data_image, angle, resize=False, mode='constant', cval=0)
        data_label = rotate(data_label, angle, resize=False, mode='constant', cval=0)'''
 
        # 将裁剪后的数据添加到列表中
        image_data.append(data_image)
        label_data.append(data_label)

    # 将图像和标签数据转换为NumPy数组
    image_data = np.array(image_data)
    label_data = np.array(label_data)

    # 将NumPy数组转换为PyTorch张量
    tensor_image = torch.tensor(image_data, dtype=torch.float32)
    tensor_image=tensor_image.permute(0,3,1,2)
    tensor_label = torch.tensor(label_data, dtype=torch.long)  # 或torch.long，取决于标签的类型

    return tensor_image, tensor_label
