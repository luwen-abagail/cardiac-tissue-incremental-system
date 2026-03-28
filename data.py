import os
import shutil
import nibabel as nib

# 原始文件夹路径
original_folder = "./ACDC"

# 新文件夹路径
new_folder = "./ACDC5"

# 获取 ACDC 文件夹下的所有文件夹列表
folders = [f for f in os.listdir(original_folder) if os.path.isdir(os.path.join(original_folder, f))]

# 遍历每个文件夹
for folder in folders:
    
    # 构建原始文件夹中图像和标签文件夹的路径
    image_folder = os.path.join(original_folder, folder)
    # 构建新文件夹中图像和处理后标签的路径
    new_image_folder = os.path.join(new_folder, folder)
    
    # 创建新文件夹
    os.makedirs(new_image_folder, exist_ok=True)
     # 遍历每个文件
    # 获取原始文件夹中的所有文件列表
    files = os.listdir(image_folder)
    i=0
    for file1 in files:
        # 检查文件名是否以 "image" 开头并以 ".nii" 结尾
        if file1.startswith("image") and file1.endswith(".nii"):
            shutil.copy(os.path.join(image_folder, file1), os.path.join(new_image_folder, file1))
        # 获取原始文件夹中的所有标签文件列表
        if file1.startswith("label") and file1.endswith(".nii"):
            # 对标签文件进行处理
             # 读取标签图像
            label_image = nib.load(os.path.join(image_folder, file1))
            label_data = label_image.get_fdata()
            # 指定要保留的物体标签值
            # 将非目标物体标签设置为背景（0）
            
            label_data[label_data == 1] = 0
            label_data[label_data == 3] = 1
            # 创建新的标签图像对象
            new_label_image = nib.Nifti1Image(label_data, label_image.affine, label_image.header)
            # 保存修改后的标签图像
            nib.save(new_label_image,os.path.join(new_image_folder, file1))
            i+1