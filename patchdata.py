import os
import nibabel as nib
import numpy as np
import random

def random_crop_nii_files(input_folder, output_folder, crop_size):
    """
    批量随机位置裁剪NII文件
    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        crop_size: 裁剪尺寸 (depth, height, width)
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if not filename.endswith('.nii'):
            continue  # 跳过非NII文件

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # 加载NII文件
            img = nib.load(input_path)
            data = img.get_fdata()
            affine = img.affine  # 保持空间变换矩阵

            # 获取原始尺寸
            original_shape = data.shape
            print(f"原始尺寸: {original_shape}")

            # 计算每个维度的最大起始位置
            max_depth_start = original_shape[0] - crop_size[0]
            max_height_start = original_shape[1] - crop_size[1]

            # 生成随机起始位置（确保不越界）
            depth_start = random.randint(60, max_depth_start-60)
            height_start = random.randint(60, max_height_start-60)

            # 执行裁剪
            cropped_data = data[
                depth_start : depth_start + crop_size[0],
                height_start : height_start + crop_size[1],
            ]

            # 创建新NII文件
            new_img = nib.Nifti1Image(cropped_data, affine)
            nib.save(new_img, output_path)
            print(f"成功随机裁剪并保存: {filename}")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")


# 使用示例
input_folder = "./data1/patient052_ES"  # 替换为你的输入文件夹路径
output_folder = "./data2"  # 替换为你的输出文件夹路径
crop_size = (32, 32)  # 设置裁剪尺寸

random_crop_nii_files(input_folder, output_folder, crop_size)