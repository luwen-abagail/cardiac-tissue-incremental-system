import torch  
import os
import numpy as np  
import nibabel as nib  

def newii(labels,i,file):
    labels=labels.cpu()
    data_numpy = labels.numpy()  
    # 确保数据类型是numpy.float32  
    data_numpy = data_numpy.astype(np.float32)  
    # 现在可以使用nibabel来保存NIfTI文件  
    affine = np.eye(4)  # 假设的affine矩阵，您需要根据实际情况来设置  
    nii_image = nib.Nifti1Image(data_numpy, affine)  
    #print(nii_image.shape)
    nib.save(nii_image,os.path.join(file, i))
def newnii(out,labels,images,i):
    file1="./newlabel"
    file2="./newimage"
    file3="./newout"
    newii(labels,i,file1)
    images=images.permute(0,2,3,1)
    newii(images,i,file2)
    newii(out,i,file3)