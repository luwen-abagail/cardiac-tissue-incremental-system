
from Unetmodel import UNet
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import time



def _dice(label, pred, smooth=1):
    # 对矩阵计算
    intersection = torch.mul(label, pred).sum()
    return (2 * intersection + smooth) / (label.sum() + pred.sum() + smooth)

def dice_single_class(label, pred, target_class, smooth=1):
    batch_size = label.shape[0]
    dice_scores = np.zeros(batch_size)
    
    for i in range(batch_size):  # 每个batch
        # 将标签和预测转换为二值形式，仅针对目标类
        t = torch.where(label[i, :, :] == target_class, 1, 0)
        p = torch.where(pred[i, :, :] == target_class, 1, 0)
        
        # 计算该类的Dice系数
        dice_scores[i] = _dice(t, p, smooth)
    
    # 返回平均Dice系数（如果需要单个值而不是每个batch的值）
    return np.mean(dice_scores)  # 或者返回 dice_scores 如果需要每个样本的Dice
def meandice(pred, label):
    sumdice = 0
    smooth = 1e-6

    for i in range(1, 5):
        pred_bin = (pred==i)*1
        label_bin = (label==i)*1

        pred_bin = pred_bin.contiguous().view(pred_bin.shape[0], -1)
        label_bin = label_bin.contiguous().view(label_bin.shape[0], -1)

        intersection = (pred_bin * label_bin).sum()
        dice = (2. * intersection + smooth) / (pred_bin.sum() + label_bin.sum() + smooth)
        sumdice += dice

    return sumdice/4
