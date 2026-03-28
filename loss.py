import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np  
import random



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





def contrastive_loss(x, labels, temperature=0.07):
    """
    改进的对比损失函数，使用NT-Xent损失
    """
    # 归一化特征
    x_norm = F.normalize(x, p=2, dim=1)
    # 计算相似度矩阵
    sim_matrix = torch.mm(x_norm, x_norm.T) / temperature
    
    # 创建标签邻接矩阵
    labels = labels.contiguous().view(-1, 1)
    mask = labels.T == labels  # 正样本对掩码
    
    # 排除自相似性
    mask.fill_diagonal_(False)
    
    # 提取正样本和负样本相似度
    pos = sim_matrix[mask].view(sim_matrix.size(0), -1)
    neg = sim_matrix[~mask].view(sim_matrix.size(0), -1)
    
    # 计算logits和损失
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(x.device)
    
    return F.cross_entropy(logits, labels)
def PCGJCL(patch_list,patch_list_old, tau, patchnum):
    """
    改进的PCGJCL函数，增加批次处理和更合理的损失计算
    """
    total_loss = 0.0
    N = len(patch_list)  # 类别数
    
    for i in range(N):
        class_i = patch_list[i]
        for k in range(2,4):  # 假设4个尺度
            if len(class_i) < patchnum or len(class_i[0])==0:
                continue
            feats_i = torch.stack([patch[k] for patch in class_i[:patchnum]], dim=0)

            feats_i = feats_i.view(feats_i.size(0)*feats_i.size(1), -1)
            for j in range(N):
                if j == i :
                    continue
                    
                class_j = patch_list[j]
                if len(class_j) < patchnum or len(class_j[0])==0:
                    continue
                feats_j = torch.stack([patch[k] for patch in class_j[:patchnum]], dim=0)
                feats_j = feats_j.view(feats_j.size(0)*feats_j.size(1), -1)
                
                # 合并特征并计算损失
                x = torch.cat([feats_i, feats_j], dim=0)
                labels_i = torch.full((feats_i.size(0),), i)
                labels_j = torch.full((feats_j.size(0),), j)
                labels = torch.cat([labels_i, labels_j], dim=0)
                total_loss += contrastive_loss(x, labels, tau)
    
    # 更合理的归一化方式
    return total_loss / max(1, (N-1)*2*N)  # 每个类别与其他N-1个类别比较，每个比较4个尺度
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        # 使用 F.one_hot 替代 _one_hot_encoder
        target_one_hot = F.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        if weight is None:
            weight = [1] * self.n_classes
        
        assert inputs.size() == target_one_hot.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range( self.n_classes):
            dice = self._dice_loss(inputs[:, i], target_one_hot[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalPOD(nn.Module):
    def __init__(self, scales=[1, 1/2, 1/4]):
        super(LocalPOD, self).__init__()
        self.scales = scales

    def forward(self, x_old, x_current):
        # x_old: 旧模型的特征图 [B, C, H, W]
        # x_current: 当前模型的特征图 [B, C, H, W]
        loss = 0
        for i in range(4):
            for scale in self.scales:
                # 计算当前尺度的POD嵌入
                pod_old = self._compute_pod(x_old[i], scale)
                pod_current = self._compute_pod(x_current[i], scale)
                # 计算L2损失
                loss += F.mse_loss(pod_old, pod_current)
        return loss

    def _compute_pod(self, x, scale):
        # 缩放特征图
        h, w = x.shape[2], x.shape[3]
        new_h, new_w = int(h * scale), int(w * scale)
        x_scaled = F.interpolate(x, size=(new_h, new_w), mode='bilinear')
        # 计算宽度和高度合并切片
        pool_h = F.avg_pool2d(x_scaled, kernel_size=(new_h, 1))
        pool_w = F.avg_pool2d(x_scaled, kernel_size=(1, new_w))
        # 拼接POD嵌入
        pod = torch.cat([pool_h.flatten(2), pool_w.flatten(2)], dim=2)
        return pod
def icarl_loss(logits, old_logits=None, temperature=2.0):
    # 新数据分类损失
    total_loss=0
    for i in range(4):
        
        soft_targets = nn.functional.softmax(old_logits[i] / temperature, dim=-1)
        log_probs = nn.functional.log_softmax(logits[i] / temperature, dim=-1)
        kd_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)
        print(log_probs,soft_targets,kd_loss)
        total_loss += kd_loss
    
    
    return total_loss