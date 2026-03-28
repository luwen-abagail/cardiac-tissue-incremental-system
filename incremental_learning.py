import torch
import os
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 设置设备为CPU
DEVICE = torch.device('cpu')

def run_incremental_learning(user_id, task_id, model_type='existing', data_type='existing'):
    """
    执行增量学习
    
    Args:
        user_id: 用户ID
        task_id: 任务ID
        model_type: 'existing' - 使用已有模型, 'custom' - 使用自定义模型
        data_type: 'existing' - 使用已有数据, 'custom' - 使用自定义数据
    
    Returns:
        结果字典
    """
    
    try:
        # 获取数据路径
        if data_type == 'existing':
            data_path = 'data/sample_data'  # 你的已有数据路径
        else:
            data_path = f'uploads/data/user_{user_id}'
        
        # 获取模型路径
        if model_type == 'existing':
            model_path = 'data/models/pretrained_model.pth'  # 你的预训练模型路径
        else:
            model_path = f'uploads/models/user_{user_id}_model.pth'
        
        # 加载模型
        print(f"[CPU Mode] 加载模型从: {model_path}")
        model = load_model(model_path)
        model.to(DEVICE)
        model.eval()
        
        # 加载数据
        print(f"[CPU Mode] 加载数据从: {data_path}")
        train_loader, val_loader = load_data(data_path)
        
        # 执行增量学习
        print("[CPU Mode] 开始增量学习过程...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        num_epochs = 10
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # 训练阶段
            train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            
            # 验证阶段
            val_loss = validate_epoch(model, val_loader, criterion, DEVICE)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存最佳模型
                save_path = f'uploads/models/user_{user_id}_task_{task_id}_best.pth'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        
        # 生成结果
        result = {
            'status': 'success',
            'message': '增量学习完成',
            'best_val_loss': float(best_val_loss),
            'model_saved_path': save_path,
            'device': 'CPU'
        }
        
        return str(result)
    
    except Exception as e:
        print(f"[ERROR] 增量学习失败: {str(e)}")
        return str({'status': 'error', 'message': str(e)})


def load_model(model_path):
    """
    加载模型
    这里需要根据你实际的模型架构修改
    """
    from Unetmodel import UNet  # 从你的代码导入
    
    model = UNet(in_channels=1, out_channels=4)  # 根据你的模型调整参数
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    return model


def load_data(data_path):
    """
    加载数据
    返回训练和验证数据加载器
    """
    from dataset import CardiacDataset  # 从你的代码导入
    
    # 假设你已经有CardiacDataset类
    train_dataset = CardiacDataset(root_dir=data_path, split='train')
    val_dataset = CardiacDataset(root_dir=data_path, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """
    验证一个epoch
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validating')
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)
