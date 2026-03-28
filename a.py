import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
 
# 1. 定义一个简单的神经网络
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
 
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
 
# 2. 定义 EWC 类
class EWC:
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.params_old = {n: p.clone().detach() for n, p in self.params.items()}
        self.fisher_matrices = self._compute_fisher_matrices()
 
    # 2.1 计算 Fisher 信息矩阵
    def _compute_fisher_matrices(self):
        fisher_matrices = {}
        self.model.eval()
        for n, p in self.params.items():
            fisher_matrices[n] = torch.zeros_like(p)
        
        # 使用dataloader加载旧任务的数据
        dataloader = DataLoader(self.dataset, batch_size=128, shuffle=True)
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            # 计算交叉熵损失
            loss = nn.CrossEntropyLoss()(outputs, targets)
            # 计算梯度
            loss.backward()
            # 累加Fisher矩阵
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    fisher_matrices[n] += p.grad.data ** 2 / len(dataloader.dataset)
        
        return fisher_matrices
 
    # 2.2 EWC 损失函数
    def ewc_loss(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                # 计算损失，惩罚新任务中与旧任务差异大的权重
                loss += (self.fisher_matrices[n] * (p - self.params_old[n]) ** 2).sum()
        return loss
 
# 3. 模拟数据集和训练函数
class SimpleDataset(Dataset):
    def __init__(self, num_samples, input_size, num_classes):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes
        self.data = torch.randn(num_samples, input_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
 
    def __len__(self):
        return self.num_samples
 
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
 
def train_ewc(model, old_dataset, new_dataset, device, ewc_lambda=1.0, epochs=10, lr=0.001):
    # 1. 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
 
    # 2. 初始化 EWC 对象
    ewc = EWC(model, old_dataset, device)
 
    # 3. 定义数据加载器
    old_dataloader = DataLoader(old_dataset, batch_size=128, shuffle=True)
    new_dataloader = DataLoader(new_dataset, batch_size=128, shuffle=True)
 
    # 4. 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # 4.1 训练新任务
        for inputs, targets in new_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_ce = nn.CrossEntropyLoss()(outputs, targets)
            # 计算EWC损失
            loss_ewc = ewc.ewc_loss(model) * ewc_lambda
            # 计算总损失
            loss = loss_ce + loss_ewc
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(new_dataloader):.4f}')
 
    print('Training finished')
 
# 5. 主函数
if __name__ == '__main__':
    # 5.1 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    # 5.2 超参数设置
    input_size = 10
    hidden_size = 20
    num_classes = 5
    num_samples_old = 1000
    num_samples_new = 1000
 
    # 5.3 模拟数据集
    old_dataset = SimpleDataset(num_samples_old, input_size, num_classes)
    new_dataset = SimpleDataset(num_samples_new, input_size, num_classes)
 
    # 5.4 初始化模型
    model = Net(input_size, hidden_size, num_classes).to(device)
 
    # 5.5 训练 EWC 模型
    train_ewc(model, old_dataset, new_dataset, device)