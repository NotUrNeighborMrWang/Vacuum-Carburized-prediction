
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils.process import Processor
from utils.models import *
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from tensorboardX import SummaryWriter

from sklearn.model_selection import train_test_split

import datetime
from PIL import Image
import json

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
   
# 定义数据集类
class MetallographicDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        # 处理图像
        image_path = sample['image']

        # 合并成分和热处理参数
        grade_and_process = torch.tensor([float(g) for g in sample['grade']] + [float(p) for p in sample['process']])

        # 只返回图像路径和合并后的成分及热处理参数
        inputs = {
            'image_path': image_path,  # 直接存储图像路径
            'grade_and_process': grade_and_process,  # 合并后的成分和热处理参数
        }

        # 处理硬度曲线作为输出
        labels = torch.tensor(sample['hardness_curve'], dtype=torch.float)

        return {**inputs, "labels": labels}

# 定义神经网络模型
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将 x 展平为 [batch_size, 291 * 768]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model_path = "../../models/facebook-multi-full"
processor = Processor.from_pretrained(model_path)
multi_model = Model.from_pretrained(model_path).to(device)

# 读取JSON文件
data_path = "../../datasets/data_hardness_enhance/data_cut.json"
with open(data_path, 'r') as f:
    data = json.load(f)
    
# 创建数据集
dataset = MetallographicDataset(data)

input_size = 300 * 768
hidden_size = 1024
output_size = 14

model = SimpleMLP(input_size, hidden_size, output_size).to(device)

# 训练配置
learning_rate = 0.001
num_epochs = 300
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 创建 TensorBoardX SummaryWriter
log_dir = f"runs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
writer = SummaryWriter(log_dir=log_dir)

# 加载数据集到DataLoader
batch_size = 32
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("train_dataset:", len(train_dataset))
print("val_dataset:", len(val_dataset))

# 初始化最佳验证损失
best_val_loss = float('inf')

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # 从batch中获取图像路径和文本特征
        image_paths = batch['image_path']
        text_features = batch['grade_and_process'].to(device)  # 将文本特征转移到GPU

        # 使用图像路径加载和处理图像
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        multi_inputs = processor(text=[str(text) for text in text_features], 
                                 images=images, 
                                 return_tensors="pt", 
                                 padding=True).to(device)  # 将处理后的输入转移到GPU

        outputs = multi_model(**multi_inputs)
        multimodal_embeddings = outputs.multimodal_embeddings.to(device)  # 确保特征也在GPU上

        desired_length = 300
        # 如果嵌入长度大于 desired_length，则进行截断
        if multimodal_embeddings.size(1) > desired_length:
            multimodal_embeddings = multimodal_embeddings[:, :desired_length, :]
        # 如果嵌入长度小于 desired_length，则进行填充
        elif multimodal_embeddings.size(1) < desired_length:
            padding_size = desired_length - multimodal_embeddings.size(1)
            padding = torch.zeros(multimodal_embeddings.size(0), padding_size, multimodal_embeddings.size(2)).to(device)
            multimodal_embeddings = torch.cat((multimodal_embeddings, padding), dim=1)

        # 喂给MLP
        outputs_MLP = model(multimodal_embeddings)
        loss = criterion(outputs_MLP, batch['labels'].to(device))  # 将标签转移到GPU

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 记录并打印损失
        running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 每一步都将损失记录到 TensorBoardX
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i)

    # 计算验证集上的损失
    print("---------- 计算验证集上的损失 ----------")
    model.eval()  # 切换到评估模式
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            image_paths = batch['image_path']
            text_features = batch['grade_and_process'].to(device) 

            images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
            multi_inputs = processor(text=[str(text) for text in text_features], 
                                     images=images, 
                                     return_tensors="pt", 
                                     padding=True).to(device) 


            outputs = multi_model(**multi_inputs)
            multimodal_embeddings = outputs.multimodal_embeddings.to(device)  


            if multimodal_embeddings.size(1) > desired_length:
                multimodal_embeddings = multimodal_embeddings[:, :desired_length, :]

            elif multimodal_embeddings.size(1) < desired_length:
                padding_size = desired_length - multimodal_embeddings.size(1)
                padding = torch.zeros(multimodal_embeddings.size(0), padding_size, multimodal_embeddings.size(2)).to(device)
                multimodal_embeddings = torch.cat((multimodal_embeddings, padding), dim=1)
            
            
            outputs_MLP = model(multimodal_embeddings)
            val_loss += criterion(outputs_MLP, batch['labels'].to(device)).item() 

    val_loss /= len(val_loader) 
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

    writer.add_scalar('Validation Loss', val_loss, epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "../../models/multi+mlp_hardness/img+grade+process-epoch300-enhance.pth")
        print(f'Best model saved at epoch {epoch+1} with Validation Loss: {val_loss:.4f}')

    writer.add_histogram('fc1.weight', model.fc1.weight, epoch)
    writer.add_histogram('fc2.weight', model.fc2.weight, epoch)

print('Finished Training')

writer.close()