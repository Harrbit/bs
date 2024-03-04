import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class EnhancedNN(nn.Module):  
    def __init__(self, state_dim, action_dim):  
        super(EnhancedNN, self).__init__()  
          
        # 假设state_dim是一个形如(channels, height, width)的元组  
        # 如果state_dim不是这种格式，可能需要对输入数据进行适当的预处理  
          
        # 卷积层  
        self.conv1 = nn.Conv2d(in_channels=state_dim[0], out_channels=32, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  
          
        # 最大池化层  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
          
        # 线性层  
        # 首先，计算卷积层和池化层之后的特征图的平坦化尺寸  
        flattened_size = 64 * (state_dim[1] // 2) * (state_dim[2] // 2)  
        self.fc1 = nn.Linear(flattened_size, 256)  
        self.fc2 = nn.Linear(256, action_dim)  
          
        # Dropout层  
        self.dropout = nn.Dropout(0.5)  
          
    def forward(self, x):  
        # 卷积层  
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))  
          
        # 最大池化层  
        x = self.pool(x)  
          
        # 平坦化  
        x = x.view(x.size(0), -1)  
          
        # 全连接层  
        x = F.relu(self.fc1(x))  
        x = self.dropout(x)  
        x = self.fc2(x)  
          
        return x  
  
# 假设state_dim是(3, 64, 64)，即3个通道，64x64像素的图像  
state_dim = (3, 64, 64)  
action_dim = 10  # 假设有10个可能的动作  
  
model = EnhancedNN(state_dim, action_dim)  
print(model)