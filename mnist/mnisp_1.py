import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
epoch = 10
pipeline = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                          torchvision.transforms.Normalize(mean = (0.1307,), std = (0.3081,))])

device = torch.device("cuda")
train_ds = torchvision.datasets.MNIST('data',
                                      train=True,
                                      transform=pipeline,  # 将数据类型转化为Tensor
                                      download=True)

test_ds = torchvision.datasets.MNIST('data',
                                     train=False,
                                     transform=pipeline,  # 将数据类型转化为Tensor
                                     download=True)


train_dl = torch.utils.data.DataLoader(train_ds,
                                       batch_size=batch_size,
                                       shuffle=True)

test_dl = torch.utils.data.DataLoader(test_ds,
                                      batch_size=batch_size)

'''imag = iter(train_dl)
print(next(imag))'''

class Module(nn.Module):
    def __init__(self):
        super().__init__
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*10*10, 250)
        self.fc2 = nn.Linear(250, 10)
    def forward(self,x):
        input_size = x.size(0)
        x = x.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(input_size, -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

model = Module.to(device)
optimizer = optim.Adam(model.parameters())

def train(model, device, train_dl, optimizer, epoch):
    model.train()
    for batch_index,(data, target) in enumerate(train_dl):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_index%3000 == 0:
            print("train epoch{} \t loss:{:.6f}".format(epoch, loss.item()))
