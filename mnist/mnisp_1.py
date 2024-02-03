import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
epoch = 5
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
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*10*10, 250)
        self.fc2 = nn.Linear(250, 10)
    def forward(self,x):
        input_size = x.size(0)
        x = self.conv1(x)
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

model = Module().to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

def train(model, device, train_dl, optimizer):
    model.train()
    train_acc, train_loss = 0, 0
    for data, target in train_dl:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        train_acc += (output.argmax(1) == target).type(torch.float).sum().item()
        train_loss += loss.item()
    train_acc /= len(train_dl.dataset)
    train_loss /= len(train_dl)
    return train_loss, train_acc


def test(test_dl, model, loss_fn):
    test_loss, test_acc = 0, 0
    for data, target in test_dl:
        output = model(data)
        loss = loss_fn(output, target)

        test_loss += loss
        test_acc += (output.argmax(1) == target).type(torch.float).sum.item
    test_loss /= len(test_dl.dataset)
    test_acc /= len(test_dl.item())
    return test_loss, test_acc

def implementation(train_dl, test_dl, epoch, model):
    for i in range(epoch):
        model.train()
        epoch_train_loss, epoch_train_acc = train(model, device, train_dl, optimizer)
        print(epoch_train_acc, epoch_train_loss)

implementation(train_dl, test_dl, epoch, model)
