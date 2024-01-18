import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./torchvision_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss=nn.CrossEntropyLoss()
tudui=Tudui()
optim=torch.optim.SGD(tudui.parameters(),lr=0.01)
for epoch in range(20):
    running_loss=0.0#学习多轮之后的整体误差
    for data in dataloader:
        imgs, targets=data
        outputs=tudui(imgs)
        result_loss=loss(outputs,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss=running_loss+result_loss
    print(running_loss)
#先调用优化器 再将优化器梯度清零
#再调用反向传播求出每个节点的梯度
#再调用step对每个节点参数调优
