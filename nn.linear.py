import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Linear

dataset=torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset,batch_size=64,drop_last=True)#drop_last什么意思

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1=Linear(196608,10)

    def forward(self,input):
        output=self.linear1(input)
        return output

linear=Tudui()

for data in dataloader:
    imgs,targets=data
    print(imgs.shape)
    output=torch.flatten(imgs)#将数据拉平 降成一维
    print(output.shape)
    output=linear(output)
    print(output.shape)

