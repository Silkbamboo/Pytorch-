#池化函数使用某一位置的相邻输出的总体统计特征来代替网络在该位置的输出。本质是 降采样，可以大幅减少网络的参数量。
#池化目的是减少数据量，减少计算参数 （压缩特征）
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./nn.dataset",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)

# input=torch.tensor([[1,2,0,3,1],
#                     [0,1,2,3,1],
#                     [1,2,1,0,0],
#                     [5,2,3,1,1],
#                     [2,1,0,1,1]],dtype=torch.float32)#最大池化无法对long数据类型进行池化 将整数变为浮点数
#
# input=torch.reshape(input,(-1,1,5,5))
# print(input.shape)

class MaxPool(nn.Module):#构建神经网络
    def __init__(self):
        super(MaxPool,self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,input):
        output=self.maxpool1(input)
        return output

maxpool=MaxPool()#调用神经网络
# output=maxpool(input)
# print(output)

writer=SummaryWriter("./logs_maxpool")
step=0

for data in dataloader:
    imgs,targets=data
    writer.add_images("input",imgs,step)
    output=maxpool(imgs)
    writer.add_images("output",output,step)
    step=step+1

writer.close()

# import torch
# import torchvision
# from torch import nn
# from torch.nn import MaxPool2d
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
#
# dataset = torchvision.datasets.CIFAR10("../data", train=False, download=True,
#                                        transform=torchvision.transforms.ToTensor())
#
# dataloader = DataLoader(dataset, batch_size=64)
#
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)
#
#     def forward(self, input):
#         output = self.maxpool1(input)
#         return output
#
# tudui = Tudui()
#
# writer = SummaryWriter("../logs_maxpool")
# step = 1
#
# for data in dataloader:
#     imgs, targets = data
#     writer.add_images("input", imgs, step)
#     output = tudui(imgs)
#     writer.add_images("output", output, step)
#     step = step + 1
#
# writer.close()