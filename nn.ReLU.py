#非线性激活的作用是提高泛化能力
#super(net,self)子类把父类的__init__()放到自己的__init__()当中，
# 这样子类就有了父类的__init__()的那些东西。
#使用父类的初始化方法来初始化子类
import torch
import torchvision
from torch import nn
from torch.nn import ReLU,Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1,-0.5],
                    [-1,3]])

input=torch.reshape(input,(-1,1,2,2))
print(input.shape)

dataset=torchvision.datasets.CIFAR10("../data",train=False,download=True,transform=torchvision.transforms.ToTensor())

dataloader=DataLoader(dataset,batch_size=64)

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.relu1=ReLU
        self.sigmoid1=Sigmoid()

    def forward(self,input):
        output=self.sigmoid1(input)
        return output

relu=ReLU()

writer=SummaryWriter("../logs_relu")
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images("input",imgs,global_step=step)
    output=relu(imgs)
    writer.add_images("output",output,step)
    step+=1

writer.close()