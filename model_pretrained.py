import torchvision
from torch import nn
from torchvision.models import VGG16_Weights


vgg16_False=torchvision.models.vgg16(weights=None)  #weights=VGG16_Weights.DEFAULT
vgg16_true=torchvision.models.vgg16(weights=True)

print(vgg16_true)

train_data=torchvision.datasets.CIFAR10("./torchvision_dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)

vgg16_true.add_module("add_Linear",nn.Linear(1000,10))

#直接在网络模型中修改
#  vgg16_true.classifier[6]=nn.Linear(4096,10)

print(vgg16_true)