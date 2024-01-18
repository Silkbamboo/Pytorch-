
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch import nn

train_data=torchvision.datasets.CIFAR10(root="./torchvision_dataset",train=True,transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data=torchvision.datasets.CIFAR10(root="./torchvision_dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                         download=True)
# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10 相当于仅打印数字
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

#创建网络
zejiang=Zejiang()

#损失函数
loss_fn=nn.CrossEntropyLoss()  #这里少一个括号都会导致错误 为什么

#优化器
learning_rate=1e-2
optimizer=torch.optim.SGD(zejiang.parameters(),lr=learning_rate)

total_train_step=0

total_test_step=0

epoch=10

writer=SummaryWriter("../logs_train")

for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i+1))

    zejiang.train()
    for data in train_dataloader:
        imgs,targets=data
        outputs=zejiang(imgs)
        loss=loss_fn(outputs,targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step=total_train_step+1
        if total_train_step%100==0:
            print("训练次数：{},loss:{}".format(total_train_step,loss.item()))

    zejiang.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            outputs = zejiang(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()   #存疑
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(zejiang, "zejiang_{}.pth".format(i))
    print("模型已保存")

writer.close()
