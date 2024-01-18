import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_tranforms=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


train_set=torchvision.datasets.CIFAR10(root="./torchvision_dataset",train=True,transform=dataset_tranforms,download=True)
test_set=torchvision.datasets.CIFAR10(root="./torchvision_dataset",train=False,transform=dataset_tranforms,download=True)
#transform=dataset_tranforms已经将图片转换为totensor类型

# print(test_set[0])
# print(test_set.classes)
#
# img,target=test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

#print(test_set[0])、
writer=SummaryWriter("p10")
for i in range(10):
    img,target=test_set[i]
    writer.add_image("test_set",img,i)

writer.close()