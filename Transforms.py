from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path= "Image/f4d760052ef34e449e2aec59a6d18a05.jpg"
img=Image.open(img_path)

writer=SummaryWriter("logs")

#如何使用transform 在transform中选择class进行创建
#创建具体的工具tool=transforms.ToTensor()
#result=tool(input)
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)

writer.add_image("Tensor.img",tensor_img,0)


writer.close()