from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter("logs")
img=Image.open("Image/f4d760052ef34e449e2aec59a6d18a05.jpg")#image是PIL数据类型
print(img)

#ToTensor
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)#image是tensor数据类型
writer.add_image("ToTensor",img_tensor)


#Normalize
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm=trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)

#Resize
print(img.size)
trans_resize=transforms.Resize((512, 512))
#img PIL -> resize -> img_resize PIL
img_resize=trans_resize(img)
print(img_resize)
#img_resize PIL -> totensor ->img_resize tensor
img_resize=trans_totensor(img_resize)
writer.add_image("Resize",img_resize,1)
#先对PIL进行resize再将PIL_resize转换为totensor类型
print(img_resize)
print(type(img_resize))#查看返回值类型

#Compose-resize-2
trans_resize_2=transforms.Resize(512)
#PIL->PIL->tensor
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])#compose作用就是多步合一步
img_resize_2=trans_compose(img)
writer.add_image("Resize",img_resize,2)



writer.close()

