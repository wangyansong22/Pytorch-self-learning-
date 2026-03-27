#本文件用来展示如何使用transforms对图像数据进行操作（比如裁剪、缩放、旋转等）
#我们只介绍常见的几个函数，并用Compose()将它们组合起来；其他不常见函数可参考官方文档

from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

#首先我们要明确Transforms的使用逻辑：它实际上是一个python文件，里面包含很多类，每个类都对应一个transform操作
#每个类里面都有__call__()函数，来保证我们可以直接用classname（参数）来调用这个transform操作
#所以，transforms的使用包括两步：
#                             （1）实例化所有要用到的transforms类
#                             （2）用Compose()将它们组合起来，形成一条图像处理“流水线”

img = Image.open("images/eg1.jpeg")
print(img) #PIL格式

#类1：ToTensor()，将PIL或者ndarray格式数据转换为tensor，后者相比于前两种更加常用
to_tensor = transforms.ToTensor() #实例化
img_tensor = to_tensor(img) #使用
print(img_tensor)
print(type(img_tensor))

#类2：Resize(), 将图像缩放到指定大小
#从官方文档可知，Resize()接受名为size的参数，该参数可以是序列(h, w), 也可以是int；若是int，则是等比例缩放，短边对齐到int
#PIL和tensor格式都支持Resize()
resize = transforms.Resize((512, 512))
img_resize = resize(img_tensor)
print(img_resize.shape) #(3, 512, 512)

#类3：Normalize()，对图像逐通道进行归一化，分布变为（x-mean）/std
#通常选定mean和std都是0.5，这样可以把分布在（0，1）之间的数据归一化到（-1，1）之间
#只接受tensor格式
norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = norm(img_tensor)
print(img_norm)

#类4：Compose()，将多个transforms组合起来，形成一条图像处理“流水线”
#Compose()的输入接受以transforms类为元素的列表，输出是一个transforms对象
compose = transforms.Compose([to_tensor,
                              resize,
                              norm])
#或者更常见的，可以直接compose = transforms.Compose([transforms.ToTensor(), 
#                                                   transforms.Resize((512, 512)), 
#                                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
img_compose = compose(img)
print(img_compose)

#更多transform操作，请参考官方文档

#下面我们用tensorboard来可视化一下每个transforms的效果
writer = SummaryWriter("logs/transforms_demo")
writer.add_image("img_tensor", img_tensor, 0)
writer.add_image("img_resize", img_resize, 0)
writer.add_image("img_norm", img_norm, 0)
writer.close()


'''
tensorboard输出：code_output/transforms_demo/tensorboard_vis.png
终端输出：code_output/transforms_demo/terminal_output.txt

'''