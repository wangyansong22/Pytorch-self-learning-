#之前我们已经初步了解了nn.Module的基本框架，但是对其中的各种子module没有进行任何介绍
#本代码将以卷积神经网络为例，来具体地介绍如何搭建神经网络
#这需要对读者对CNN有基本了解；如果没有，则可参考《深度学习入门：基于python的理论与实现》，这里不再赘述

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

#先导入数据集
dataset = datasets.CIFAR10(root = "dataset", train = False, transform = transforms.ToTensor(), download = True)
dataloader  = DataLoader(dataset, batch_size = 64, shuffle = False, drop_last = True)

#接下来搭建一个最简单的卷积神经网络，只包含一层卷积层
#但实际上卷积神经网络包括卷积层，池化层，以及全连接层
'''
 |  Args:
 |      in_channels (int): Number of channels in the input image
 |      out_channels (int): Number of channels produced by the convolution
 |      kernel_size (int or tuple): Size of the convolving kernel
 |      stride (int or tuple, optional): Stride of the convolution. Default: 1
 |      padding (int, tuple or str, optional): Padding added to all four sides of
 |          the input. Default: 0
 |      dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
 |      groups (int, optional): Number of blocked connections from input
 |          channels to output channels. Default: 1
 |      bias (bool, optional): If ``True``, adds a learnable bias to the
 |          output. Default: ``True``
 |      padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
 |          ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
 关键参数：输入通道数，输出通道数，卷积核大小，步幅，填充，偏置
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() #这是python2的写法，对于python3而言，可以直接super().__init()__即可
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 1, kernel_size = 3, stride = 1, padding = 0)
        #这里区分一下torch.nn和torch.functional的区别。我们可以简单地认为torch.nn中的参数是会更新的，会注册到nn.Module中，而torch.functional中的参数是不可训练的
        #所以这里我们现在也并没有人为规定卷积核的值，而是nn.Conv2d会自动初始化
        #另外还有一点需要说明，就是卷积核实际的尺寸是3*3*3，因为卷积核数量=输出通道=1，卷积核通道数=输入通道=3

    def forward(self, x):
        return self.conv1(x)

#实例化网络
cnn = CNN()

#接下来利用tensorboard来可视化一下卷积结果
writer = SummaryWriter("logs/cnn")
step = 0

for data in dataloader:
    img, label = data
    print(img.shape) #[64, 3, 32, 32]
    output = cnn(img)
    print(output.shape) #[64, 1, 30, 30] 30是怎么来的？kernal_size=3, stride=1, padding=0
    writer.add_images("input", img, step)
    writer.add_images("output", output, step)
    step += 1

writer.close()

#在/code_output/cnn目录下，可以看到卷积可视化结果，我们可以发现卷积结果貌似是原图像的黑白版本，这是因为输出通道是1，所以只保留了灰度信息
