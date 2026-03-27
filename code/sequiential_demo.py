#本代码目的有两个：讲解nn.Sequential的用法，并搭建一个真实可以用于训练的卷积神经网络，网络pipeline可以在/code_output/sequiential_demo中找到
import torch
import torch.nn as nn




class CIFAR_CNN(nn.Module):  
    def __init__(self):
        super().__init__()
        conv1 = nn.Conv2d(3, 32, 5, padding=2) #padding和stride是通过已知的输入和输出通道数计算出来的。这里相当于有32个5*5*3的卷积核
        pool1 = nn.MaxPool2d(2)
        conv2 = nn.Conv2d(32, 32, 5, padding=2)
        pool2 = nn.MaxPool2d(2)
        conv3 = nn.Conv2d(32, 64, 5, padding=2)
        pool3 = nn.MaxPool2d(2)
        flatten = nn.Flatten() #这个模块没有单独介绍，它的原理很简单，对于二维张量，以行优先的方式展平为一维张量
        linear1 = nn.Linear(64*4*4, 10)

        self.pipeline = nn.Sequential(conv1, pool1, conv2, pool2, conv3, pool3, flatten, linear1)
        #nn.sequential的作用和torchvision.transforms.Compose类似，都是把多个模块组合在一起
    
    def forward(self, x):
        return self.pipeline(x)

#实例化
model = CIFAR_CNN() 

#打印模型结构
print(model)

#检验输出形状的正确性
input = torch.zeros(1, 3, 32, 32)
output = model(input)
print(output.shape) #(1, 10)

#至此，数据加载与模型搭建的内容就完成了，接下来开始学习如何训练模型