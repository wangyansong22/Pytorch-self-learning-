#本代码用来讲解torch中module的使用
#之前的dataset和dataloader内容已经涵盖了数据的加载和处理，现在我们的workflow已经到了模型构建的阶段
#torch中的nn.Module类是所有神经网络的基类，任何模型都是它的子类
#我们完全没必要觉得nn.Module神秘，归根到底它只是一个类，只不过里面封装了许多集成度很高的函数罢了
#所以它的使用也和其他对象一样，包含init和forward两个部分
#对于没学过面向对象编程的读者而言，也有形象的理解：init相当于搭建网络，来决定网络由哪些部分组成；forward则相当于告诉用户，这个网络怎么工作，数据在里面怎么流通
'''
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        其他内容，比如网络的结构，参数的初始化等
    def forward(self, x):
        数据在这个网络中怎么流通
'''
#下面我们不涉及到任何深度学习内容，只是用nn.Module来自定义一个类

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        output = x + 1
        return output

#对MyModel进行实例化
model = MyModel()
output = model(torch.tensor(1)) #输入数据必须是tensor类型
print(output)
#tensor(2)

'''
本节内容较简单，是因为没有具体介绍nn.Module中的各种函数，旨在让读者理解神经网络的类的定义和使用
下一节将以简单的卷积网络作为示例，构建一个真正可以用于训练的神经网络
各种各样其他的网络类型，读者可以自行探索
在之后也会发布《Dive into Deep Learning》的系列笔记，敬请期待
'''