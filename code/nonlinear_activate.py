#本代码将介绍非线性激活函数的作用，非线性激活函数是卷积神经网络中重要的组成部分
#非线性激活是一种很重要的引入随机性的手段，可以让模型学习到非线性分布
#激活函数有很多，这里只举例两个最常见的例子，ReLU和Sigmoid

import torch
import torch.nn as nn
from torch.nn.functional import relu6
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[-1, 0.5],
                      [1, 2]])
dataset = datasets.CIFAR10(root = "../dataset", train = False, transform = transforms.ToTensor(), download = True)
dataloader = DataLoader(dataset, batch_size = 64, shuffle = False, drop_last = True)

'''
class ReLU(torch.nn.modules.module.Module)
 |  ReLU(inplace: bool = False) -> None
 |  
 |  Applies the rectified linear unit function element-wise.
 |  
 |  :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`
 |  
 |  Args:
 |      inplace: can optionally do the operation in-place. Default: ``False``
 |  
 |  Shape:
 |      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
 |      - Output: :math:`(*)`, same shape as the input.
 |  
 |  .. image:: ../scripts/activation_images/ReLU.png
 |  
 |  Examples::
 |  
 |      >>> m = nn.ReLU()
 |      >>> input = torch.randn(2)
 |      >>> output = m(input)
'''
#inplace：是否用运算结果直接代替原数据，默认False
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x)

relu = ReLU()
output = relu(input)
print(output) #([0, 0.5],
              # [1, 2])


'''
class Sigmoid(torch.nn.modules.module.Module)
 |  Sigmoid(*args: Any, **kwargs: Any) -> None
 |  
 |  Applies the Sigmoid function element-wise.
 |  
 |  .. math::
 |      \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
 |  
 |  
 |  Shape:
 |      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
 |      - Output: :math:`(*)`, same shape as the input.
 |  
 |  .. image:: ../scripts/activation_images/Sigmoid.png
 |  
 |  Examples::
 |  
 |      >>> m = nn.Sigmoid()
 |      >>> input = torch.randn(2)
 |      >>> output = m(input)
 |  
'''
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)

sigmoid = Sigmoid()
output = sigmoid(input)
print(output) #([0.26894143, 0.62245933],
              # [0.7310586, 0.88079706])

step = 0
writer = SummaryWriter("../logs/nonlinear_activate")
for data in dataloader:
    imgs, labels = data
    writer.add_images("input", imgs, step)
    output_relu = relu(imgs)
    writer.add_images("output_relu", output_relu, step)
    output_sigmoid = sigmoid(imgs)
    writer.add_images("output_sigmoid", output_sigmoid, step)
    step += 1
writer.close()

#从输出可以看到，sigmoid作用后的图像明显变得“灰灰的”，是因为每个通道的像素值都被压缩到了0-1之间，所以图像变得很暗
#而ReLU作用后不变，是因为通道值本身就是非负的，所以图像保持不变