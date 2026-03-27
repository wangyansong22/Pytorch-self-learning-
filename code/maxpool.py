#本代码将介绍池化层的作用，池化层也是卷积神经网络中重要的组成部分
#其作用可以被理解为一种将采样，保留主要特征。我们介绍最大池化层
#池化层具体的原理不再赘述

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

dataset = datasets.CIFAR10(root = "../dataset", train = False, transform = transforms.ToTensor(), download = True)
dataloader = DataLoader(dataset, batch_size = 64, shuffle = False, drop_last = True)

input = torch.tensor([[1, 2, 3, 4], 
                      [5, 6, 7, 8], 
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]], dtype = torch.float32) #这里默认数据是float32类型，否则默认是long，后面MaxPool2d会报错
print(input.shape) #[4, 4]
input = input.reshape(-1, 1, 4, 4) #可以参考下面的输入格式要求，reshape不会改变原数据，而是返回一个新的tensor
print(input.shape) #[1, 1, 4, 4]
print(input)
'''
tensor([[[[ 1.,  2.,  3.,  4.],
          [ 5.,  6.,  7.,  8.],
          [ 9., 10., 11., 12.],
          [13., 14., 15., 16.]]]])
'''

class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size = 3, ceil_mode = False)
        #ceil_mode，可以简单理解成，如果待池化的部分小于kernel_size，那也照样进行池化；False则是直接舍弃

        '''
        |  Args:
        |      kernel_size: The size of the sliding window, must be > 0.
        |      stride: The stride of the sliding window, must be > 0. Default value is :attr:`kernel_size`.
        |      padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        |      dilation: The stride between elements within a sliding window, must be > 0. 空洞卷积
        |      return_indices: If ``True``, will return the argmax along with the max values.
        |                      Useful for :class:`torch.nn.MaxUnpool1d` later
        |      ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
        |                 ensures that every element in the input tensor is covered by a sliding window.

        |  Shape:
        |      - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        |      - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`,
        '''

    def forward(self, x):
        return self.MaxPool(x)

#实例化
maxpool = MaxPool()

output = maxpool(input) 
print(output.shape) #[1, 1, 1, 1]

#下面还是在CIFAR10上来看一下池化层的可视化结果
writer = SummaryWriter("../logs/maxpool")
step = 0
for data in dataloader:
    imgs, label = data
    print(imgs.shape) #[64, 3, 32, 32]
    writer.add_images("input", imgs, step)
    output = maxpool(imgs)
    writer.add_images("output", output, step)
    step += 1
writer.close()

#从输出可以看出，池化后的图像貌似变模糊了，但是还保留着图片的主要特征，我们仍然可以大致辨别出来图片中的物体，这就是最大池化层的作用