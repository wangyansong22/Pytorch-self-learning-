#本代码比较简短，用来讲解如何使用torch中提供的预训练模型以及如何对模型架构进行修改，本代码将使用torchvision.models中的预训练模型

import torch
import torchvision
'''
vgg16(*, weights: Optional[torchvision.models.vgg.VGG16_Weights] = None, progress: bool = True, **kwargs: Any) -> torchvision.models.vgg.VGG
    VGG-16 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    
    Args:
        weights (:class:`~torchvision.models.VGG16_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG16_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.
'''


vgg_pretrained = torchvision.models.vgg16(weights= torchvision.models.vgg.VGG16_Weights) #加载预训练模型
vgg_raw = torchvision.models.vgg16() #加载原始模型
print(vgg_pretrained) #可以在code_output/pretrained_model/before.txt中查看模型结构

#对预训练模型进行修改，比如修改最后一层线性层，使其输出10类而不是1000类
vgg_pretrained.classifier[-1] = torch.nn.Linear(4096, 10)
print(vgg_pretrained) #可以在code_output/pretrained_model/after1.txt中查看模型结构

#或者我们也可以添加一层线性层，使其输出10类而不是1000类
vgg_raw.classifier.add_module("add_linear", torch.nn.Linear(1000, 10)) #子模块的名字叫做add_linear，可以通过vgg_raw.classifier.add_linear访问

'''
    def add_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Add a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (str): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
'''
print(vgg_raw) #可以在code_output/pretrained_model/after2.txt中查看模型结构

#通过复用torch中已有的模型和预训练权重，并对模型结构进行修改，可以极大便利我们的开发过程，不必从零开始搭建模型
#但是注意修改模型架构之后预训练权重将变得不再可用，需要重新训练

