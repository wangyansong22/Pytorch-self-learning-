#本代码用于讲解损失函数，损失函数是模型训练的源动力
#合适的损失函数定义加上合适的优化算法构成完成的训练过程
#workflow大致是：model前向计算loss，loss反向传播计算梯度，优化器根据梯度更新参数，循环直至收敛
#torch.nn中也为我们封装好了许多常见的损失函数，下面对它们进行简单的讲解，但是只涉及到pytorch实现，而不会讲述损失函数为什么这样定义

import torch
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss

#L1损失函数
'''
class L1Loss(_Loss)
 |  L1Loss(size_average=None, reduce=None, reduction: str = 'mean') -> None
 |  
 |  Creates a criterion that measures the mean absolute error (MAE) between each element in
 |  the input :math:`x` and target :math:`y`.
 |  
 |  Shape:
 |      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
 |      - Target: :math:`(*)`, same shape as the input.
 |      - Output: scalar. If :attr:`reduction` is ``'none'``, then
 |        :math:`(*)`, same shape as the input.
'''
input = torch.tensor([1,2,3], dtype = torch.float32)
target = torch.tensor([1,2,5], dtype = torch.float32)

l1_loss = L1Loss()
print(l1_loss(input, target)) #tensor(0.6667)

#MSE损失函数
'''
class MSELoss(_Loss)
 |  MSELoss(size_average=None, reduce=None, reduction: str = 'mean') -> None
 |  
 |  Creates a criterion that measures the mean squared error (squared L2 norm) between
 |  each element in the input :math:`x` and target :math:`y`.
 |  
 |  Shape:
 |      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
 |      - Target: :math:`(*)`, same shape as the input.
 |  
'''
mse_loss = MSELoss()
print(mse_loss(input, target)) #tensor(1.3333)

#CrossEntropy损失函数
'''
class CrossEntropyLoss(_WeightedLoss)
 |  CrossEntropyLoss(weight: torch.Tensor | None = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean', label_smoothing: 
 |  float = 0.0) -> None
 |  
 |  This criterion computes the cross entropy loss between input logits
 |  and target.
 |  
 |  It is useful when training a classification problem with `C` classes.
 |  If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
 |  assigning weight to each of the classes.
 |  This is particularly useful when you have an unbalanced training set.
 |  
 |  Shape:
 |      - Input: Shape :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
 |        in the case of `K`-dimensional loss.
 |      - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
 |        :math:`K \geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`. The
 |        target data type is required to be long when using class indices. If containing class probabilities, the
 |        target must be the same shape input, and each value should be between :math:`[0, 1]`. This means the target
 |        data type is required to be float when using class probabilities. Note that PyTorch does not strictly enforce
 |        probability constraints on the class probabilities and that it is the user's responsibility to ensure
 |        ``target`` contains valid probability distributions (see below examples section for more details).
 |      - Output: If reduction is 'none', shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
 |        in the case of K-dimensional loss, depending on the shape of the input. Otherwise, scalar.
'''
input = torch.tensor([[1,2,3], [4,5,6]], dtype = torch.float32)
target = torch.tensor([[1,0,0], [0,1,0]], dtype = torch.float32)
#注：假设这是一个三分类问题，input中每个一维张量是模型对同一个输入是三个类别的预测概率，target中每个一维张量是该输入的实际类别（one-hot表示）
#我们可以注意到，input中每个一维张量的逐元素和并不是1，这是因为pytorch中的交叉熵函数会默认对数据进行softmax处理
cross_entropy_loss = CrossEntropyLoss()
print(cross_entropy_loss(input, target)) #tensor(1.9076)