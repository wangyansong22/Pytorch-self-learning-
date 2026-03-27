#本节来补充深度学习训练的几乎是最后一个板块，优化器
#每个优化器背后都对应着一套优化算法，它们之前存在继承关系，也各自有优点。
#常见的优化器包括SGD，Momentum，Adam等，在这里不做介绍，有兴趣的可以自行了解。
#Adam优化器恐怕是应用最广泛的优化器，但是它有些复杂，本代码将采用SGD优化器做讲解


import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from sequiential_demo import CIFAR_CNN #直接复用sequential_demo.py中定义的模型

#实例化
model = CIFAR_CNN()

#定义优化器
'''
>>> dir(torch.optim)
['ASGD', 'Adadelta', 'Adafactor', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Muon', 'NAdam', 'Optimizer', 'RAdam', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '_adafactor', '_functional', '_muon', 'lr_scheduler', 'swa_utils']
'''
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #第一个参数相当于告诉优化器要优化哪个模型的参数
'''
 |  Args:
 |      params (iterable): iterable of parameters or named_parameters to optimize
 |          or iterable of dicts defining parameter groups. When using named_parameters,
 |          all parameters in all groups should be named
 |      lr (float, Tensor, optional): learning rate (default: 1e-3)
'''

#加载数据集
train_dataset = torchvision.datasets.CIFAR10(root="../dataset", train= True, transform = torchvision.transforms.ToTensor(), download= True)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64,  shuffle = True)

#tensorboard
writer= SummaryWriter("../logs/optimizer_demo")

step = 0

#训练模型
for epoch in range(20):

    loss_per_epoch = 0

    for data in dataloader:
        imgs, targets = data

        #forward
        output = model(imgs)

        #定义损失函数并计算损失
        loss_func = torch.nn.CrossEntropyLoss() #多分类问题
        loss = loss_func(output, targets)
        writer.add_scalar("training_loss", loss, step) #绘制loss曲线
        loss_per_epoch += loss

        #backward
        optimizer.zero_grad() #先清空优化器中的梯度记录，防止梯度累加
        loss.backward() #计算单步损失的梯度
        optimizer.step() #更新参数

        step += 1
    
    print(f"epoch {epoch+1} finished, loss for this epoch: {loss_per_epoch}")
    #可以简单地记录训练进度并检测优化是否正常进行。在笔记本上跑可以需要几分钟时间，可以耐心等待
    
writer.close()
    
#至此，我们完成了一个最简单的深度学习模型搭建与训练，训练曲线可在code_output/optimizer_demo中找到