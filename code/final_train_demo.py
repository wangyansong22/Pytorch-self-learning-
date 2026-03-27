#本代码将综合前面所有提到的模块，来呈现出最终的可以用于训练的模型架构，相比于optimizer_demo.py，本代码增加了定期的验证环节（用分类准确率而非交叉熵），更加细致的终端输出信息以及定期的权重保存
#本代码在篇幅上会比以往都长，但是所有模块都是在前面已经提到的，所以在阅读和学习上不要有心理负担

#===========导入必要模块=============
import torch 
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from sequiential_demo import *
import time

#============数据集准备===============
print(f"训练集下载...")
train_dataset = torchvision.datasets.CIFAR10(root = "../dataset", train = True, transform = torchvision.transforms.ToTensor(), download = True)
print(f"测试集下载...")
test_dataset = torchvision.datasets.CIFAR10(root = "../dataset", train = False, transform = torchvision.transforms.ToTensor())


train_dataloader = DataLoader(train_dataset, batch_size = 64)
test_dataloader = DataLoader(test_dataset, batch_size = 64)

#============模型定义===============
#由于我们之前已经在sequential_demo.py中定义了模型架构，所以这里直接复用即可，但请注意要在代码开头导入相应模块
model = CIFAR_CNN()


#============损失函数与优化器定义===========
learning_rate = 1e-3
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

#============开始训练==================
epoch = 20
global_step = 0
loss = 0
writer = SummaryWriter("../logs/final_train_demo")

for epoch in range(epoch):
    model.train()

    start_time = time.time()

    for data in train_dataloader:
        imgs, targets = data
        loss = loss_func(model(imgs), targets)
        writer.add_scalar("training_loss", loss, global_step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        #终端信息打印
        if global_step % 100 == 0 :
            print(f"epoch {epoch+1} step {global_step} loss: {loss}")
    
    end_time = time.time()
    print(f"本轮训练时间: {end_time - start_time}秒")

    print(f"epoch {epoch+1} finished")

    #=============每轮训练完成后用验证集做分类准确率验证========
    model.eval() #设置为评估模式
    accuracy = 0
    correct_num = 0
    with torch.no_grad(): #关闭梯度计算，节省计算资源
        for data in test_dataloader:
            imgs, targets = data
            pred = model(imgs)
            correct_num += (pred.argmax(1) == targets).sum()
        accuracy = correct_num / len(test_dataset)

        writer.add_scalar("accuracy", accuracy, epoch+1)
        print(f"epoch {epoch+1} 测试集准确率: {accuracy}")

    #=============保存权重==============
    torch.save(model.state_dict(), f"../code_output/final_train_demo/ckpts/epoch_{epoch+1}.pth")