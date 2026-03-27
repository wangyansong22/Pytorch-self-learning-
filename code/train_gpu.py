#本代码用来展示如何在gpu上进行训练。读者如果想复现，则需要保证自己的设备上安装了cuda，并且torch.cuda.is_available()返回True
#同时本代码也会利用time库来计算运行消耗时间来展示gpu训练相比于cpu的加速效果
#本代码绝大部分内容与final_train_demo.py相同，只是将训练过程转移到了gpu上

#我们利用device = torch.device("cuda" if torch.cuda.is_available() else "cpu")来设置设备，然后用 .to(device)将相关数据和模型转移到gpu上
#需要搬到gpu上的内容有：数据，模型，损失函数，优化器

import torch 
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from sequiential_demo import *
import time

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

print(f"训练集下载...")
train_dataset = torchvision.datasets.CIFAR10(root = "../dataset", train = True, transform = torchvision.transforms.ToTensor(), download = True)
print(f"测试集下载...")
test_dataset = torchvision.datasets.CIFAR10(root = "../dataset", train = False, transform = torchvision.transforms.ToTensor())


train_dataloader = DataLoader(train_dataset, batch_size = 64)
test_dataloader = DataLoader(test_dataset, batch_size = 64)

model = CIFAR_CNN()
model.to(device) #把模型搬到gpu上

learning_rate = 1e-3
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

epoch = 20
global_step = 0
loss = 0
writer = SummaryWriter("../logs/final_train_demo")



for epoch in range(epoch):
    model.train()
    for data in train_dataloader:

        start_time = time.time()

        imgs, targets = data

        #把数据搬到gpu上，这里需要设置变量把搬运后的数据赋值给原来的变量，而前面model的参数会就地搬运，所以不需要
        imgs = imgs.to(device)
        targets = targets.to(device)

        loss = loss_func(model(imgs), targets)
        writer.add_scalar("training_loss", loss, global_step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        #终端信息打印
        if global_step % 100 == 0 :
            print(f"epoch {epoch+1} step {global_step} loss: {loss}")
    
    print(f"epoch {epoch+1} finished")

    end_time = time.time()
    print(f"本轮训练时间: {end_time - start_time}秒")


    model.eval()
    accuracy = 0
    correct_num = 0
    with torch.no_grad(): 
        for data in test_dataloader:
            imgs, targets = data

            imgs = imgs.to(device)
            targets = targets.to(device)

            pred = model(imgs)
            correct_num += (pred.argmax(1) == targets).sum()
        accuracy = correct_num / len(test_dataset)

        writer.add_scalar("accuracy", accuracy, epoch+1)
        print(f"epoch {epoch+1} 测试集准确率: {accuracy}")


    torch.save(model.state_dict(), f"../code_output/final_train_demo/ckpts/epoch_{epoch+1}.pth")



#总结 用torch.device("cuda" if torch.cuda.is_available else "cpu")来设置设备，然后用 .to(device)将相关数据和模型转移到gpu上
#model可以直接model.to(device)将模型搬到gpu上
#数据需要手动搬到gpu上，用imgs = imgs.to(device)，targets = targets.to(device)
#cpu和gpu训练耗时可以在code_output/train_gpu中找到，gpu训练速度相比于cpu训练速度提升显著，提升倍数约为240倍左右




'''
到此为止，pytorch入门全部内容已经结束，本仓库全程参考bilibili上up主我是土堆的教学视频，学习流程并非原创
但相比于教学视频，本仓库直接提供了可运行的python代码，并且添加了详细的中文注释，包括官方文档内容与便于理解的注释
同时也将代码运行结果保存在code_output目录下，方便读者查看
用到的数据集也已经下载好。对于设备有限制的读者更加友好
归根到底本仓库只是pytorch入门的自学笔记，目的在于让笔者与读者能对pytorch有一个客观的理解，内容较为基础，很多高级内容都没有涵盖
但是想要系统地学习pytorch所有内容是不现实的，只能在后续开发中逐渐积累，共勉
'''