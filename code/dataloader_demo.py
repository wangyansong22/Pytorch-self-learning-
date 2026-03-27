#本代码用来讲解torch中dataloader的使用
#或许有人会疑惑，明明dataset已经实现了数据集的加载，为什么还需要dataloader呢？
#诚然dataset已经实现了对数据集的加载与逐条读取，但是它只支持最原始的例如img,label = dataset[i]这样的读取方式
#而真正可以送进模型进行训练的数据往往需要更加灵活的组织形式，比如成批地送入gpu，或者打乱顺序，或者进行多线程并行计算等
#总结来说，dataloader可以把我们的数据组织成更多样的形式，举例来说，dataset是摸牌，dataloader则是把摸到的牌按照一定的规则组织成一副扑克牌，方便打出

#仍然利用CIFAR10数据集来进行说明
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

test_data = datasets.CIFAR10(root = "./dataset/CIFAR10", train = False, download = True, transform = transforms.ToTensor())
'''
下面要用DataLoader来组织数据, DataLoader的部分常见参数如下
Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.
Args:
 |      dataset (Dataset): dataset from which to load the data.
 |      batch_size (int, optional): how many samples per batch to load
 |          (default: ``1``).
 |      shuffle (bool, optional): set to ``True`` to have the data reshuffled
 |          at every epoch (default: ``False``).
 |      num_workers (int, optional): how many subprocesses to use for data
 |          loading. ``0`` means that the data will be loaded in the main process.
 |          (default: ``0``)
 |      drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
 |          if the dataset size is not divisible by the batch size. If ``False`` and
 |          the size of dataset is not divisible by the batch size, then the last batch
 |          will be smaller. (default: ``False``)

 batch_size: 每批送入的样本数量
 shuffle: 是否打乱数据集(不同epoch之间顺序是否不同)
 num_workers: 多少个子进程来加载数据
 drop_last: 是否丢弃最后一个不完整的batch（比如数据集大小=10，batch_size=3，那么最后一个batch只有1个样本，如果drop_last=True，则丢弃，如果drop_last=False，则保留）

'''
#我们构建三个不同的DataLoader，来直观感受不同参数的影响
test_loader_1 = DataLoader(dataset = test_data, batch_size = 64, shuffle = True, num_workers = 0, drop_last = False)
test_loader_2 = DataLoader(dataset = test_data, batch_size = 64, shuffle = False, num_workers = 0, drop_last = False)
test_loader_3 = DataLoader(dataset = test_data, batch_size = 64, shuffle = True, num_workers = 0, drop_last = True)

writer = SummaryWriter("logs/dataloader_demo")


step_1, step_2, step_3 = 0, 0, 0
for epoch in range(10):
    
    for data in test_loader_1: #为什么能直接用for data in test_loader_1: 来遍历DataLoader？因为“provides an iterable over the given dataset”告诉我们DataLoader是一个可迭代的对象
        img, label = data
        writer.add_images(f"test_loader_1_epoch{epoch}", img, step_1)
        step_1 += 1
    for data in test_loader_2:
        img, label = data
        writer.add_images(f"test_loader_2_epoch{epoch}", img, step_2)
        step_2 += 1
    for data in test_loader_3:
        img, label = data
        writer.add_images(f"test_loader_3_epoch{epoch}", img, step_3)
        step_3 += 1

writer.close()