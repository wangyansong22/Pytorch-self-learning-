#本文件用来讲解如何使用torchvision.datasets中现成的数据集
#并且希望通过本文件，使读者能够将之前的内容串联起来，也就是从dataset中读取数据，并通过transforms进行处理，用于下游训练，以及最后的可视化
#但是实际上，我们往往需要自己手写数据集，这里利用torchvision中的数据集只是为了更好展示工作流
#torchvision官方文档：https://docs.pytorch.org/vision/stable/datasets.html
#我们要用到的CIFAR10数据集链接：https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
#这是一个包含10个类别的数据集，每个类别有6000张32x32的彩色图像，通常用于图像分类任务的训练和评估
#可以通过help(torchvision.datasets.CIFAR10)查看数据集

from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

writer = SummaryWriter("logs/torchvision_dataset")

train_data = datasets.CIFAR10(root = "./dataset/CIFAR10", train = True, download = True)
tset_data = datasets.CIFAR10(root = "./dataset/CIFAR10", train = False, download = True)
'''
直接阅读官方文档的能力是很重要的，因为官方文档是权威的，往往比我们自己写的文档更详细
Args:
 |      root (str or ``pathlib.Path``): Root directory of dataset where directory
 |          ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
 |      train (bool, optional): If True, creates dataset from training set, otherwise
 |          creates from test set.
 |      transform (callable, optional): A function/transform that takes in a PIL image
 |          and returns a transformed version. E.g, ``transforms.RandomCrop``
 |      target_transform (callable, optional): A function/transform that takes in the
 |          target and transforms it.
 |      download (bool, optional): If true, downloads the dataset from the internet and
 |          puts it in root directory. If dataset is already downloaded, it is not
 |          downloaded again.
'''
#我们来利用在dataset.py中学习到的Dataset类来查看下载好的数据
img, label = train_data[0] 
print(type(img))
print(label)

#接着利用transforms_demo.py中学习到的transforms来处理数据
transform = transforms.Compose(
    [
        #转换为tensor
        transforms.ToTensor(),
        #resize为224x224
        transforms.Resize((224, 224)),
        #归一化
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    ]
)

for i in range(10):
    img, label = train_data[i]
    img_tensor = transform(img)
    writer.add_image("train_data", img_tensor, i)

writer.close()