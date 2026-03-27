from torch.utils.data import Dataset
from PIL import Image  #python image library 用于图像处理
import os #用于操作和管理文件路径

class MyDataset(Dataset): #实例化dataset类，继承自torch.utils.data.dataset

    #函数1，初始化函数，所有类都要用
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        #其实也可以直接传入一个完整的绝对路径，但是像现在这样更加灵活

        self.path = os.path.join(self.root_dir, self.label_dir)
        #os.path.join() 函数将两个路径A和B拼接起来，相当于A/B
        
        self.img_path = os.listdir(self.path)
        #os.listdir() 函数返回一个列表，包含该路径下的所有文件和目录
    
    #函数2，用来获取数据集中每一条数据的信息
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)
        #在init中初始化的self.path其实只是类别路径，而想进一步获取图片路径，需要利用img_path这个列表，通过idx来获取

        image = Image.open(img_item_path)
        #Image.open() 函数的作用是打开图片，获取图片的信息，比如大小，格式，像素值，这样就可以在训练时对图片进行操作
        label = self.label_dir

        return image, label
    
    #函数3，用来获取数据集的长度
    def __len__(self):
        return len(self.img_path)
        #直接返回列表长度即可



#上面的代码块是dataset.py文件中的内容
#下面是使用这个类来创建一个数据集

root_dir = "dataset/hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"

#利用真实的参数创建一个实例化的dataset
ant_dataset = MyDataset(root_dir, ants_label_dir)
bee_dataset = MyDataset(root_dir, bees_label_dir)
dataset = ant_dataset + bee_dataset
#这里使用了加法运算符，将两个dataset合并成一个

#检测dataset功能
print(ant_dataset.path)

img, label = ant_dataset[0]
print(label)

print(len(ant_dataset))
#这里可能会让人感到疑惑：为什么定义的是一个类，却可以用类似于len()和[0][0]这样的类似于list的语法？
#这是因为Dataset类实现了__len__()和__getitem__()这两个魔法方法，使得它具有类似于list的特性