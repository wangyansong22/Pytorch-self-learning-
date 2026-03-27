#tensorboard是tensorflow中用来可视化的工具，在pytorch高版本中被集成到了torchvision.utils.tensorboard中
#虽然相比于wandb，tensorboard略显老旧，但是其本地化的属性使得它的使用更加方便
#熟练地使用tensorboard可以让你在训练过程中直观地看到比如loss、learning rate、accuracy等指标的变化情况   

from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2

'''
我们要用到SummaryWriter来创建一个写入器，它的初始化如下：
其实除了log_dir, 其他基本用不到
    def __init__(
        self,
        log_dir=None,
        comment="",
        purge_step=None,
        max_queue=10,
        flush_secs=120,
        filename_suffix="",
    ) -> None:
        """Create a `SummaryWriter` that will write out events and summaries to the event file.
'''

#初始化
writer = SummaryWriter("logs") 
#logs是文件夹名，用来存储tensorboard的日志
#运行该python文件后，文件夹下会生成logs子文件夹，里面会存储event文件(每运行一次就会生成一个)

#函数1，add_scalar(),简单来说就是y-x图，一般来说x都是training steps
'''
def add_scalar(
        self, tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False
    ) -> None:
    参数：
        tag：数据标识符
        scalar_value：标量值，用来记录数据是第几次生成的
        global_step：全局步数，用来记录数据是第几次生成的
        walltime：时间戳，用来记录数据生成的时间
'''
for step in range(100):
    writer.add_scalar("y=2x", 2 * step, step) #三个参数分别是：图表名称，纵轴因变量，横轴自变量
    writer.add_scalar("y=pow(x,2)", step**2, step)

#函数2，add_images(),用来将图像数据添加到tensorboard中
'''
def add_image(
        self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW"
    ) -> None:
    参数：
        tag：数据标识符
        img_tensor：图像数据，可以是一个张量，也可以是一个numpy数组
        global_step：全局步数，用来记录数据是第几次生成的
        walltime：时间戳，用来记录数据生成的时间
        dataformats：图像数据格式，可以是"CHW"，"HWC"，"HW"，"WH"等
'''
image_path = "dataset/hymenoptera_data/train/ants/0013035.jpg"

img_PIL = Image.open(image_path) #要先打开，才能转换为np
print(type(img_PIL))
img_array = np.array(img_PIL)#转换为numpy数组
print(type(img_array))
print(img_array.shape)

img_cv = cv2.imread(image_path)
print(type(img_cv)) #opencv读取图像直接就是np数组格式


writer.add_image("test", img_array, 1, dataformats = 'HWC')
#dataformats用来表示图像的形状，因为add_image默认是CHW格式，所以这里需要指定为HWC格式

#记住最后一定要关闭writer，否则会占用大量内存
writer.close()

# 如何打开tensorboard? 运行程序之后，运行tensorboard --logdir logs，在local host中查看；可以用--port指定端口，默认是6006