<div align="center">

# PyTorch快速入门 学习笔记

**PyTorch Learning Note**（GitHub 仓库名：`Pytorch-self-learning-`）· 面向初学者的 **PyTorch 入门与实战** 代码仓库

[环境准备](#环境准备) · [仓库结构](#仓库结构) · [推荐阅读顺序](#推荐阅读顺序)

</div>

---

## 仓库作用

本仓库用于系统整理 **PyTorch 基础到完整训练流程** 的可运行示例：从 `nn.Module`、数据集与 `DataLoader`，到卷积网络、池化与激活、损失与优化器，再到 **CPU/GPU 训练**、**TensorBoard 可视化** 与 **预训练模型** 等主题。

> **参考视频**：https://www.bilibili.com/video/BV1hE411t7RN?t=930.6&p=32。
> **学习目标**：初步了解pytorch库，熟悉深度学习模型搭建与训练流程
> **面向人群**：基本了解深度学习基本概念，熟练使用Linux命令行以及文件管理，初步掌握python编程


---

## 仓库结构

```
Pytorch-self-learning-/          # 与 GitHub 仓库名一致；克隆后目录名通常如此
├── code/                    # 示例源码（按主题分文件，可单独运行）
├── code_output/             # 运行输出：终端记录、说明文字、对比图、权重等（与 code 中脚本对应）
├── dataset/                 # 本地数据集（如 CIFAR-10、教程用图像等）
├── logs/                    # TensorBoard 事件文件（默认不提交，见 [.gitignore](.gitignore)）
├── images/                  # （可选）文档或实验用图片资源
├── requirements.txt         # Python 依赖（见 [环境准备](#环境准备)）
├── LICENSE
├── .gitignore
└── README.md
```

### `code/` 脚本一览

| 文件 | 主题（简述） |
|------|----------------|
| `module_demo.py` | `nn.Module` 基本写法 |
| `dataset.py` | 自定义 `Dataset` |
| `torchvision_dataset.py` | `torchvision` 内置数据集 |
| `dataloader_demo.py` | `DataLoader`：`batch`、`shuffle`、`drop_last` 等 |
| `transforms_demo.py` | 图像 `transforms` |
| `tensorboard_demo.py` | `SummaryWriter` 与标量曲线 |
| `sequiential_demo.py` | `nn.Sequential` 搭建简单 CNN（供其他脚本复用） |
| `cnn.py` | 卷积层 |
| `maxpool.py` | 最大池化 |
| `nonlinear_activate.py` | ReLU、Sigmoid 等非线性激活 |
| `loss.py` | 常见损失函数 |
| `optimizer_demo.py` | 优化器（如 SGD）与训练步 |
| `final_train_demo.py` | 完整训练流程（含验证与 TensorBoard） |
| `train_gpu.py` | GPU 训练与设备放置示例 |
| `pretrained_model.py` | 预训练模型相关 |

> **学习顺序说明**：tensorboard_demo --> dataset.py --> transforms_demo.py --> dataloader_demo.py --> torchvision_dataset.py --> module_demo.py --> cnn.py --> maxpool.py --> nonlinear_activate.py --> loss.py --> optimizer_demo.py --> sequiential_demo.py --> final_train_demo.py --> pretrained_model.py --> train_gpu.py

### `code_output/` 说明

与 `code/` 中实验一一对应，用于存放**可复现的终端输出、说明 `.txt`、对比图、保存的权重**等，便于对照「改了什么 → 现象如何」。
如果读者没有可运行环境，也可以直接参考输出示例



---

## 推荐阅读顺序

建议按 **数据 → 模型组件 → 训练闭环 → 工程扩展** 的顺序阅读；若已熟悉某节可跳过。

### 第一阶段：数据与管线

1. `module_demo.py` — 模块与参数的基本概念  
2. `dataset.py` → `torchvision_dataset.py` — 自定义数据集与官方数据集  
3. `dataloader_demo.py` — 批量与遍历行为  
4. `transforms_demo.py` — 预处理  
5. `tensorboard_demo.py` — 日志与曲线  

### 第二阶段：网络结构

6. `sequiential_demo.py` — 顺序搭建小网络（后续脚本常复用其中模型）  
7. `cnn.py` → `maxpool.py` → `nonlinear_activate.py` — 卷积、池化、激活  

### 第三阶段：训练核心

8. `loss.py` — 损失函数  
9. `optimizer_demo.py` — 优化器与一步训练循环  
10. `final_train_demo.py` — **完整训练 + 验证 + TensorBoard**  

### 第四阶段：设备与进阶

11. `train_gpu.py` — **CPU / GPU** 与 `tensor.to(device)` 等注意事项  
12. `pretrained_model.py` — 预训练与迁移相关  

```mermaid
flowchart LR
  A[数据与 DataLoader] --> B[网络结构]
  B --> C[损失与优化器]
  C --> D[完整训练]
  D --> E[GPU 与预训练]
```


---

## 环境准备

依赖见仓库根目录 [`requirements.txt`](requirements.txt)（`torch`、`torchvision`、`tensorboard`、`Pillow`、`numpy`、`opencv-python`）。克隆后建议用 Conda 单独建环境，再安装依赖：

```bash
git clone https://github.com/wangyansong22/Pytorch-self-learning-.git
cd Pytorch-self-learning-

conda create -n pytorch-learning python=3.10 -y
conda activate pytorch-learning

pip install -r requirements.txt
```

说明：`pip install -r requirements.txt` 默认会装 **CPU 版** PyTorch，可跑通本仓库大部分示例；无独显时把脚本里的 `device` 设为 `cpu` 即可。

**关于 `dataset/`**：该目录**不会**随仓库上传（已在 [`.gitignore`](.gitignore) 中忽略），以控制体积并避免超过 GitHub 单文件大小限制。克隆后首次运行相关脚本时，一般会按代码里的路径**自动下载** CIFAR-10 等到本地 `dataset/`，或按各脚本注释准备自定义数据即可。

若需要 **NVIDIA GPU + CUDA**，请先按 [PyTorch 官网](https://pytorch.org) 用 Conda 安装带 CUDA 的 `pytorch` / `torchvision`，再执行 `pip install tensorboard Pillow numpy opencv-python`（勿再 `pip install torch`，以免覆盖成 CPU 版）。

### 运行示例与 TensorBoard

```bash
cd code
python module_demo.py
```

```bash
cd code
tensorboard --logdir=../logs
```

---

## 推送到 GitHub（首次上传）

仓库根目录已包含 [`.gitignore`](.gitignore)，**默认不跟踪 `logs/`**，避免把 TensorBoard 事件文件推上去。

### 1. 在 GitHub 上新建仓库

1. 登录 [GitHub](https://github.com)，右上角 **+** → **New repository**。  
2. **Repository name** 须与 GitHub 上已有仓库一致（本仓库为 `Pytorch-self-learning-`）。  
3. 选 **Public**；**不要**勾选「Add a README」等（本地已有文件）。  
4. 点 **Create repository**，页面会给出后续命令，与下面类似。

### 2. 本地初始化并推送

在**项目根目录**（含 `README.md`、`code/` 的那一层）执行：

```bash
cd /path/to/Pytorch-self-learning-   # 换成你本机实际路径

git init
git branch -M main
git add .
git status                          # 确认没有 logs/ 被加入
git commit -m "Initial commit: PyTorch learning notes"

git remote add origin https://github.com/wangyansong22/Pytorch-self-learning-.git
git push -u origin main
```

若 GitHub 已要求用 **Personal Access Token** 代替密码，在提示输入密码时粘贴 Token。

**SSH 用户**（已配置 `ssh-key`）可把 `remote` 换成：

```bash
git remote add origin git@github.com:wangyansong22/Pytorch-self-learning-.git
```

### 3. 之后有改动时

```bash
git add .
git commit -m "说明本次改了什么"
git push
```

---

## 许可证与致谢

### 许可证

本仓库采用 **[MIT License](LICENSE)**：可自由使用、修改与再分发；分发时请保留 `LICENSE` 中的版权声明。可将版权行中的 **`wys`** 改成你的常用署名。

### 致谢

学习笔记与代码整理为个人练习用途；主要思路与教学顺序参考 **B 站「小土堆」PyTorch 入门教程**（见上文 [仓库作用](#仓库作用) 中的视频链接）。
