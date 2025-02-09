---
date: 2025年1月18日
tags:
  - 深度学习
---

## 常用的库
#### torch
`import torch`导入torch库，包含了pytorch的所有核心功能
- `torch.utils`
	`from torch.utils import data`,`data.TensorDataset`可以用来构造python的数据迭代器，提供随机批量。
- `torch.nn`
	`from torch import nn`,`nn`是neural network的缩写，集成了大量的函数，详情见[[torch.nn]]。
- `torch.optim`
	`torch.optim`中包含多种常用的优化器，包括SGD优化器，Adam优化器，Adagrad优化器等，详情见[chatGPT聊天](https://chatgpt.com/share/67930312-df1c-800c-a9be-da4a31fcc256)

#### `torchvision`
`torchvision` 是一个基于 PyTorch 的计算机视觉库，提供了一些常用的工具和功能，帮助开发者更轻松地进行图像和视频处理任务。它包含了几个重要模块：

1. **数据集（Datasets）**：`torchvision` 提供了一些常见的公开数据集的接口，比如 CIFAR-10、ImageNet、COCO、MNIST 等。这些数据集可以直接下载并使用，避免了手动处理数据加载的麻烦。
    
2. **数据变换（Transforms）**：提供了图像数据预处理和增强的常用方法，例如裁剪、旋转、标准化、调整大小等。这些变换可以链式操作，方便构建数据流水线。
    
3. **模型（Models）**：`torchvision` 包含了许多预训练的深度学习模型，常用于图像分类、目标检测、实例分割等任务，如 ResNet、VGG、AlexNet、Faster R-CNN 等。这些模型可以用于迁移学习或者直接进行推理。
    
4. **工具（Utils）**：`torchvision` 提供了很多图像处理的工具，比如图像可视化、边界框的操作等。


### random
`import random` 导入random库，可以用于生成随机序列等，用于随机梯度下降法中的批量生成。
- `random.shuffle(x) #x表示要打乱的list列表` : 是 Python 的 `random` 模块中用于对一个列表（或类似序列类型）中的元素进行随机打乱的函数。并且`random.shuffle` 会直接修改该列表，不会返回新的列表。

### d2l
`from d2l import torch as d2l` 导入d2l库中的pytorch模块，详情可见[[d2l]]

## pytorch函数

```python
torch.tensor()
#生成一个张量数组
#eg:
torch.tensor([3.,4,5],dtype=torch.float32,requires_grad=True,retain_graph=True,create_graph=True)

torch.arange()
#生成元素为等差数列的张量数组
#eg:
torch.arange([4.0],dtype=torch.float32,requires_grad=True)

torch.normal()
#用于生成服从正态分布（高斯分布）的随机张量。
#eg:
torch.normal(0,1,(3,3))#生成一个元素的随机数在0,1之间的3x3矩阵

torch.matmul()#matrix multiplication
#用于进行矩阵乘法
#eg:
y=torch.matmul(X,w)
```
`torch.normal`与`torch.matmul`详情可见[chatGPT聊天](https://chatgpt.com/share/678b8b4a-7ff8-800c-8420-ffd4153bb3a9)

## 损失函数

### [L1 Loss ， L2 Loss 与 L1 Smooth](https://zhuanlan.zhihu.com/p/641356166)


## 数据集
### 1. MNIST数据集
- **内容**：包含 28x28 像素的手写数字图像，数字范围从 0 到 9，总共有 60,000 张训练图片和 10,000 张测试图片。
- **目标**：任务是对手写数字进行分类，即将每个图像分类为 0 到 9 之间的数字。
- **难度**：相对简单，因为手写数字在图像中通常比较清晰，并且背景较为简单。
### 2. Fashion-MNIST数据集
- **内容**：包含 28x28 像素的灰度时尚商品图像，如 T 恤、裤子、鞋子、包等。共有 60,000 张训练图片和 10,000 张测试图片。
- **目标**：任务是将每张图像分类为 10 类之一，表示不同类型的时尚物品。
- **难度**：比 MNIST 稍微复杂一些，因为图像的类别更为多样且图像内容不如数字那么简单。