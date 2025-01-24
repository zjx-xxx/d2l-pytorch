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
	`form torch import nn`,`nn`是neural network的缩写，集成了大量的函数，详情见[[torch.nn]]。


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

