# 手写神经网络测评分类问题的攻击算法、防御算法



## Introduction

该工作是基于老师的要求，希望笔者熟悉如何编写一个大的工程的整体框架而做的。本项目代码基于pytorch设计并实现了选择数据集，选择神经网络，选择攻击算法，选择防御算法来完成整体的一套测评计算测评流程
> 该任务仅当练手,完成之初刚刚入门神经网络,仅供参考



## Requirements

- PyTorch ≥ 1.6
- torchvision that matches the Pytorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
- tensorflow 2.1.0
- numpy 1.21.0 `pip install numpy`
- matplotlib 3.3.4 `pip install matplotlib`
- openpyxl `pip install openpyxl`

## 设计思路
- start.py文件作为整体调用文件

- 从python命令行中获取信息，需要获取的信息{dataset,network,attack,defense}

- 设计configure.py文件作为配置文件，保存并且定义一些基础属性，保存在cfg中
  
- 设计dataset.py文件处理得到数据集，得到训练数据和测试数据
  
- 设计network.py定义神经网络，笔者设计了两种简单网络，用于后续实验
  
- 进行train和test并且保存数据并且绘制图像，保存模型
  
- import [ares](https://github.com/thu-ml/ares)   （算法库，本项目直接将其下载并导入，不需额外import）

<!--ares中的tensorflow采用的是1.15.4版本，但笔者已经预先下载了tensorflow 2.1.0版本了，故在ares的每一个文件的tf引用采取了修改，使其1、2兼容-->

- 设计attack.py调用攻击算法，测评，并且保存得到的tensor，传给defense使用

- 设计defense.py调用防御算法，测评

- 设计了utils.py 完成所有测评函数，并且将记录保存至excel记录下来




## 使用方法

```python
python start.py --dataset MNIST --network Net --attack FGSM --defense BDR
```
> 注：
> 
> 数据集的选择只能是pytorch中自带的，不同数据集在调用时传入的参数不同，详细细节请阅读代码dataset.py，网络的模型只有笔者写的两 个，Net/NeuralNetwork
> 
> 攻击算法只引用了ares.attack 中的三个算法 Deepfool、 FGSM 、MIM
> 
> 防御算法引用了ares.defense的两个BDR、JPEG
> 
> 使用时可自行替换
> 
> 更多细节请阅读代码文档，内附详细注释



## 记录格式

笔者测试了`MNIST/Fashion_MNIST`两个数据集在`Net/NeuralNetwork`两个网络上的训练效果,
并且将训练和测试中的训练损失和准确率都绘制成图像，保存在`./train&test`文件夹下，每组两种图片，一共八张,
同时针对四种组合分别调用攻击算法`MIM/FGSM/deepfool`,防御算法`BDR/JPEG`,并且自己设计了一种表格用于记录，详细信息可以参考代码以及`./result.xlsx`文件中的数据格式.同时考虑到可能会对攻击函数以及防御函数调参，笔者也在每一次实验中记录了攻击函数和防御函数的参数信息，并且一同保存记录到了excel对应位置下<br>
注意：如果重复上次实验，会覆盖掉原表格中位置的信息，但不会覆盖实验记录信息<br>
细节请查阅[代码](utils.py)



## 说明：

笔者给出了自己测试的实验结果，包含`train&test`文件夹下的图片以及`result.xlsx`文件，如果希望重复实验可以删除掉`train&test`文件夹下所有图片（不要删除文件夹），以及`result.xlsx`文件重复实验

本项目仅供参考，如有不妥之处还望指正