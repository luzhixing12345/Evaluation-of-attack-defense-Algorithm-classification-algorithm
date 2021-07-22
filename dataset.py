
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets


#关于pytorch数据集的介绍
#https://blog.csdn.net/godblesstao/article/details/110280317
#https://pytorch.org/vision/stable/datasets.html#
#手写字符识别数据集
Hand_written_reg_datasets={
    'EMNIST':datasets.EMNIST,
    'MNIST':datasets.MNIST,
    'QMNIST':datasets.QMNIST,
    'USPS':datasets.USPS,
    'SVHN':datasets.SVHN,
    'KMNIST':datasets.KMNIST,
    'Omniglot':datasets.Omniglot
    }


#实物分类
Physical_Classification_dataset = {
    'Fashion_MNIST':datasets.FashionMNIST,
    'CIFAR10':datasets.CIFAR10,
    'CIFAR100':datasets.CIFAR100,
    'LSUN':datasets.LSUN,
    'SLT-10':datasets.STL10,
    'ImageNet':datasets.ImageNet
    }

Dataset={}
Dataset.update(Hand_written_reg_datasets)
Dataset.update(Physical_Classification_dataset)


def get_dataset(cfg):  
    
    '''
    arguments:cfg代表configure 配置文件，其中包含的信息可以从./configure.py中查找到

    return  :train_dataloader,test_dataloader分别是训练集和测试集的tensor形式
    '''
    # Download test data from open datasets.
    # 这一步运行过程中可能出现RuntimeError,这是由于网络波动问题，重复操作即可
    # 该函数会优先扫描root路径下是否存在数据集，如果已经下载过数据集则会自动跳过，不需担心重复下载
    # root路径为下载路径，你也可以选择对应的文件夹位置下载
    # 不同数据集需要调用的接口可能不同，请阅读pytorch文档进行修改
    training_data = Dataset[cfg.dataset](
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    # 不同数据集需要调用的接口可能不同，请阅读pytorch文档进行修改
    test_data = Dataset[cfg.dataset](
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    train_dataloader = DataLoader(training_data, batch_size=cfg.batch_size_train)
    test_dataloader = DataLoader(test_data, batch_size=cfg.batch_size_test)     


    examples = enumerate(test_data)
    batch_idx, (example_data, example_targets) = next(examples)

    cfg.channel = list(example_data.shape)[0]
    cfg.height  = list(example_data.shape)[1]
    cfg.width   = list(example_data.shape)[2]
    
    '''
    将数据张量以batch_size的大小进行划分
    关于batch_size：
    https://blog.csdn.net/qq_34886403/article/details/82558399/
    '''
    return train_dataloader,test_dataloader

