

'''
配置文件，用于记录部分参数
以及对应各种输入输出进行对应的数据集、网络、攻击算法、防御算法的切换

'''


import torch.nn as nn

class configuration(object):

    def __init__(self,dataset,network,attack,defense) -> None:
        super().__init__()
        self.batch_size_train = 64
        self.batch_size_test = 1000
        self.momentum=0.5
        self.learning_rate=1e-3
        self.loss_fn = nn.CrossEntropyLoss()
        #loss_fn = F.nll_loss()

        self.n_epochs=10


        self.dataset=dataset
        self.network=network
        self.attack_method=attack
        self.defense_method=defense

        self.channel  = 1
        self.height   = 28
        self.width    = 28

        self.attack_arguments={}
        self.defense_arguments = {} 
        
        #代码中实现了将攻击防御的参数传出来并且记录保存的功能
        #用于完成仅对于攻击、防御算法内置参数调参测试结果的功能
        #详细用法请阅读README.md中的教程以及utils.py代码