

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
