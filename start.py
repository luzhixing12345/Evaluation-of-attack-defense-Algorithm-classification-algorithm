
from numpy import show_config
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
from configure import configuration
from dataset import get_dataset
from network import get_model_pytorch,get_model_tf,train,test
from attack import start_attack
from defense import start_defense
from evaluation import evaluate_train_loss,evaluate_accurency,show_pic
import torch.optim as optim
import torch

def setup(args):
    cfg=configuration(args.dataset,args.network,args.attack,args.defense)
    return cfg



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="classifer_network_argument")
    #关于命令行传参
    #https://docs.python.org/zh-cn/3/library/argparse.html
    
    parser.add_argument('--dataset', default='None')
    parser.add_argument('--network', default='None')
    parser.add_argument('--attack', default='None')
    parser.add_argument('--defense', default='None')
    args = parser.parse_args()
    print(f'Using the network of {args.network}')
    print(f'Using the dataset of {args.dataset}')
    print(f'Using the attack method of {args.attack}')
    print(f'Using the defense method of {args.defense}')

    cfg=setup(args)
    train_data,test_data=get_dataset(cfg)

    # examples = enumerate(test_data)
    # batch_idx, (example_data, example_targets) = next(examples)
    
    # show_pic(example_data, example_targets)       #这三行代码用于显示你所调用的数据集的测试集部分图片


    model_cuda = get_model_pytorch(cfg)#使用pytorch的cuda
    optimizer = optim.SGD(model_cuda._inner.parameters(), lr=cfg.learning_rate,
                      momentum=cfg.momentum)

    
    #优化器参数：
    #https://blog.csdn.net/willduan1/article/details/78070086
    #这里可以修改
    
    loss_fn=cfg.loss_fn


    train_losses = []
    train_counter = []
    test_accurency =[]
    test_counter = [i for i in range(1,cfg.n_epochs + 1)]

    

    for epoch in range(1, cfg.n_epochs+1):
        train(model_cuda._inner,train_data,optimizer,loss_fn,epoch,train_losses,train_counter)
        test(model_cuda._inner,test_data,loss_fn,test_accurency)

    evaluate_train_loss(train_losses,train_counter)
    evaluate_accurency(test_accurency,test_counter)

    torch.save(model_cuda._inner.state_dict(),'model_trained.pth')#保存模型
    
    model = get_model_tf(cfg)#不使用cuda，因为攻击防御用到了tensorflow,与cuda不兼容
    model._inner.load_state_dict(torch.load('model_trained.pth'))#导入模型


    with tf.compat.v1.Session() as sess:
        adv_dataset= start_attack(model,cfg,loss_fn,sess,test_data)#attack
    start_defense(model,cfg,loss_fn,adv_dataset) #defense


    print('You have done all the job !')