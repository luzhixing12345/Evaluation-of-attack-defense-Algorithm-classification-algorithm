
from ares import attack
from ares.attack.mim import MIM
from ares.attack.fgsm import FGSM
from ares.attack.deepfool import DeepFool
import torch

from torch.utils.data import Dataset,DataLoader,TensorDataset
from network import test_adv ,get_model_pytorch
from evaluation import evaluate_accurency
from ares.loss.cross_entropy import CrossEntropyLoss
from utils import change_data_size,rechange_data_size

device = "cuda" if torch.cuda.is_available() else "cpu"


#请参考https://thu-ml-ares.readthedocs.io/en/latest/api/ares.attack.html

#关于攻击算法的内置参数请查阅攻击函数定义，有默认推荐的输入

def start_attack(model,cfg,loss_fn,session,test_data):
    attack=find_attack_method(model,cfg,session)
    adv_test_data = []
    for batch_idx, (data, target) in enumerate(test_data):
        adv_data = attack.batch_attack(data, ys=target)
        adv_test_data.append([torch.from_numpy(adv_data),target])
        print('Finshed one batch of attack !')
    adv_result = test_adv(model._inner,adv_test_data,loss_fn,cfg,tf= True)
    return adv_test_data,adv_result


def find_attack_method(model,cfg,session):
    return attack_set[cfg.attack_method](model,cfg.batch_size_test,session,cfg)

def atk_MIM(model,batch_size,session,cfg):
    loss = CrossEntropyLoss(model)
    attack = MIM(
        model=model,
        batch_size=batch_size,
        loss=loss,
        goal='ut',
        distance_metric='l_inf',
        session=session
    )
    config = {        'iteration':10,
                      'decay_factor':1.0,
                      'magnitude':8.0 / 255.0,
                      'alpha':1.0 / 255.0}
    attack.config(iteration=10,decay_factor=1.0,magnitude=8.0 / 255.0,alpha=1.0 / 255.0)
    cfg.attack_arguments = config
    return attack
def atk_FGSM(model,batch_size,session,cfg):
    loss = CrossEntropyLoss(model)
    attack = FGSM(
        model=model,
        batch_size=batch_size,
        loss=loss,
        goal='ut',
        distance_metric='l_inf',
        session=session,
    )
    config = {        'iteration':10,
                      'decay_factor':1.0,
                      'magnitude':8.0 / 255.0,
                      'alpha':1.0 / 255.0}
    attack.config(iteration=10,decay_factor=1.0,magnitude=8.0 / 255.0,alpha=1.0 / 255.0)
    cfg.attack_arguments = config
    return attack


def atk_DF(model,batch_size,session,cfg):
    attack = DeepFool(
        model=model,
        batch_size=batch_size,
        distance_metric='l_inf',
        session=session,
    )
    config = {
        'iteration':10,
        'overshot':0.02
    }
    attack.config(iteration=10,overshot=0.02)
    cfg.attack_arguments = config
    return attack


attack_set = {'MIM':atk_MIM,'FGSM':atk_FGSM,'deepfool':atk_DF,}