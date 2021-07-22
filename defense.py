

from PIL.Image import new
from ares.defense.bit_depth_reduction import bit_depth_reduce
from ares.defense.jpeg_compression import jpeg_compress
import torch
from network import test_def
from utils import change_data_size,rechange_data_size
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#注释同attack.py,change_to_defense
def start_defense(model,cfg,loss_fn,test_set):
    defense=find_defense_method(model,cfg)
    def_test_data = []
    for data,target in test_set:
        new_data = change_data_size(cfg,data)#修改pytorch的tensor形状适应tensorflow
        def_data = defense(new_data)         #返回tensorflow的tensor
        with tf.compat.v1.Session() as sess:
            np_data = def_data.eval()        #得到numpy
        def_data = torch.from_numpy(np_data)#得到pytorch的tensor
        def_data = rechange_data_size(cfg,def_data)#修改tensor形状以适应pytorch
        def_test_data.append([def_data,target])
        print('Finished one batch of defense !')
    test_def(model._inner,def_test_data,loss_fn,cfg,tf= True)



def find_defense_method(model,cfg):
    return defense_set[cfg.defense_method]

def DEF_BDR(new_data):
    return bit_depth_reduce(new_data, x_min=0.0, x_max=1.0, step_num=5, alpha=1e6)

def DEF_JPEG(new_data):
    return jpeg_compress(new_data, x_min=0.0, x_max=1.0,quality=95)

defense_set = {'BDR':DEF_BDR,'JPEG':DEF_JPEG}

    