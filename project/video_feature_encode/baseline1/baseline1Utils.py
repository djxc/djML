import os
import torch
import random
import numpy as np
import torch.nn as nn

from models import MLP, MLPDJ

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

        
def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_args(logger, args):
    opt = vars(args)
    logger.cprint('------------ Options -------------')
    for k, v in sorted(opt.items()):
        logger.cprint('%s: %s' % (str(k), str(v)))
    logger.cprint('-------------- End ----------------\n')


def init_logger(log_dir, args):
    mkdir(log_dir)
    log_file = os.path.join(log_dir, 'log_%s.txt' % args.method)
    logger = IOStream(log_file)   
    print_args(logger, args)
    return logger


def set_gpu(gpu):
    gpu_list = [int(x) for x in gpu.split(',')]
    print('use gpu:', gpu_list)
    return gpu_list.__len__()


def str2loss(args):

    if args.loss == "Cross_Entropy":
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("Loss \"" + args.loss + "\" not yet implemented")
    

def str2model(args):
    if args.model == "MLP":
        return MLP(args)
    if args.model == "DJMLP":
        return MLPDJ(args)
    else:
        raise NotImplementedError("Model \"" + args.model + "\" not yet implemented")
    

def noise_injection(features, noise_factor):
    """噪声注入"""
    noise = torch.randn_like(features) * noise_factor
    return features + noise