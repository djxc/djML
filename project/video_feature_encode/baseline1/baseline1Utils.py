import os
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from models import MLP, MLPDJ, MLPlus

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
    if args.model == "MLPlus":
        return MLPlus(args)
    if args.model == "DJMLP":
        return MLPDJ(args)
    else:
        raise NotImplementedError("Model \"" + args.model + "\" not yet implemented")
    

def noise_injection(features, noise_factor):
    """噪声注入"""
    noise = torch.randn_like(features) * noise_factor
    noise = torch.where(noise > 0.1, 0.1, noise)
    noise = torch.where(noise < -0.1, -0.1, noise)
    return features + noise


def draw_acc_line(log_path: str):
   
    with open(log_path) as log_file:
        log_lines = log_file.readlines()
        train_acc = []
        valify_acc = []
        train_loss = []
        valify_loss = []
        epoch_list = []
        epoch_num = 1
        for log_line in log_lines:
            if log_line.startswith("Training results for epoch"):
                train_str_list = log_line.split(":")
                train_acc.append(float(train_str_list[-1]))
                train_loss.append(float(train_str_list[2].split(";")[0]))
                epoch_list.append(epoch_num)            
                epoch_num = epoch_num + 1
            if log_line.startswith("accuracy"):
                acc_str_list = log_line.split(" ")
                valify_acc.append(float(acc_str_list[2].replace(";", "")))
                valify_loss.append(float(acc_str_list[-1]))

        if len(valify_acc) < len(epoch_list):
            for i in range(len(epoch_list)- len(valify_acc)):
                valify_acc.append(valify_acc[-1])
                valify_loss.append(valify_loss[-1])
        print(len(epoch_list), len(train_acc), len(valify_acc))
        return epoch_list, train_acc, valify_acc, train_loss, valify_loss
        
def show_lr_line(log_path: str):
    with open(log_path) as log_file:
        log_lines = log_file.readlines()
        lr_list = []
        epoch_list = []
        epoch_num = 1
        for log_line in log_lines:
            if log_line.startswith("----------Start Training Epoch"):
                train_str_list = log_line.split("[")[-1].split("]")[0].split("/")
                lr_list.append(float(train_str_list[-1]))
                epoch_list.append(epoch_num)            
                epoch_num = epoch_num + 1
        fig = plt.figure(figsize=None, dpi=None)        
        plt.plot(epoch_list, lr_list, color="red", label="lr")
        fig.legend()
        plt.title("lr")
        plt.show()
        return epoch_list, lr_list
        
def show_model_result(show_type="acc"):
    # log_file1 = r"E:\Data\MLData\videoFeature\video_results\first_stage\baseline\MLP\bs32_ep100_lr0.0001_step20_gm0.5\06_13_13_13\log_baseline.txt"
    # log_file1 = r"E:\Data\MLData\videoFeature\video_results\first_stage\baseline\MLPlus\bs32_ep100_lr0.0001_step20_gm0.5\06_14_09_06\log_baseline.txt"
    log_file1 = r"E:\Data\MLData\videoFeature\video_results\first_stage\baseline\MLPlus\bs32_ep200_lr0.0001_step20_gm0.5\06_14_11_08\log_baseline.txt"
    log_file2 = r"E:\Data\MLData\videoFeature\video_results\first_stage\baseline\MLPlus\bs32_ep500_lr0.0001_step20_gm0.5\06_14_14_51\log_baseline.txt"

    epoch_list1, train_acc1, valify_acc1, train_loss1, valify_loss1 = draw_acc_line(log_file1)
    epoch_list2, train_acc2, valify_acc2, train_loss2, valify_loss2 = draw_acc_line(log_file2)    
    fig = plt.figure(figsize=None, dpi=None)
    if show_type == "acc":

        plt.plot(epoch_list1, train_acc1, color="red", label="train1_acc")
        plt.plot(epoch_list1, valify_acc1, color="orange", label="valify1_acc")
        plt.plot(epoch_list2, train_acc2, color="blue", label="train2_acc")
        plt.plot(epoch_list2, valify_acc2, color="green", label="valify2_acc")
    else:
        plt.plot(epoch_list1, train_loss1, color="red", label="train1_loss")
        plt.plot(epoch_list1, valify_loss1, color="orange", label="valify1_loss")
        plt.plot(epoch_list2, train_loss2, color="blue", label="train2_loss")
        plt.plot(epoch_list2, valify_loss2, color="green", label="valify2_loss")
    fig.legend()
    plt.title(show_type)
    plt.show()


def statics_max_min_by_class():
    """统计每类数据最大最小值"""
    import json
    file_path = r"E:\Data\MLData\videoFeature\train\train_list.txt"
    data_folder = r"E:\Data\MLData\videoFeature\train\train_feature"
    statics_map = {}
    with open(file_path) as file_data:
        data_list = json.loads(file_data.read())
        for data in data_list:
            data_path = os.path.join(data_folder, data)
            cls_name = data_list[data]
            data_np = np.load(data_path)
            data_np = np.squeeze(data_np, -1)
            data_np = np.squeeze(data_np, -1)
            max = np.max(data_np)
            min = np.min(data_np)
            if cls_name not in statics_map:
                statics_map[cls_name] = {"max": 0, "min": 0}
            if max > statics_map[cls_name]["max"]:
                statics_map[cls_name]["max"] = max
            if min < statics_map[cls_name]["min"]:
                statics_map[cls_name]["min"] = min
    print(statics_map)


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # show_lr_line(r"E:\Data\MLData\videoFeature\video_results\first_stage\baseline\MLP\bs32_ep100_lr0.0001_step20_gm0.5\06_13_10_55\log_baseline.txt")
    show_model_result("acc")
    # statics_max_min_by_class()
