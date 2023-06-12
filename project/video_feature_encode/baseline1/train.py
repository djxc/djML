import os
import os.path
import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
from copy import deepcopy
import argparse
import time

from baseline1Utils import init_logger, str2model, str2loss, set_gpu, set_seed, IOStream
from baselineData import VideoDataset1, random_remove_frame, split_frame

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_decay = 0.0
workspace = r"D:\\Data\\MLData\\videoFeature"
def train(args):
    args.device = device
    logger = init_logger(args.log_dir, args)
    
    train_dataset = VideoDataset1(args.train_label_path, train="train", transform=True)
    val_dataset = VideoDataset1(args.varify_label_path, train="verify")  
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=8, shuffle=False, pin_memory=True)

    model = str2model(args)
    model = nn.DataParallel(model, device_ids=list(range(args.num_gpu)))
    model = model.to(device)
    loss_fn = str2loss(args)
    loss_fn.to(device)

    start_epoch = 1
    if args.resume > 0:
        logger.cprint(f'Resume previous training, start from epoch {args.resume}, loading previous model')
        start_epoch = args.resume
        resume_model_path = os.path.join(args.checkpoint_path, f'best_model.pth')

        if os.path.exists(resume_model_path):
            model.load_state_dict(torch.load(resume_model_path)['model'])
            for name, param in model.named_parameters():
                if "layer" in name:
                    param.requires_grad = False
                if 'classifer' in name:
                    param.requires_grad = False
            print(model.module.classifer[0].weight)

        else:
            raise RuntimeError(f'{resume_model_path} does not exist, loading failed')


    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, weight_decay=args.weight_decay)   
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00001, last_epoch=-1)                        
                    

    # Start Traning process--------------------------------------
    model.train()
    best_epoch = 0
    best_acc = 0. 
    sep = 1e-6

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.cprint(f'----------Start Training Epoch-[{epoch}/{args.epochs}/{current_lr:.6f}]------------')
        ttl_rec = 0.
        total_correct = 0
        for i, (inputs, targets) in enumerate(train_dataloader):
            # inputs = random_remove_frame(inputs)
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs) 
            predicted = torch.argmax(logits, dim=-1)
            total_correct += (predicted == targets).sum().item()
            loss = loss_fn(logits, targets)

            loss.backward()
            optimizer.step()

            batch_rec = loss
            ttl_rec += batch_rec
        epoch_rec = ttl_rec / len(train_dataloader)
        epoch_acc = total_correct / (len(train_dataloader) * args.batch_size)
        logging = f'Training results for epoch -- {epoch}: Epoch_Rec:{epoch_rec}; Epoch_acc: {epoch_acc}'
        logger.cprint(logging)
        scheduler.step()
       
        accuracy = verify(model, epoch, val_dataloader, logger, loss_fn)
        logger.cprint(f'accuracy = {accuracy}')  
        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch
            save_model(model, epoch, best_acc)
            logger.cprint('******************Model Saved******************')
        if epoch % 20 == 0:
            save_model(model, epoch, best_acc, "model_{}.pth".format(epoch))
        logger.cprint(f'best_epoch = {best_epoch}, best_acc = {best_acc}')

def save_model(model, epoch, acc, model_name='best_model.pth'):
        save_dict = {
            'model': deepcopy(model.state_dict()),
            'best_epoch': epoch,
            'best_acc': acc
        }
        torch.save(save_dict, os.path.join(args.log_dir, model_name))

def verify(model: nn.DataParallel, epoch: int, val_dataloader: DataLoader, logger, loss_fn):
    model.eval()
    total_correct = 0
    total_samples = 0
    verify_result = {
        "0": {"total": 0, "error": 0, "error_info": {}},
        "1": {"total": 0, "error": 0, "error_info": {}},
        "2": {"total": 0, "error": 0, "error_info": {}},
        "3": {"total": 0, "error": 0, "error_info": {}},
        "4": {"total": 0, "error": 0, "error_info": {}}
        }
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in tqdm(val_dataloader, desc='Epoch:{:d} val'.format(epoch)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            total_loss = total_loss + loss
            predicted = torch.argmax(logits, dim=-1)
            total_correct += (predicted == targets).sum().item()
            total_samples += inputs.size(0)
            result = torch.eq(predicted, targets)
            for j, r in enumerate(result):
                    y = str(targets[j].cpu().item())
                    verify_result[y]["total"] = verify_result[y]["total"] + 1
                    if not r:
                        verify_result[y]["error"] = verify_result[y]["error"] + 1
                        y_pre = str(predicted[j].cpu().item())
                        if y_pre not in verify_result[y]["error_info"]:
                            verify_result[y]["error_info"][y_pre] = 1
                        else:
                            verify_result[y]["error_info"][y_pre] = verify_result[y]["error_info"][y_pre] + 1
        loss_rec = total_loss / len(val_dataloader)
    for cls in verify_result:
            total_num = verify_result[cls]["total"]
            true_num = total_num - verify_result[cls]["error"]
            error_info_str = json.dumps(verify_result[cls]["error_info"])
            class_result = "cls {0:s} acc is {1:1.3f}, total: {2:d}, error: {3:d}".format(cls, 
                    true_num/total_num, 
                    total_num, verify_result[cls]["error"])
            class_result = "{}, error_info: {}".format(class_result, error_info_str)
            logger.cprint(class_result)
    # 计算准确率
    accuracy = total_correct / total_samples    
    model.train()   
    return accuracy


def test(args):
    """预测"""
    result_root = r"D:\Data\MLData\videoFeature\video_results"
    model = str2model(args)
    model = model.to(device)
    test_dataset = VideoDataset1(args.to_be_predicted, train="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    model.eval()
    model_dict = torch.load(os.path.join(result_root, 'best_model.pth'))["model"]
    new_model_dict = dict()
    for k in model_dict:
        new_k = k.replace("module.", "")
        new_model_dict[new_k] = model_dict[k]
    model.load_state_dict(new_model_dict)
    predicted_result = []
    with torch.no_grad(): 
        for inputs , _ in tqdm(test_dataloader, desc='test'):
            inputs = inputs.to(device)
            logits = model(inputs)
            predicted = torch.argmax(logits, dim=-1)

            predicted_list = predicted.tolist()
            predicted_result.extend(predicted_list)
    
    with open(os.path.join(result_root, 'submit_example.txt'), 'r') as f:
        example = eval(f.read())
    
    temp_idx = 0
    for key, _ in example.items():
        example[key] = predicted_result[temp_idx]
        temp_idx+=1

    with open(f'{result_root}/example.txt', 'w', encoding='utf-8') as f:
        json.dump(example, f)

def test1(args):
    """预测"""
    result_root = r"D:\Data\MLData\videoFeature\video_results"
    model = str2model(args)
    model = model.to(device)
    test_dataset = VideoDataset1(args.to_be_predicted, train="test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True)

    model.eval()
    model_dict = torch.load(os.path.join(result_root, 'best_model.pth'))["model"]
    new_model_dict = dict()
    for k in model_dict:
        new_k = k.replace("module.", "")
        new_model_dict[new_k] = model_dict[k]
    model.load_state_dict(new_model_dict)
    predicted_result = []
    result = {}
    with torch.no_grad(): 
        for inputs , labels in tqdm(test_dataloader, desc='test'):
            inputs = split_frame(inputs)
            inputs = inputs.to(device)
            logits = model(inputs)
            predicted = torch.argmax(logits, dim=-1)

            predicted_list = predicted.tolist()
            predicted_result.extend(predicted_list)
            image_name = Path(labels[0]).name
            counts = np.bincount(predicted_list)
            cls = np.argmax(counts)
            result[image_name] = str(cls)
            if np.max(counts) == 2:
                print(predicted_list, counts)
    print(result)
    with open(os.path.join(result_root, 'result.txt'), 'w') as f:
        f.write(json.dumps(result))


if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='first_stage') 
    parser.add_argument('--train_label_path', type=str, required=False, default='{}\\train\\train_4.csv'.format(workspace),
                            help='Path to the label file ;')
    parser.add_argument('--varify_label_path', type=str, required=False, default='{}\\train\\verify_4.csv'.format(workspace),
                            help='Path to the label file ;')
    parser.add_argument('--to_be_predicted', type=str, required=False, default='{}\\test_A\\test.csv'.format(workspace),
                            help='Path to the numpy data to_be_predicted ;')


    parser.add_argument('--method', type=str, default='baseline')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size 1024 for yelp, 256 for amazon.')

    parser.add_argument('--lr', type=float, default=0.0001,
                            help='Initial learning rate. [0.1 for amazon and 0.001 for yelp]')
    parser.add_argument('--workers', type=int, default=-1,
                            help='number of workers')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
    parser.add_argument('--resume', type=int, default=-1,help='Flag to resume training [default: -1];')
    parser.add_argument('--action', type=str, default='train', help='Flag to resume training [default: False];')


    # lr_scheduler StepLR 
    parser.add_argument('--step_size', type=float, default=20, help='Decay learning rate every step_size epoches [default: 50];')
    parser.add_argument('--gamma', type=float, default=0.5, help='lr decay')
    # optimizer
    parser.add_argument('--SGD', action='store_true', help='Flag to use SGD optimizer')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--gpu', type=str, default='2', help='gpu id')

    # model pth
    parser.add_argument('--save_path', type=str, default='{}\\video_results'.format(workspace), help='Directory to the save log and checkpoints')
    parser.add_argument('--extra_info', type=str, default='', help='Extra information in save_path')

    # model 
    parser.add_argument('--model', type=str, default='DJMLP',help='Name of model to use')
    # loss
    parser.add_argument('--loss', type=str, default='Cross_Entropy',help='Name of loss to use')

    parser.add_argument('--num_classes', type=int, default=5, help='The number of class')
    parser.add_argument('--n_input', type=int, default=250, help='The number of the input feature')
    parser.add_argument('--d_input', type=int, default=2048, help='The dimension of the input feature')
    # checkpoint_path
    parser.add_argument('--checkpoint_path', type=int, default=2048, help='The dimension of the input feature')


    # args = parser.parse_args()
    args = parser.parse_known_args()[0]
    set_seed(args.seed)
    args.num_gpu = set_gpu(args.gpu)
    args.log_dir = args.save_path + f"/{args.dataset}/{args.method}/{args.model}/bs{args.batch_size}_ep{args.epochs}_lr{args.lr:.4f}_step{args.step_size}_gm{args.gamma:.1f}"

    timestamp = time.strftime('%m_%d_%H_%M')
    if args.extra_info:
        args.log_dir = args.log_dir + '/' + args.extra_info + '_' + timestamp
    else:
        args.log_dir = args.log_dir + '/' + timestamp
    
    if args.action == "train":
        train(args)
    else:
        test(args)