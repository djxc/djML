# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:53
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : inference.py
"""

"""
from config import UNetConfig
from utils.colors import get_colors
from utils.dataset import BasicDataset
from unet import UNet
from unet import NestedUNet
import cv2
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch
import numpy as np
import argparse
import logging
import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 测试运行
# python inference_color.py -m /djxc/data/Unet_data/checkpoints/09262_rs_epoch_3.pth -i /djxc/rs_detection/change_detection_train/val/ -o /djxc/rs_detection/change_detection_train/val/

cfg = UNetConfig()


def inference_one(net, image, device, modelType):
    net.eval()

    if modelType:
        img = torch.from_numpy(BasicDataset.preprocess(image, cfg.scale))
    else:
        img = torch.from_numpy(image)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        if cfg.deepsupervision:
            output = output[-1]

        if cfg.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)        # C x H x W

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize((image.size[1], image.size[0])),
                transforms.ToTensor()
            ]
        )

        if cfg.n_classes == 1:
            probs = tf(probs.cpu())
            mask = probs.squeeze().cpu().numpy()
            return mask > cfg.out_threshold
        else:
            masks = []
            for prob in probs:
                prob = tf(prob.cpu())
                mask = prob.squeeze().cpu().numpy()
                mask = mask > cfg.out_threshold
                masks.append(mask)
            return masks


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='epoch_11.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', dest='input', type=str, default='',
                        help='Directory of input images')
    parser.add_argument('--output', '-o', dest='output', type=str, default='',
                        help='Directory of ouput images')
    return parser.parse_args()


def predict_img():
    '''预测图像'''
    args = get_args()
    print(args)
    input_imgs = os.listdir(args.input)

    net = eval(cfg.model)(cfg)
    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    # logging.info("Model loaded !")
    print("Model loaded !")

    for i, img_name in tqdm(enumerate(input_imgs)):
        # logging.info("\nPredicting image {} ...".format(img_name))

        print("\nPredicting image {} ...".format(img_name))
        img_path = osp.join(args.input, img_name)
        img = Image.open(img_path)

        mask = inference_one(net=net,
                             image=img,
                             device=device)
        # print(mask)
        img_name_no_ext = osp.splitext(img_name)[0]
        output_img_dir = osp.join(args.output, img_name_no_ext)
        os.makedirs(output_img_dir, exist_ok=True)

        if cfg.n_classes == 1:
            image_idx = Image.fromarray((mask * 255).astype(np.uint8))
            image_idx.save(osp.join(output_img_dir, img_name))
        else:
            # colors = get_colors(n_classes=cfg.n_classes)
            # w, h = img.size
            # img_mask = np.zeros([h, w, 3], np.uint8)
            # for idx in range(0, len(mask)):
            #     image_idx = Image.fromarray((mask[idx] * 255).astype(np.uint8))
            #     array_img = np.asarray(image_idx)
            #     img_mask[np.where(array_img == 255)] = colors[idx]

            # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            # img_mask = cv2.cvtColor(np.asarray(img_mask), cv2.COLOR_RGB2BGR)
            # # cv2.imwrite(osp.join(output_img_dir, img_name + "_predict"), img_mask)
            # output = cv2.addWeighted(img, 0.7, img_mask, 0.3, 0)
            # cv2.imwrite(osp.join(output_img_dir, img_name), output)

            colors = get_colors(n_classes=cfg.n_classes)
            w, h = img.size
            for idx in range(0, len(mask)):
                # img_mask = np.zeros([h, w], np.uint8)
                image_idx = Image.fromarray((mask[idx] * 255).astype(np.uint8))
                array_img = np.asarray(image_idx)
                cv2.imwrite(osp.join(output_img_dir, img_name +
                                     "_" + str(idx)), array_img)

                # img_mask[np.where(array_img == 255)] = colors[idx]

            # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            # img_mask = cv2.cvtColor(np.asarray(img_mask), cv2.COLOR_RGB2BGR)
            # # cv2.imwrite(osp.join(output_img_dir, img_name + "_predict"), img_mask)
            # output = cv2.addWeighted(img, 0.7, img_mask, 0.3, 0)
            # cv2.imwrite(osp.join(output_img_dir, img_name), output)


if __name__ == "__main__":
    args = get_args()
    print(args)
    all_image_name = open(args.input + "/val.txt", "r")
    input_imgs = all_image_name.readlines()
    all_image_name.close()
    # input_imgs = os.listdir(args.input)

    net = eval(cfg.model)(cfg)
    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    print("Model loaded !")
    value_map = [
        11, 12, 13, 14, 15, 16,
        21, 22, 23, 24, 25, 26,
        31, 32, 33, 34, 35, 36,
        41, 42, 43, 44, 45, 46,
        51, 52, 53, 54, 55, 56,
        61, 62, 63, 64, 65, 66,
    ]
    for i, img_name in tqdm(enumerate(input_imgs)):
        # logging.info("\nPredicting image {} ...".format(img_name))
        img_name = img_name.replace("\n", "")
        print("\nPredicting image {} ...".format(img_name))
        img1_path = osp.join(args.input + "/im1/", img_name)
        img2_path = osp.join(args.input + "/im2/", img_name)
        # img = Image.open(img_path)
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img1Bands = cv2.split(img1)
        img2Bands = cv2.split(img2)
        img1Bands.extend(img2Bands)
        newIMG = cv2.merge(img1Bands)
        img = newIMG.transpose((2, 0, 1))
        mask = inference_one(net=net,
                             image=img,
                             device=device,
                             modelType=False)
        # print(mask)
        img_name_no_ext = osp.splitext(img_name)[0]
        output_img_dir = osp.join(args.output, "change")
        os.makedirs(output_img_dir, exist_ok=True)

        if cfg.n_classes == 1:
            image_idx = Image.fromarray((mask * 255).astype(np.uint8))
            image_idx.save(osp.join(output_img_dir, img_name))
        else:
            # colors = get_colors(n_classes=cfg.n_classes)
            # w, h = img.size
            # img_mask = np.zeros([h, w, 3], np.uint8)
            # for idx in range(0, len(mask)):
            #     image_idx = Image.fromarray((mask[idx] * 255).astype(np.uint8))
            #     array_img = np.asarray(image_idx)
            #     img_mask[np.where(array_img == 255)] = colors[idx]

            # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            # img_mask = cv2.cvtColor(np.asarray(img_mask), cv2.COLOR_RGB2BGR)
            # # cv2.imwrite(osp.join(output_img_dir, img_name + "_predict"), img_mask)
            # output = cv2.addWeighted(img, 0.7, img_mask, 0.3, 0)
            # cv2.imwrite(osp.join(output_img_dir, img_name), output)

            colors = get_colors(n_classes=cfg.n_classes)
            # w, h = img.size
            img_mask = np.zeros([512, 512, 3], np.uint8)
            img_mask1 = np.zeros([512, 512], np.uint8)
            img_mask2 = np.zeros([512, 512], np.uint8)
            for idx in range(0, len(mask)):
                predict_mask = Image.fromarray(
                    (mask[idx] * 255).astype(np.uint8))
                array_img = np.asarray(predict_mask)
                # 如果最大值为255则将其保存
                if array_img.max() == 255:
                    # cv2.imwrite(osp.join(output_img_dir, img_name +
                    #  "_" + str(idx)), array_img)
                    # if value_ != 0:
                    #     idx_ = str(idx)
                    # else:
                    #     idx_ = "00"
                    value_ = value_map[idx]
                    idx_ = str(value_)
                    img_mask1[np.where(array_img == 255)] = int(idx_[0])
                    img_mask2[np.where(array_img == 255)] = int(idx_[1])
                    img_mask[np.where(array_img == 255)] = colors[idx]

            img_mask = cv2.cvtColor(np.asarray(img_mask), cv2.COLOR_RGB2BGR)
            cv2.imwrite(osp.join(args.output +
                                 "/prediction/im1/", img_name), img_mask1)
            cv2.imwrite(osp.join(args.output +
                                 "/prediction/im2/", img_name), img_mask2)
            cv2.imwrite(osp.join(output_img_dir,
                                 "predict_" + img_name), img_mask)
