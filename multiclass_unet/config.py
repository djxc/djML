# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:54
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : config.py
"""

"""
import os

rootPath = "/djxc"  # "/document/2020" # 
myPath = rootPath + "/data/Unet_data"


class UNetConfig:
    def __init__(self,
                 epochs=100,  # Number of epochs
                 batch_size=1,    # Batch size
                 # Percent of the data that is used as validation (0-100)
                 validation=10.0,
                 out_threshold=0.5,

                 optimizer='SGD',
                 lr=0.0005,     # learning rate
                 lr_decay_milestones=[20, 50],
                 lr_decay_gamma=0.9,
                 weight_decay=1e-8,
                 momentum=0.9,
                 nesterov=True,

                 n_channels=6,  # Number of channels in input images
                 n_classes=36,  # Number of classes in the segmentation
                 scale=1,    # Downscaling factor of the images

                 load= myPath + "/checkpoints/09262_rs_epoch_3.pth",   # Load model from a .pth file
                 save_cp=True,

                 model='NestedUNet',
                 bilinear=True,
                 deepsupervision=True,

                 train_file=rootPath + "/rs_detection/change_detection_train/train/train.txt",
                 val_file=rootPath + "/rs_detection/change_detection_train/train/val.txt",

                 ):
        super(UNetConfig, self).__init__()

        # self.images_dir = myPath + '/images'
        # self.masks_dir = myPath + '/masks'

        self.images_dir = rootPath + "/rs_detection/change_detection_train/train/"
        self.masks_dir = rootPath + "/rs_detection/change_detection_train/train/"
        self.checkpoints_dir = myPath + '/checkpoints'

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation = validation
        self.out_threshold = out_threshold

        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay_milestones = lr_decay_milestones
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale = scale

        self.load = load
        self.save_cp = save_cp

        self.model = model
        self.bilinear = bilinear
        self.deepsupervision = deepsupervision

        self.train_file = train_file
        self.val_file = val_file

        os.makedirs(self.checkpoints_dir, exist_ok=True)
