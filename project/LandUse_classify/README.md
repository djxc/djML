# 构建土地利用分类算法
- 1、利用UCMerced_LandUse数据集，包含21类，每类具有100个数据。图像256*256大小，分辨率在0.3米
- 2、对数据集进行训练集与测试集的划分，7：3
- 3、首先用预训练的resnet50进行训练，学习率=0.01, 验证集精度在0.985左右