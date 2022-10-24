# 树叶分类比赛https://www.kaggle.com/c/classify-leaves
- 1、共有176类，训练集共有18353张图片，每个类至少有50张图片

# 解决方案
## 数据处理
- 1、首先将数据划分为训练集与验证集，这里按照7：3进行划分。
- 2、查看数据每个类分布情况，这里生成了category.json文件，为每一类的图片数量，最少的数据为51个，最多的为353.类别分布不均匀
- 3、将每个类创建一个文件夹，将属于该类数据的放在该文件夹下


## resnet

## 数据增强
该任务为分类任务，需要根据任务进行特定的数据增强方法。
- 1、随机缩放裁剪，水平翻转
- 2、随机擦除、mixup
- 3、安装albumentations库，相比较torchvision，该库处理图像速度更快些。pip install albumentations -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
- 4、安装timm库，pip install timm -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com。timm库包括众多模型，可以直接使用。


## 训练方法
- 1、k折交叉验证,将数据分为k组，每组数据重新划分训练测试，充分利用数据。每组按照正常进行训练，每一组数据训练出一个模型，在验证阶段，将多组训练出来的模型计算出结果，然后将多个模型结果取均值。
- 2、batch_size如果设置过大，相应需要调整学习率，大batch_size会使学习速度下降，但可以提高模型的泛化性



## 训练记录
- 1、使用非预训练的resnet50，训练80轮，测试精度0.875
- 2、采用相同的数据集，用预训练的resnet50进行训练，修改最后的全连接层输出为当前分类的个数。学习率设置较小0.0003，学习率调整采用余玄退火方式CosineAnnealingLR。batch_size设置为16.数据增强增加了水平与垂直翻转。训练400轮，验证集上精度在0.945左右。每个epoch需要70s。
- 3、采用相同数据集，用预训练的resnext50进行训练。学习率设为0.001，batch_size为24，数据增强水平与垂直翻转。训练300轮验证集精度在0.951左右。每个epoch需要100s。
- 4、相同数据集，在1的基础上增加数据增强，学习率为0.01，batch_size为12，数据增强为水平、垂直翻转，旋转、mixup及cutmix等。每个epoch需要30s。训练300轮最高精度在0.95，但模型不稳定。为了模型稳定性，减少数据增强，仅保留mixup；增加batch_size到24
- 3、训练多个模型，
