# 道路提取

- 1、构建深度学习提取道路轮廓，unet
- 2、通过形态学处理，优化道路
- 3、数据处理，首先将数据拆分为训练集和验证集，这里比例为8：2，生成5个文件记录5折交叉验证的训练集和验证集数据路径
- 4、优化方向：损失函数、数据增强、网络结构调整、误差较大数据类别分析
- 5、数据增强：图像旋转
- 6、可以选择使用外部数据集：
  - 6.1 Massachusetts Roads Dataset（train_size 1108, 尺寸1500 * 1500）
  - 6.2 DeepGlobe Road Extraction Dataset
  - 6.3 CHN6-CUG Road Dataset
  - 6.4 Aerial Image Segmentation Dataset
  - 6.5 Spacenet
  - 6.6 [遥感AI数据集](https://blog.csdn.net/nominior/article/details/105247990)
- 7、损失函数
  - 7.1 结合二元交叉熵（BCE）损失函数与 Dice 损失函数
- 8、评价指标
  - 用准确率（Rprecision）、召回率（Rrecall）、F1 分数（SF1）和交并比（RIoU）
- 9、消融实验
  消融实验评价不同的改进对结果的影响。
