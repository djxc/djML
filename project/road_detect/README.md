# 道路提取

- 1、构建深度学习提取道路轮廓，unet
- 2、通过形态学处理，优化道路
- 3、数据处理，首先将数据拆分为训练集和验证集，这里比例为8：2，生成5个文件记录5折交叉验证的训练集和验证集数据路径
- 4、优化方向：损失函数、数据增强、网络结构调整、误差较大数据类别分析
- 5、数据增强：图像旋转
