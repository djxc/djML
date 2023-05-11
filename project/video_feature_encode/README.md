# 视觉特征编码
数据维交通场景视频的视觉特征向量以及事件标签。需要学习视觉特征向量与事件标签之间的对应关系，并对视觉特征向量进行压缩重建。900段交通场景视频中提取的900段特征流数据以及对应的事件ID标签。每个特征文件对应一段特征流数据，包含250个特征向量。标注文件将由文本文件提供。文本文件每一行提供一个标注，标注格式为：文件名 ID。视觉特征数据值在0-10之内的浮点数据。数据集中的视觉特征向量应该是用其他网络模型提取的特征数据。
- 1、首先分析下数据分布  
对标签分析下共包含多少类别，每个类别的占比。{'1': 200, '2': 200, '3': 200, '4': 200, '0': 100}.可以看到0类数据较少，类别不均匀。每个数据为250*2408*1*1尺寸

- 2、可视化数据
将每个特征向量转换为250*2408尺寸的图片，查看不同类别的数据。看不出区别

- 3、拆分数据为训练集于测试集
每个类别按照7：3划分训练集于测试集。

- 4、模型选择
  - 4.1 多层感知机进行初步计算
把数据放入多层感知机需要太多的内存，不可行。
  - 4.2 循环神经网络
考虑到该数据是由视频获取到的，可能存在时序关系。可能是每个特征向量为一个时间点，一个特征数据流包含250个特征向量即为250个时间点。
  - 4.3 卷积神经网络
把特征数据流看为250*2408尺寸的单波段图片。
    首先采用LeNet结构训练，需要调整输入通道以及输出通道，网络结构增加多个卷积层、激活与池化块来缩小参数。模型参数很多，最后存储模型文件为1.3G，应该是全连接层参数太多，需要增加卷积层，减少全连接层参数。
    - 由于内存限制采用batchSize为4，在cpu下每个epoch需要四分钟。学习率设置为0.1，训练20epoch，loss不怎么下降。正确率也很低0.2左右。
    - 修改学习率为0.01：在20epoch内loss没有变化，1.6左右
    - 将lenet的激活函数修改为ReLU：loss明显下降。AlexNet与LeNet的一个区别就是采用了ReLU激活函数。sigmod激活函数可能会使梯度消失或梯度爆炸，相当于没有更新权重，导致loss没有变化，ReLU计算简单，仅会使一部分权重为0，有利于防止过拟合，又不会使所有的权重消失。loss从开始1.6下降到0.25，但在第25epoch之后loss突然上升到1.6，这应该是双下降，之后又下降了。50epoch loss在0.04左右，准确率在0.68左右。
    - 优化函数增加动量法，并增加学习率衰减：缓解参数震荡问题，loss会出现波动，但总体上会更快的趋于平稳。修改之后10epoch内loss没有下降，准确率很低，没有提升。
    - 降低学习率为0.001：loss下降在20epoch为0.9，准确率为0.55.学习率每10轮缩小10倍。收敛的很慢，可能学习率衰减的太快了。
    - dropout，AlexNet与leNet区别为AlexNet在全连接层之间采用dropout，防止过拟合。


- 5、数据增强，如何进行？

- 6、使用k折交叉验证

- 7、分析每个类的错误率