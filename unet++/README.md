## 多分类的unet以及unet++
- 1. 目录结构
 - 1. train.py 为训练程序文件
 - 2. losses.py 为损失函数计算文件
 - 3. LovaszSoftmax 为损失函数计算算法
 - 4. config.py 为配置文件
 - 5. inference.py 以及 inference_color.py 为预测图像文件，第一个输出为单通道的图像，第二个输出彩色图便于人眼观看。
 - 6. unet 文件夹为具体的unet的算法的定义
 - 7. utils 文件夹为一些工具函数，包括：数据加载、颜色生成等。
- 2. 数据说明
 - 1. 数据结构采用如下分布，
 ```javascript
    data
    ├── images
    |   ├── 0a7e06.jpg
    │   ├── 0aab0a.jpg
    │   ├── 0b1761.jpg
    │   ├── ...
    |
    └── masks
        ├── 0a7e06.png
        ├── 0aab0a.png
        ├── 0b1761.png
        ├── ...
```
- 3. 模型训练
`python train.py`
- 4. 模型预测
`python inference.py -m ./data/checkpoints/epoch_10.pth -i ./data/test/input -o ./data/test/output`