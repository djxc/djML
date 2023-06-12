- 1、用MLP进行训练

数据集4，
cls 0 acc is 0.824, total: 17, error: 3, error_info: {"1": 2, "4": 1}
cls 1 acc is 0.872, total: 39, error: 5, error_info: {"4": 1, "3": 2, "0": 1, "2": 1}
cls 2 acc is 0.857, total: 42, error: 6, error_info: {"3": 2, "4": 3, "0": 1}
cls 3 acc is 0.950, total: 40, error: 2, error_info: {"1": 2}
cls 4 acc is 0.619, total: 42, error: 16, error_info: {"2": 10, "0": 3, "3": 3}
accuracy = 0.8222222222222222


对mlp增加池化层，提取重要数据，减少其他数据干扰