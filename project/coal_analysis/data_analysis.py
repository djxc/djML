import pandas
import numpy as np

data = pandas.read_csv(r"D:\Data\MLData\基于近红外光谱的煤质参数预测挑战赛公开数据\train_data.csv")
data = data.to_numpy()
raw_data = data[:, 1:]
raw_data_mean = raw_data.mean(axis=0)
dif_result = raw_data - raw_data_mean
print(dif_result.max(axis=0))
print(data.shape)