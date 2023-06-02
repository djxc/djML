import pandas as pd  # 数据集处理包
import numpy as np # 矩阵处理包
import matplotlib as mpl #画图包
import matplotlib.pyplot as plt # 画图包
from sklearn.model_selection import train_test_split # 模型训练集和测试集划分包
import xgboost as xgb # XGBoost 包
from xgboost.sklearn import XGBClassifier # 设置模型参数
import matplotlib.pylab as plt 
from xgboost import plot_importance #特征重要性图形输出
from sklearn import metrics # 模型评估包



df= pd.read_csv(r'E:\Data\MLData\鸢尾花\iris.csv')
print(df.shape)

# 2. 查看分组情况
groups=df.groupby(['Class'])
print('group_count',groups.count()) # 按class 分类输出每个特征个数
print('group_mean',groups.mean()) #按class 分类输出每个特征均值
print('group_std',groups.std()) #按class分类输出每个特征标准差

# 3. 查看统计描述
max=df['SepalLength'].max() #最大值
min=df['SepalLength'].min() #最小值
mean=df['SepalLength'].mean() #平均值
median=df['SepalLength'].median() #中位数
mode=df['SepalLength'].mode()#众数
print('最大值:',max)
print('最小值:',min)
print('平均值:',mean)
print('中位数:',median)
print('众数:',mode)

df['Target'] = df.Class.map({
  "setosa":0,
  "versicolor":1,
  "virginica":2})

print(df.head())

df_x=df[['SepalLength','SepalWidth','PetalLength','PetalWidth']] #特征变量
df_y=df['Target'] # 目标变量


train_x,test_x,train_y,test_y = train_test_split(df_x,df_y,test_size=0.3,random_state=123)
print("train_x:",train_x.shape, "test_x:",test_x.shape,'train_y:',train_y.shape,'test_y:',test_y.shape)
    
# df_x：待划分样本数据自变量（模型特征变量）
# df_y：待划分样本数据的目标变量
# test_size：测试数据占样本数据的比例
# random_state：设置随机数种子，保证每次都是同一个随机数

# 模型参数设置
model_new = XGBClassifier(
    learning_rate =0.1,
    n_estimators=50,
    max_depth=5,
    min_child_weight=5,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    seed=42)

# 对训练集训练模型
model_new.fit(train_x,train_y)

plot_importance(model_new)
plt.show()