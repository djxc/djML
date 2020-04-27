# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 0、导入数据集,绘制小提琴图
df = pd.read_excel('G://dj.xlsx', 'Sheet1')

plt.figure(figsize=(15,10))

plt.subplot(181)
sns.violinplot(y=df['Cr'])
sns.despine()
plt.subplot(182)
sns.violinplot(y=df['Cu'])
sns.despine()
plt.subplot(183)
sns.violinplot(y=df['Ni'])
sns.despine()
plt.subplot(184)
sns.violinplot(y=df['Pb'])
sns.despine()
plt.subplot(185)
sns.violinplot(y=df['Zn'])
sns.despine()
plt.subplot(186)
sns.violinplot(y=df['Cd'])
sns.despine()
plt.subplot(187)
sns.violinplot(y=df['As'])
sns.despine()
plt.subplot(188)
sns.violinplot(y=df['Hg'])
sns.despine()


#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.boxplot(df['Cr'])
#plt.show()