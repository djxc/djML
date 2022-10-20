# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:17:54 2019

@author: gis
"""

import itchat
import numpy as np
from pyecharts import Bar, Pie
import jieba
import collections
from PIL import Image
import matplotlib.pyplot as plt
import wordcloud

class djChat:
    def __init__(self):
        itchat.auto_login(hotReload=True)
        self.male = 0
        self.female = 0
        self.other = 0
        self.total = 0
        self.words = []
        
    def listFriends(self):
        friends = itchat.get_friends(update=True)
        self.total = len(friends) -1
        remove_word = ['，', '', ' ', '。', '\n', '"', '>', '<', '!', 'span',
                       '！', '=', 'class', '/', 'emoji', '的', ',', '？', '.',
                       '了', '我', '是', '你']
        for f in friends:
            sex = f["Sex"]
            if sex == 1:
                self.male += 1
            elif sex == 2:
                self.female +=1
            else:
                self.other += 1
#            print(f["NickName"] + "--------" + f["RemarkName"] + "----------" + f["Signature"])
            sig = jieba.cut(f["Signature"], cut_all=False)
            for s in sig:
                if s not in remove_word:
                    self.words.append(s)
                
        
    def showManorWoman(self):
        self.listFriends()
        attr =["male", "female", "other"]
        v1 =[self.male, self.female, self.other]
        pie =Pie("饼图示例")
        pie.add("", attr, v1, is_label_show=True)
        pie.show_config()
        pie.render()
        
        
    def drawWordCloud(self):
        self.listFriends()
        word_counts = collections.Counter(self.words) # 对分词做词频统计
        word_counts_top10 = word_counts.most_common(30) # 获取前10最高频的词
        for w in word_counts_top10:
            print(w)
        mask = np.array(Image.open('longmao2.jpg')) # 定义词频背景
        wc = wordcloud.WordCloud(
            font_path='C:/Windows/Fonts/STXINGKA.TTF', # 设置字体格式
            mask=mask, # 设置背景图
            max_words=200, # 最多显示词数
            max_font_size=100 # 字体最大值
        )
        
        wc.generate_from_frequencies(word_counts) # 从字典生成词云
        image_colors = wordcloud.ImageColorGenerator(mask) # 从背景图建立颜色方案
        wc.recolor(color_func=image_colors) # 将词云颜色设置为背景图方案
        plt.imshow(wc) # 显示词云
        plt.axis('off') # 关闭坐标轴
        plt.show() # 显示图像

        
    
    
if __name__ == '__main__':
    myChat = djChat()
#    myChat.listFriends()
    myChat.drawWordCloud()
#    myChat.showManorWoman()