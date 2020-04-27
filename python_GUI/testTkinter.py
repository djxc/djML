# -*- coding: utf-8 -*-
import tkinter
import myWindow
from tkinter.filedialog import askdirectory, askopenfilename

def on_click():
    print('你真点呀！')

# 选择文件，然后修改photo
def selectFile():
    filename = askopenfilename()
    pWin.changePhoto(filename)


pWin = myWindow.PyWindow('myApp')

pWin.addLabel('小孩已经长大了。。。')
pWin.addButton('有本事点我呀', on_click)
pWin.addButton('选择一张图片', selectFile)
#pWin.addPhoto("G:/bui.gif")

# 添加可变字符串变量
text =  tkinter.StringVar()  
text.set('这是一个StringVar变量')  

pWin.addEntry(text)
pWin.addFram()
pWin.run()

