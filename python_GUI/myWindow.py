# -*- coding: utf-8 -*-
import tkinter

class PyWindow:
    def __init__(self, title):        
        self.root = tkinter.Tk()
        self.root.title(title)
        self.photo = tkinter.PhotoImage()       # photo要与root在同一个作用域内才显示图片，否则不显示

    # 开始运行程序，才能显示界面
    def run(self):
        self.root.mainloop()


    # 添加label控件
    def addLabel(self, text):
        label = tkinter.Label(self.root, font=('幼圆', 20), fg='red')    
        label['text'] = text
        label.pack()                      # 自动调节组件自身的尺寸


    def addButton(self, BText, BEvent):        
        button = tkinter.Button(self.root)  
        button['text'] = BText 
        button['command'] = BEvent        # 设置按钮的点击事件
        button.pack()
    
    def addEntry(self, EText):
        entry =  tkinter.Entry(self.root)  
        entry['textvariable'] = EText  
        entry.pack()

    # 创建一个label用来显示一个图片。先创建photo然后将其放在label中
    def addPhoto(self, filePath):
        self.photo['file'] = filePath
        theLabel = tkinter.Label(self.root, justify=tkinter.LEFT, image=self.photo, compound=tkinter.CENTER)
        theLabel.pack()

    # 根据文件选择的图片，修改photo
    def changePhoto(self, photoPath):
        self.photo['file'] = photoPath

    def getRoot(self):
        return self.root
    
    def addFram(self):
        tkinter.Label(self.root, text='界面布局', font=('Arial', 20)).pack()

        frm =tkinter.Frame(self.root)
        #left
        frm_L = tkinter.Frame(frm)
        tkinter.Label(frm_L, text='厚德', font=('Arial', 15)).pack(side=tkinter.TOP)
        tkinter.Label(frm_L, text='博学', font=('Arial', 15)).pack(side=tkinter.TOP)
        frm_L.pack(side=tkinter.LEFT)

        #right
        frm_R = tkinter.Frame(frm)
        tkinter.Label(frm_R, text='敬业', font=('Arial', 15)).pack(side=tkinter.TOP)
        tkinter.Label(frm_R, text='乐群', font=('Arial', 15)).pack(side=tkinter.TOP)
        frm_R.pack(side=tkinter.RIGHT)

        frm.pack()
        
