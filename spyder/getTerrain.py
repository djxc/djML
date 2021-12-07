# -*- coding: utf-8 -*-
import os
import  requests
import PIL.Image as Image

# https://assets.cesium.com/1/1/3/0.terrain?extensions=octvertexnormals-watermask-metadata&v=1.2.0
def image_joint(image_list,opt):#opt= vertical ,horizontal 选择水平显示拼接的图像，或者垂直拼接
    image_num=len(image_list)
    image_size=image_list[0].size
    height=image_size[1]
    width=image_size[0]
    
    if opt=='vertical':
        new_img=Image.new('RGB',(width,image_num*height),255)
    else:
        new_img=Image.new('RGB',(image_num*width,height),255)
    x=y=0
    count=0
    for img in image_list:
        
        new_img.paste(img,(x,y))
        count+=1
        if opt=='horizontal':
            x+=width
        else : y+=height
    return new_img

def wirtePicture(x, y):    
    z = '&z=21'
    url = "http://www.google.cn/maps/vt?lyrs=s@803&gl=cn&x="    # 1731111&y=836147&z=21"   
    for y in range(835900, 836302):        
        for x in range(1730000, 1731300):        
            img_url = url + str(x) + '&y=' + str(y) + z
            img = requests.get(img_url) 
            file = 'img/%s.jpg' %( str(x) + '****' + str(y))
            print(file)
            with open(file,'ab') as f: #存储图片，多媒体文件需要参数b（二进制文件）
                f.write(img.content) #多媒体存储content
            x += 1
            
def getFileFromWeb(webUrl):
    '''从网络上获取文件
        @param webUrl 网络地址
    '''
    for fileName in fileList:
        xmlName = fileName.split(".")[0] + ".xml"
        img_url = webUrl + fileName
        xml_url = webUrl + xmlName
        print(xml_url)
        img = requests.get(img_url) 
        imgSavePath = '/2020/clothes_person/%s' %(fileName)
        with open(imgSavePath,'ab') as f: #存储图片，多媒体文件需要参数b（二进制文件）
            f.write(img.content)  #多媒体存储content
        xml = requests.get(xml_url)
        xmlSavePath = '/2020/clothes_person/%s' % (xmlName)
        with open(xmlSavePath,'ab') as f: #存储图片，多媒体文件需要参数b（二进制文件）
            f.write(xml.content)  #多媒体存储content
                  
    
def mergePicture(x):
    y = 835900
    img_list = []
    for y in range(835900, 836302):             
        imlist = []
        for x in range(1730000, 1731300):     
            file = 'img/%s.jpg' %( str(x) + '****' + str(y))
            print(file)
            img = Image.open(file)       
            imlist.append(img)
        jimg = image_joint(imlist, 'horizontal')        
        img_list.append(jimg)
    limg = image_joint(img_list, 'vertical')   
#    limg.show()
    limg.save('demo2.jpg', 'jpeg')
    

def getTerrainData():
    savePath = "/2020/terrain"
    for i in range(0, 100):
        for j in range(0, 100):
            for k in range(0, 100):
                terrainUrl = "https://assets.cesium.com/1/" + str(i) + "/" + str(j) + "/" + str(k) + ".terrain?extensions=octvertexnormals-watermask-metadata&v=1.2.0"
                headers = {
                    "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:46.0) Gecko/20100101 Firefox/46.0",
                    "Accept": "application/vnd.quantized-mesh,application/octet-stream;q=0.9,*/*;q=0.01",
                    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyNWZjMjA0OC0zMWYyLTRkZWYtOTQ3OC03ZWM0MjE1NWY2YjkiLCJpZCI6MTQwNjAsImFzc2V0cyI6eyIxIjp7InR5cGUiOiJURVJSQUlOIiwiZXh0ZW5zaW9ucyI6W3RydWUsdHJ1ZSx0cnVlXSwicHVsbEFwYXJ0VGVycmFpbiI6ZmFsc2V9fSwic3JjIjoiYzk5YThmZTUtZjQwMC00YjM5LWE4MDEtZmJhZjhiYmM4OGIxIiwiaWF0IjoxNjMwNDc4NDk4LCJleHAiOjE2MzA0ODIwOTh9.BWiBM4ylMWXMVPfTXbNc7YSRLilPVqaDJkPsgnF92SQ"
                }
                terrainToSavePath = os.path.join(savePath, "/" + str(i) + "/" + str(j))
                if not os.path.exists(terrainToSavePath):
                    os.makedirs(terrainToSavePath)
                response = requests.get(terrainUrl, headers=headers) 
                with open(terrainToSavePath + str(k) + ".terrain",'w') as f:
                    if response.status_code == 200:
                        print(terrainToSavePath)
                        f.write(response.text)  
                    else:
                        print("error get data")
    
if __name__ == "__main__":
    # x = 1730000
    # y = 835900
    # wirtePicture(x, y)
    # mergePicture(x)
    # getFileFromWeb("https://minio.cvmart.net/test-image-pvc/0b1b99a6ad68a7b0110894aa18945afd/sample/fanguangyishibie/v2/")
    getTerrainData()
    
    