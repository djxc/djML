#coding:utf-8
import os
from PIL import Image
import glob
import fitz
import os

def getImages(imagePath, pdfName):
    images = os.listdir(imagePath)
    images_ = []
    firestImage = None
    i = 0
    for image in images:
        img = Image.open(os.path.join(imagePath, image))
        if i == 0:
            firestImage = img
        else:
            images_.append(img)
        i = i + 1
    firestImage.save(pdfName, "PDF", resolution=100.0, save_all=True, append_images=images_)


  
def pictopdf():
    '''将图片转换为pdf'''
    doc = fitz.open()
    for img in sorted(glob.glob("pdf/*")):  # 读取图片，确保按文件名排序
        print(img)
        imgdoc = fitz.open(img)                 # 打开图片
        pdfbytes = imgdoc.convertToPDF()        # 使用图片创建单页的 PDF
        imgpdf = fitz.open("pdf", pdfbytes)
        doc.insertPDF(imgpdf)                   # 将当前页插入文档
    if os.path.exists("newpdf.pdf"):        # 若文件存在先删除
        os.remove("newpdf.pdf")
    doc.save("newpdf.pdf")                   # 保存pdf文件
    doc.close()

def getIMGFromPDF(pdfPath, originPath):
    '''从pdf中获取图片保存在文件中'''
    pdfName = pdfPath.split(".")[0]
    print(os.path.join(originPath, pdfPath))
    doc = fitz.open(os.path.join(originPath, pdfPath))

    width, height = fitz.PaperSize("a4")
    
    totaling = doc.pageCount
    
    for pg in range(totaling):
        page = doc[pg]
        zoom = int(100)
        rotate = int(0)
        print(page)
        trans = fitz.Matrix(zoom / 100.0, zoom / 100.0).preRotate(rotate)
        pm = page.getPixmap(matrix=trans, alpha=False)
    
        lurl= originPath + '/%s.jpg' % (pdfName + str(pg+1))
        pm.writePNG(lurl)
    doc.close()

if __name__ == "__main__":
    # getImages("/2020/testPDF", "/2020/testpdf.pdf")
    pdf_root = "/2020/pdf/originPDF"
    pdfs = os.listdir(pdf_root)
    for pdf in pdfs:
        print(pdf, pdf_root)
        getIMGFromPDF(pdf, pdf_root)