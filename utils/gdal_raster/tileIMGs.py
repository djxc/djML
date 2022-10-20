# 对指定目录下的tif进行切片
import os
import gdal2tiles

tileSavePath = "/document/Data/nginx_home/oneMapData/tileCache"


def tileIMGs(imgsPath):
    ''''''
    print(imgsPath)
    imgList = os.listdir(imgsPath)
    imagePathList = []
    for img in imgList:
        img_full_path = os.path.join(imgsPath, img)
        suffix = img.split(".")[-1]
        if os.path.isfile(img_full_path) and suffix in ["tif", "TIF", "tiff", "TIFF", "img"]:
            print(img)
            imagePathList.append(img_full_path)

    print("待切片影像个数：", len(imagePathList))
    r = 1
    for img in imagePathList:
        print("正在进行第%d张图切片" % r)
        name = img.split("/")[-1]
        gdal2tiles.generate_tiles(img,
                                  os.path.join(tileSavePath, name),
                                  nb_processes=3,
                                  #   zoom='10-20',
                                  resume=True)
        r = r + 1


if __name__ == "__main__":
    tileIMGs("/document/Data/image_data")
