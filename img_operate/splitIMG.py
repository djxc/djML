import cv2

img = cv2.imread("/2020/DJI_0168 Panorama.jpg")
height, width, bands = img.shape
print(height, width, bands)
size = 16
ySize = 8
for i in range(size):
    for j in range(ySize):
        img_part = img[int(height/ySize) * j:int(height/ySize) * (j+1), int(width / size) * i:int(width/size) * (i+1)]
        cv2.imwrite("/2020/DJI_part"+ str(i+1) + str(j+1) + ".jpg", img_part)
        # img_part = img[int(height/8):int(height/8) * 2, int(width / size) * i:int(width/size) * (i+1)]
        # cv2.imwrite("/2020/DJI_part"+ str(i+1) + "2.jpg", img_part)
        print("/2020/DJI_part"+ str(i+1) + str(j+1) + ".jpg")
