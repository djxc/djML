import cv2
import numpy as np

imgdata = 255 * np.ones([256, 256, 3])
text = "no data this level"
cv2.putText(imgdata, text, (50, 115), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1)
cv2.imwrite("./nodata_tile.jpg", imgdata)