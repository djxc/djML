# 学习使用detectron2
----
   1. 在detectron2中又很多模型，*faster-rcnn、mask-rcnn*等等；
   2. detectron2使用coco格式读取数据方便，labelme2coco.py文件将labelme格式的数据转换为coco格式
   3. test_detectron2使用自己标注的数据，采用faster-rcnn模型训练，效果不错