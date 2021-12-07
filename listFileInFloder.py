import os
import shutil
import random


def listFileInFloader():
    change_image_floder = "/document/2020/rs_detection/change_detection_val/val/im1"
    result_file = "/document/2020/rs_detection/change_detection_val/val/val_allImage.txt"
    img1files = os.listdir(change_image_floder)
    with open(result_file, 'w') as allImageFile:
        for imgName in img1files:
            print(imgName)
            allImageFile.write(imgName + "\n")

def moveValImage():
    input_floder = "/djxc/rs_detection/change_detection_train/train/"
    out_floder = "/djxc/rs_detection/change_detection_train/val/"
    val_file = "/djxc/rs_detection/change_detection_train/train/val.txt"
    with open(val_file, 'r') as valFile:
        lines = valFile.readlines()
        img_label = [line.replace("\n", "").split("  ") for line in lines]
        for valF in img_label:
            m1 = input_floder + "im1/" + valF[0]
            print(m1)
            m2 = input_floder + "im2/" + valF[0]
            label1 = input_floder + "label1/" + valF[0]
            label2 = input_floder + "label2/" + valF[0]
            label1_rgb = input_floder + "label1_rgb/" + valF[0]
            label2_rgb = input_floder + "label2_rgb/" + valF[0]
            
            shutil.copy(m1, out_floder + "im1/")
            shutil.copy(m2, out_floder + "im2/")
            shutil.copy(label1, out_floder + "label1/")
            shutil.copy(label2, out_floder + "label2/")
            shutil.copy(label1_rgb, out_floder + "label1_rgb/")
            shutil.copy(label2_rgb, out_floder + "label2_rgb/")

def split_ccfdata():
    image_floder = "/document/2020/CCF_data/train_data/img_train/"
    train_file = "/document/2020/CCF_data/train_data/train_data.txt"
    verfy_file = "/document/2020/CCF_data/train_data/verfy_data.txt"
    img1files = os.listdir(image_floder)
    train_data = open(train_file, "w")
    val_data = open(verfy_file, "w")    

    random.shuffle(img1files)
    random.shuffle(img1files)
    for data in img1files:
        proba = random.random()
        if proba < val_percent:
            val_data.write(data +  "\n")
        else:
            train_data.write(data + "\n")
    train_data.close()
    val_data.close()   

if __name__ == "__main__":
    listFileInFloader(0.1)   
    # moveValImage()