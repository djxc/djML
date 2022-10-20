# 操作txt文件
import os

def margeTxtFiles(textFolder):
    '''将多个txt文件合并为一个'''
    print(textFolder)
    textFiles = os.listdir(textFolder)
    txt_lines = []
    for textFile in textFiles:
        textName = textFile.replace(".txt", "")
        if "_" in textName:
            textName = textName.replace("_", "/")
        with open(os.path.join(textFolder, textFile)) as txt_file:
            lines = txt_file.readlines()
            for line in lines:
                line = textName + "/" + line
                txt_lines.append(line)
        print(textName)
    with open("/2020/9月核查点/allFile.txt", "x") as allFile:
        allFile.writelines(txt_lines)
    # print(txt_lines)

if __name__ == "__main__":
    margeTxtFiles(r"/2020/9月核查点")