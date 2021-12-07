# 获取kennet定标数据以及除治点位

import os
import requests
import psycopg2
import json

# keenet服务地址
KEENETAPI_URL = "http://112.6.211.10:8091"


def insetDataToPG():
    conn = psycopg2.connect(database="onemap", user="postgres",
                            password="123", host="192.168.0.110", port="5433")
    print("Opened database successfully")

    cur = conn.cursor()
    cur.execute("SELECT * from onemap_user;")
    rows = cur.fetchall()
    for row in rows:
        print(row)
        print("NAME = ")
        print("threshhold0 = ")
        print("threshhold0 = ")
    print("Operation done successfully")
    conn.close()


def getDataFromJSONFile(filePath):
    '''从json文件中获取定标数据，将其写入到数据库中
        1、定标点位，包含病树位置信息，定标信息，以及除治状态，如果为除治则除治状态为1则为除治之后的病树，其具体的除治信息存储在除治点表中
        2、除治点表仅包含除治的信息，不包括病树详细信息，详细信息可在定标点表中获取
    '''
    conn = psycopg2.connect(database="treeprotect", user="postgres",
                            password="123", host="192.168.0.110", port="5433")
    print("Opened database successfully")
    cur = conn.cursor()
    files = os.listdir(filePath)
    for f in files:
        print(f)
        with open(os.path.join(filePath, f)) as jsonFile:
            dbDatas = json.load(jsonFile)
            for dbD in dbDatas:
                woodID = dbD["woodId"]
                markSn = ""
                if 'markSn' in dbD:
                    markSn = dbD["markSn"]
                sealedflag = -1
                if 'sealedFlag' in dbD:
                    sealedflag = dbD["sealedFlag"]
                markSelf = -1
                if 'markSelf' in dbD:
                    markSelf = dbD["markSelf"]
                markUploadTime = '2000-01-01 00:00:00'
                if 'markUploadTime' in dbD:
                    markUploadTime = dbD["markUploadTime"]
                markTime = '2000-01-01 00:00:00'
                if 'markTime' in dbD:
                    markTime = dbD["markTime"]
                markOperate = ""
                if 'markOperate' in dbD:
                    markOperate = dbD["markOperate"]
                compart = ""
                if "compart" in dbD:
                    compart = dbD["compart"]
                diam = ""
                if 'diam' in dbD:
                    diam = dbD["diam"]
                deptId = -1
                if 'deptId' in dbD:
                    deptId = dbD["deptId"]
                treeStatus = -1
                if 'treeStatus' in dbD:
                    treeStatus = dbD["treeStatus"]
                markFlag = -1
                if 'markFlag' in dbD:
                    markFlag = dbD["markFlag"]
                createTime = '2000-01-01 00:00:00'
                if 'createTime' in dbD:
                    createTime = dbD["createTime"]
                markPic1 = ""
                if 'markPic1' in dbD:
                    markPic1 = dbD["markPic1"]
                markPic2 = ""
                if 'markPic2' in dbD:
                    markPic2 = dbD["markPic2"]
                markPic3 = ""
                if 'markPic3' in dbD:
                    markPic3 = dbD["markPic3"]
                userId = -1
                if 'user' in dbD:
                    userId = dbD["user"]["userId"]
                lat = 0.0
                if 'lat' in dbD:
                    lat = dbD["lat"]
                lng = 0.0
                if 'lng' in dbD:
                    lng = dbD["lng"]
                height = 0.0
                if 'height' in dbD:
                    height = dbD["height"]
                geomStr = "ST_GeomFromText('POINT(" + \
                    str(lng) + ' ' + str(lat) + ")',4326)"
                sqlStr = 'insert into mark_tree_data (woodid, marksn, sealedflag, markself, markoperate, markuploadtime, marktime, compart, diam, height, deptid, treestatus, markflag, createtime, markpic1, markpic2, markpic3, userid, geom) values(%d, \'%s\', %d, %d, \'%s\', \'%s\', \'%s\', \'%s\', \'%s\', %d, %d, %d, %d, \'%s\', \'%s\', \'%s\', \'%s\', %d, %s);' % (
                    woodID, markSn, sealedflag, markSelf, markOperate, markUploadTime, markTime, compart, diam, height, deptId, treeStatus, markFlag, createTime, markPic1, markPic2, markPic3, userId, geomStr)
                # print(sqlStr)
                cur.execute(sqlStr)
            conn.commit()
    cur.close()
    conn.close()


def getSealeTreeDataFromJSONFile(filePath):
    '''获取除治点信息'''
    conn = psycopg2.connect(database="treeprotect", user="postgres",
                            password="123", host="192.168.0.110", port="5433")
    print("Opened database successfully")
    cur = conn.cursor()
    files = os.listdir(filePath)
    for f in files:
        print(f)
        with open(os.path.join(filePath, f)) as jsonFile:
            dbDatas = json.load(jsonFile)
            for dbD in dbDatas:
                woodID = dbD["woodId"]
                sealedUploadTime = '2000-01-01 00:00:00'
                if 'sealedUploadTime' in dbD:
                    sealedUploadTime = dbD["sealedUploadTime"]
                sealedTime = '2000-01-01 00:00:00'
                if 'sealedTime' in dbD:
                    sealedTime = dbD["sealedTime"]
                sealedOperate = ''
                if 'sealedOperate' in dbD:
                    sealedOperate = dbD["sealedOperate"]
                sealedPic1 = ""
                if 'sealedPic1' in dbD:
                    sealedPic1 = dbD["sealedPic1"]
                sealedPic2 = ""
                if 'sealedPic2' in dbD:
                    sealedPic2 = dbD["sealedPic2"]
                sealedPic3 = ""
                if 'sealedPic3' in dbD:
                    sealedPic3 = dbD["sealedPic3"]
                sqlStr = 'insert into seale_tree_data (woodid, sealeduploadtime, sealedtime, sealedoperate, sealedpic1, sealedpic2, sealedpic3) values(%d, \'%s\', \'%s\', \'%s\', \'%s\', \'%s\', \'%s\');' % (
                    woodID, sealedUploadTime, sealedTime, sealedOperate, sealedPic1, sealedPic2, sealedPic3)
                cur.execute(sqlStr)
            conn.commit()
    cur.close()
    conn.close()


def getUserToken():
    '''获取用户token'''
    tokenInfo = requests.get(
        KEENETAPI_URL + "/auth/oauth/token?client_id=android&client_secret=123456&grant_type=password&scope=server&ignore_capcha=1&username=lyj&password=123456")
    if tokenInfo.status_code == 200:
        token = json.loads(tokenInfo.text)["access_token"]
        return token
    else:
        print("登录失败！")
        return None


def getSealeTreeDataFromServer(savePath):
    '''通过接口获取除治数据
        1、根据日期进行数据查询，然后保存为json文件。具体从哪天进行查询可以查看数据库中的markUploadEndTime日期
        2、
    '''
    print("开始登录...")
    kToken = getUserToken()
    if kToken is None:
        return
    print("开始获取数据...")
    # 除治获取401页，定标数据848页
    for page in range(1, 5, 1):
        param = "pageNum=" + str(page) + \
            "&pageSize=1000&markUploadBeginTime=2021-08-12&markUploadEndTime=2021-12-01"
        header = {
            'Content-Type': 'application/json',
            'accept': "*/*",
            "connection": "Keep-Alive",
            "connection": "Keep-Alive",
            "Authorization": "Bearer " + kToken,
            "user-agent": "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1;SV1)"
        }
        result = requests.get(
            KEENETAPI_URL + "/tree/sealed/list" + "?" + param, headers=header)
        if result.status_code == 200:
            resultData = json.loads(result.text)
            if resultData["code"] == 200:
                saveData = resultData["rows"]
                print(len(saveData))
                with open(savePath + "/page_" + str(page) + ".json", "w") as saveJson:
                    for data in saveData:
                        print(data)
                        saveJson.write(data)
                print("获取页码：" + str(page) + "数据成功！")
            else:
                print("获取页码：" + str(page) + "数据失败！")
        else:
            print("获取页码：" + str(page) + "数据失败！")


def getMarkTreeDataFromServer():
    '''通过接口获取定标数据'''
    pass


if __name__ == "__main__":
    # getDataFromJSONFile("/document/keenetTreeData/dingbiao")
    # getSealeTreeDataFromJSONFile("/image_data/keenetTreeData/chuzhi")
    getSealeTreeDataFromServer("/document/keenetTreeData/dingbiao")
