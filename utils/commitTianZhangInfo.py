# 自动化脚本将田长信息录入系统
# @Date 2021-11-19
# @Author small dj

# 1、读取csv中的田长信息，以街道为单位，
# 2、首先通过接口获取该街道所有的村信息，然后根据csv中的田长所属的村名获取该村对应的村编码
# 3、最后组成post请求参数调用添加田长接口
import json
import requests

global streetInfos
global streetInfoFilePath

streetInfos = []
streetInfoFilePath = "/document/田长信息/莱西街道信息.json"

streetIDs = []   # 存放已经查询过的街道id
currentStreetCunInfos = None  # 当前街道的村信息


def checkTZInfofromDB():
    import psycopg2
    # 创建连接对象
    conn = psycopg2.connect(database="onemap", user="postgres",
                            password="123", host="localhost", port="5433")
    cur = conn.cursor()  # 创建指针对象
    a = 0
    with open("/document/tzInfos.csv") as tzInfosFiles:
        tzInfos = tzInfosFiles.readlines()
        for tzinfo in tzInfos:
            tzinfo = tzinfo.split(",")
            sql = 'SELECT count(1) FROM laixi_jbntbhtb where zldwdm=\'' + \
                tzinfo[-1].replace("\n", "") + "0000000\'"
            # 获取结果
            cur.execute(sql)
            results = cur.fetchall()
            if results[0][0] == 0:
                print(results[0][0])
                print(sql)
                a = a + 1
    print(a)

    # 关闭练级
    conn.commit()
    cur.close()
    conn.close()


def getTianZhangInfoFromFile(tzInfoFilePath):
    ''''''
    tzInfo_demo = []  # 获取每个村的村级编码，用于与之前的数据基于此字段进行挂接
    with open(tzInfoFilePath) as tzInfoFile:
        tzInfos = tzInfoFile.readlines()
        for tzInfo in tzInfos:
            infos = tzInfo.split(",")
            streetName = infos[0]
            cunName = infos[1].replace(" ", "")
            tzName = infos[2].replace(" ", "")
            tzPhone = infos[3].replace("\n", "")
            streetCode = getStreetCodeByName(streetName)
            cunCode = getCunCodeByName(cunName, streetCode)
            if cunCode is None:
                print("没有获取到村编码：", streetName, cunName,
                      tzName, tzPhone, streetCode, cunCode)
            else:
                # if tzName == "孙玉珍":
                # bsm = getTZInfo(cunCode)
                # if bsm:
                #     removeTZInfo(bsm, cunCode)
                # else:
                #     print("没有找到该田长", cunCode)
                # addTZInfo(tzName, cunCode, tzPhone)
                tzInfo_demo.append(streetName + "," + cunName +
                                   "," + tzName + "," + tzPhone + "," + cunCode + "\n")
                # print(streetName, cunName, tzName,
                #       tzPhone, streetCode, cunCode)
    saveTZInfo(tzInfo_demo)


def saveTZInfo(tzInfos):
    '''保存田长信息
    '''
    print("田长信息个数：", len(tzInfos))
    with open("/document/tzInfos.csv", 'w') as tzInfoFile:
        tzInfoFile.writelines(tzInfos)


def getStreetCodeByName(name):
    '''根据街道名称获取街道的编码
    '''
    global streetInfos
    global streetInfoFilePath
    if len(streetInfos) == 0:
        with open(streetInfoFilePath) as streetInfoFile:
            streetInfos = json.load(streetInfoFile)["data"]
    for streetInfo in streetInfos:
        if name in streetInfo["label"]:
            return streetInfo["id"]


def getCunCodeByName(cunName, streetCode):
    '''根据村的名称以及街道的编码获取村的编码
        1、首先根据街道的id获取该街道所有的村，将其存放在全局变量中
        2、如果当前街道查询过了，则将街道的id放在全局变量中。之后直接在全局变量中取数据。
    '''
    global streetIDs
    global currentStreetCunInfos
    if streetCode not in streetIDs:
        url = "http://117.73.254.243:8058/webapi/api/tzz/loadchildrenlist?xzqdm=" + streetCode
        response = requests.get(url)
        currentStreetCunInfos = json.loads(response.text)["data"]
        # print(currentStreetCunInfos)
        streetIDs.append(streetCode)
    for cunInfo in currentStreetCunInfos:
        if cunName in cunInfo["label"]:
            return cunInfo["id"]
    return None


def addTZInfo(tzName, cunCode, tzPhone):
    '''新增田长信息
        @param tzName 田长名称
        @param cunCode 村编码
        @param tzPhone 田长电话
    '''
    url = "http://117.73.254.243:8058/webapi/api/tzz/save"
    header = {
        'Content-Type': 'application/json'
    }
    data_all = {
        "tzdh": tzPhone,
        "tzjb": "3",
        "tzxm": tzName,
        "xzqh": cunCode}
    data = json.dumps(data_all)
    req = requests.post(url, data, headers=header)
    result = json.loads(req.text)
    if result["code"] != 200:
        print(req.text, "田长信息添加失败")
    else:
        print("田长信息添加成功：", data_all)


def getTZInfo(cunCode):
    '''根据村编号获取田长信息
        @param cunCode 村编号
    '''
    bsm = None
    url = "http://117.73.254.243:8058/webapi/api/tzz/getuserlist?xzqdm=" + cunCode + "&tzjb=30"
    response = requests.get(url)
    result = json.loads(response.text)["data"]
    for re in result:
        bsm = re["bsm"]
    return bsm


def removeTZInfo(bsm, cunCode):
    '''删除田长信息
        @param bsm 用户编号
    '''
    url = "http://117.73.254.243:8058/webapi/api/tzz/delete?bsm=" + str(bsm)
    response = requests.get(url)
    result = json.loads(response.text)
    if result["code"] != 200:
        print("田长信息删除失败", cunCode)
    else:
        print("田长删除成功：", cunCode)


if __name__ == "__main__":
    print(streetInfos)
    # getTianZhangInfoFromFile("/document/田长信息/三级田长联系手册_all.csv")
    checkTZInfofromDB()
