"""
使用python对excel进行操作
1、读取excel，获取其中一个sheet，遍历每一行，读取其中数据。
@author small dj
@date 2020-04-12
"""
import xlrd


def read_excel(excelPath):
    """
    读取excel文件，excelPath为excel文件的路径
    可以获取某一列数据或是某一行数据；
    也可以逐行遍历，获取改行下的某一列数据，类似于矩阵。
    """
    # 打开文件
    workbook = xlrd.open_workbook(excelPath)
    # 获取所有sheet
    print(workbook.sheet_names())  # [u'sheet1', u'sheet2']
    # 根据sheet索引或者名称获取sheet内容
    sheet1 = workbook.sheet_by_index(0)  # sheet索引从0开始

    # sheet的名称，行数，列数
    print(sheet1.name, sheet1.nrows, sheet1.ncols)

    # 获取整行和整列的值（数组）
    # rows = sheet1.row_values(3)  # 获取第四行内容
    # cols = sheet1.col_values(2)  # 获取第三列内容
    beizhaiIllTree = [0, 0, 0, 0, 0, 0, 0]
    shazikouIllTree = [0, 0, 0, 0, 0, 0, 0]
    wanggezhuangIllTree = [0, 0, 0, 0, 0, 0, 0]
    jinjialingIllTree = [0, 0, 0, 0, 0, 0, 0]
    zhonghanIllTree = [0, 0, 0, 0, 0, 0, 0]
    linchangIllTree = [0, 0, 0, 0, 0, 0, 0]
    illTree_sum = 0
    for r in range(1, sheet1.nrows):
        """逐行遍历，第一行为表头不读取"""
        if r % 100 == 0:
            print(r)
        row = sheet1.row(r)
        name = row[0].value
        lon = row[1].value
        lat = row[2].value
        alt = row[3].value
        size = row[4].value
        # if alt >= 150:
        #     illTree_sum += 1
        if isinstance(size,str):
            size = float(size)
        
        if name[0] == 'B':
            leval_value(beizhaiIllTree, alt, size)
        elif name[0] == 'S' or name[0] == 's':
            leval_value(shazikouIllTree, alt, size)
        elif name[0] == 'W':
            leval_value(wanggezhuangIllTree, alt, size)
        elif name[0] == 'J':
            leval_value(jinjialingIllTree, alt, size)
        elif name[0] == 'Z':
            leval_value(zhonghanIllTree, alt, size)
        elif name[0] == 'L':
            leval_value(linchangIllTree, alt, size)

    print('beizhai' ,beizhaiIllTree)
    print('wnaggezhuang', wanggezhuangIllTree)
    print('shazikou', shazikouIllTree)
    print('jinjialing', jinjialingIllTree)
    print('zhonghan', zhonghanIllTree)
    print('linchang', linchangIllTree)

    # 获取单元格内容
    # print(sheet1.cell(1, 0).value.encode('utf-8'))
    # print(sheet1.cell_value(1, 0).encode('utf-8'))
    # print(sheet1.row(1)[0].value.encode('utf-8'))

    # 获取单元格内容的数据类型
    print(sheet1.cell(1, 0).ctype)
    # print(illTree_sum)


def leval_value(illPolygon, alt, size):
    '''统计不同街道的六种等级病树个数
        illPolygon为街道的不同等级病树个数的数组；
        alt为该病树的海拔；
        size为病树的胸径
    '''
    illPolygon[0] += 1
    if alt < 200:
        if size < 15:
            illPolygon[1] += 1
        elif size > 25:
            illPolygon[3] += 1
        else:
            illPolygon[2] += 1
    else:
        if size < 15:
            illPolygon[4] += 1
        elif size > 25:
            illPolygon[6] += 1
        else:
            illPolygon[5] += 1

if __name__ == '__main__':
    read_excel('D:\\2020\\病树前头万木春\\illTree_size_sum1.xlsx')
