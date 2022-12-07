import xlrd
import openpyxl as xl
import os
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time


# 合并两表的属性，对于主键相同的项，对欠费金额、收入加和
def attrMerge():
    main_chart = xl.load_workbook(os.path.dirname(__file__) + '/data/raw_data.xlsx', data_only=True)
    main_sheet = main_chart['Sheet1']
    supple_chart = xlrd.open_workbook(os.path.dirname(__file__) + '/data/attr_supplement.xlsx')
    supple_sheet = supple_chart.sheet_by_name('Sheet1')
    print("open 2 files successfully")

    # 取attr_supplement.xlsx中第5列的主键
    key_supple = supple_sheet.col_values(5)

    i = 2  # 原表
    while i < 8640:
        if main_sheet.cell(row=i, column=28).value != 1:
            # 找到重复的主键和重复次数
            key_dup = main_sheet.cell(row=i, column=26).value
            dup = main_sheet.cell(row=i, column=28).value
            # print(key_dup, ", ", dup)

            # 计算总税后收入、总欠费金额
            """
            total = 0
            j = 1  # 属性表
            while 1:
                if key_dup == key_supple[j]:
                    # print(supple_sheet.cell(j, 5))
                    k = 0
                    while k < dup:
                        total += supple_sheet.cell(j+k, 9).value
                        k += 1
                    # print("total: ", total)
                    break
                j += 1
            # print("第%s行" % (j + 1), "相同")
            main_sheet.cell(row=i, column=30).value = total
            """

            # 计算最大欠费月份
            max = 0
            j = 1  # 属性表
            while 1:
                if key_dup == key_supple[j]:
                    # print(supple_sheet.cell(j, 5))
                    k = 0
                    while k < dup:
                        if max < supple_sheet.cell(j + k, 10).value:
                            max = supple_sheet.cell(j + k, 10).value
                        k += 1
                    break
                j += 1
            main_sheet.cell(row=i, column=31).value = max
        else:
            main_sheet.cell(row=i, column=31).value = 0

        if i % 500 == 0:
            print("计算到第%d行" % i)
            main_chart.save(os.path.dirname(__file__) + '/data/raw_data.xlsx')
        i += 1

    main_chart.save(os.path.dirname(__file__) + '/data/raw_data.xlsx')


def boolMake():
    chart = xl.load_workbook(os.path.dirname(__file__) + '/data/raw_data.xlsx', data_only=True)
    sheet = chart['Sheet1']
    print("open file successfully")

    i = 2
    while i < 8640:
        if sheet.cell(row=i, column=12).value != 0:
            sheet.cell(row=i, column=12).value = 1

        if i % 500 == 0:
            print("计算到第%d行" % i)
            chart.save(os.path.dirname(__file__) + '/data/raw_data.xlsx')
        i += 1

    chart.save(os.path.dirname(__file__) + '/data/raw_data.xlsx')


def distributionAnalyze(column):
    s = pd.read_excel(os.path.dirname(__file__) + '/data/raw_data.xlsx', sheet_name='Sheet1', usecols=column)
    # 画散点图和直方图
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)  # 创建子图1
    ax1.scatter(s.index, s.values)
    plt.grid()


    ax2 = fig.add_subplot(2, 1, 2)  # 创建子图2
    s.hist(bins=30, alpha=0.5, ax=ax2)
    s.plot(kind='kde', secondary_y=True, ax=ax2)
    plt.grid()
    plt.savefig(os.path.dirname(__file__) + '\\image\\TotalCallDuration.png')


def FeatureSelection():
    chart = xlrd.open_workbook(os.path.dirname(__file__) + '/data/raw_data.xlsx')
    sheet = chart.sheet_by_name('Sheet1')

    # 相关矩阵
    correlation = []
    for i in range(14):
        x = sheet.col_values(7+i)
        del x[0]
        j = 7+i
        correlation.append([])
        while j < 21:
            y = sheet.col_values(j)
            del y[0]

            # 使用spearman系数计算属性间相关性
            s, p = stats.spearmanr(x, y)
            correlation[i].append(s)

            j += 1

    print(correlation)

    chart = xl.load_workbook(os.path.dirname(__file__) + '/data/raw_data.xlsx', data_only=True)
    sheet_show = chart['Sheet3']

    for i in range(14):
        for j in range(14-i):
            sheet_show.cell(row=i+1, column=j+1).value = correlation[i][j]
            j += 1
        i += 1
    chart.save(os.path.dirname(__file__) + '/data/raw_data.xlsx')


def dataNormalize():
    chart = xl.load_workbook(os.path.dirname(__file__) + '/data/association_data.xlsx', data_only=True)
    sheet = chart['Sheet1']

    s = pd.read_excel(os.path.dirname(__file__) + '/data/association_data.xlsx', sheet_name='Sheet1', usecols='O')
    s.head()

    # 使用min-max归一化
    min_max = preprocessing.MinMaxScaler()

    result = min_max.fit_transform(s)
    result = np.round(result, 5)  # 取5位小数

    j = 0
    while j in range(1):
        i = 0
        while i < 8638:
            sheet.cell(row=i+2, column=j+15).value = float(result[i][j])
            i += 1
        chart.save(os.path.dirname(__file__) + '/data/association_data.xlsx')
        j += 1


def dateTransform():
    chart = xl.load_workbook(os.path.dirname(__file__) + '/data/classify_data.xlsx', data_only=True)
    sheet = chart['Sheet1']

    i = 0
    while i < 8638:
        date = sheet.cell(row=i+2, column=8).value  # 读取数据
        # 将格式为“20210101”的字符串转换为日期
        date = time.mktime(time.strptime(date, '%Y%m%d'))
        now = time.time()  # 记录当前时间
        total_s = now - date
        total_d = int(total_s/(60*60*24))  # 计算天数
        sheet.cell(row=i+2, column=8).value = total_d  # 写入数据
        i += 1
    chart.save(os.path.dirname(__file__) + '/data/classify_data.xlsx')


def CustomerLevel():
    raw_data = pd.read_excel(os.path.dirname(__file__) + '/data/raw_data.xlsx', sheet_name='Sheet1', usecols='A')
    raw_data.head()

    level_data = pd.read_excel(os.path.dirname(__file__) + '/data/attr_supplement.xlsx', sheet_name='Sheet1',
                               usecols='A, I')
    level_data.head()
    print('Open 2 files Successfully')

    # result = pd.DataFrame()
    result = []

    # 去除电话号码重复值
    raw_data.drop_duplicates(subset='PhoneNumber', keep='first', inplace=True)
    raw_data = raw_data.reset_index(drop=True)
    # raw_data = raw_data.tolist()

    for i in range(raw_data.shape[0]):
        # print(i)
        temp = level_data.where(level_data.PhoneNumber == raw_data['PhoneNumber'][i])
        # 去除空值
        temp = temp.dropna()
        temp = temp.reset_index(drop=True)
        m, n = temp.shape

        # 记录号码和重复次数
        levels = [raw_data['PhoneNumber'][i], m]

        # 记录星级
        for j in range(m):
            s = temp['Level'][j]
            levels.extend([s])
        result.append(levels)

        if i % 1000 == 0:
            print('%s rows have been finished reading' % i)

    df = pd.DataFrame(result)
    df.to_excel(os.path.dirname(__file__) + '/data/level _data.xlsx')


if __name__ == '__main__':
    # 合并两张表格的属性值
    # attrMerge()

    # 将二元属性统一为布尔值
    # boolMake()

    # 判断属性是否为正态分布
    # distributionAnalyze('P')

    # 特征选择
    # FeatureSelection()

    # 数据归一化
    # dataNormalize()

    # 将日期转化为时间长度
    # dateTransform()

    # 统计客户星级
    CustomerLevel()
