import graphviz
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, naive_bayes, svm, ensemble, model_selection, tree, linear_model
from sklearn.metrics import accuracy_score
from sklearn import metrics
import time
import decimal
from sklearn.base import is_classifier # 用于判断是回归树还是分类树
from dtreeviz.colors import adjust_colors # 用于分类树颜色（色盲友好模式）
import seaborn as sns #用于回归树颜色
from matplotlib.colors import Normalize # 用于标准化RGB数值
import matplotlib.cm as cm
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'  # 设定Graphviz环境


UsageColumn = ['TotalFlow', '4GDuration', '4GFlow', 'TotalCallDuration', 'CallingDuration', 'TotalRevenue',
               'Registration']
ArrearageColumn = ['TotalArrearage', 'MaxArrearageTime', 'Registration']
features = ['IfUnlimited', 'IfCoalesce', 'IfContract', 'IfSMSSilence', 'TotalFlow', '4GDuration',
            '4GFlow', 'TotalCallDuration', 'CallingDuration', 'TotalRevenue', 'TotalArrearage',
            'MaxArrearageTime', 'Registration']


def clusterAnalyze(cluster_name):
    # 从文件中读取数据
    global result
    global columns
    data = pd.read_excel(os.path.dirname(__file__) + '/data/classify_data.xlsx', sheet_name='Sheet3')
    data.head()

    num = 0
    # 读取对应列并创建结果DF
    if cluster_name == 'Usage':
        UsageColumn.append(cluster_name)
        columns = UsageColumn
        num = 3
    elif cluster_name == 'Arrearage':
        ArrearageColumn.append(cluster_name)
        columns = ArrearageColumn
        num = 4

    data = data[columns]
    result = pd.DataFrame(columns=columns)

    # 计算每个聚类每一列的均值
    for i in range(num):
        temp = data.where(data[cluster_name] == i)
        means = []
        for col in columns:
            means.append(temp[col].mean())
        result.loc[i] = means

    # 保留两位小数
    result.round(2)
    result.to_excel(os.path.dirname(__file__) + '/data/Mean%s.xlsx' % cluster_name)
    print(result)


def dataClassify():
    # Sheet6 -> 用于聚类的数据
    data = pd.read_excel(os.path.dirname(__file__) + '/data/classify_data.xlsx', sheet_name='Sheet6')
    data.head()

    data_X = data[features]  # 属性值
    data_Y = data['Clusters']  # 标签

    m, n = data_X.shape

    score = []
    run_time = []

    # KNN、NB、SVM、RF、DT, LR六种分类器
    KNN = neighbors.KNeighborsClassifier(n_neighbors=5, weights="distance")
    NB = naive_bayes.GaussianNB()
    SVM = svm.SVC()
    RF = ensemble.RandomForestClassifier()
    DR = tree.DecisionTreeClassifier(criterion='entropy')
    LR = linear_model.LogisticRegression()

    # 分出训练集和测试集
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data_X, data_Y, train_size=0.7, random_state=0)

    # KNN分类
    start = time.time()
    KNN.fit(X_train, Y_train)
    Y_predict = KNN.predict(X_test)
    end = time.time()
    score.append(accuracy_score(Y_test, Y_predict))
    run_time.append(end-start)

    # NaiveBayes分类
    start = time.time()
    NB.fit(X_train, Y_train)
    Y_predict = NB.predict(X_test)
    end = time.time()
    score.append(accuracy_score(Y_test, Y_predict))
    run_time.append(end - start)

    # SVM分类
    start = time.time()
    SVM.fit(X_train, Y_train)
    Y_predict = SVM.predict(X_test)
    end = time.time()
    score.append(accuracy_score(Y_test, Y_predict))
    run_time.append(end - start)

    # 随机森林分类
    start = time.time()
    RF.fit(X_train, Y_train)
    Y_predict = RF.predict(X_test)
    end = time.time()
    score.append(accuracy_score(Y_test, Y_predict))
    run_time.append(end - start)

    # 决策树分类
    start = time.time()
    DR.fit(X_train, Y_train)
    Y_predict = DR.predict(X_test)
    end = time.time()
    score.append(accuracy_score(Y_test, Y_predict))
    run_time.append(end - start)

    # 逻辑回归分类
    start = time.time()
    LR.fit(X_train, Y_train)
    Y_predict = LR.predict(X_test)
    end = time.time()
    score.append(accuracy_score(Y_test, Y_predict))
    run_time.append(end - start)
    """

    # 确定K值
    i = 3
    score = []
    while i < 10:
        KNN = neighbors.KNeighborsClassifier(n_neighbors=i, weights="distance")
        KNN.fit(X_train, Y_train)
        Y_predict = KNN.predict(X_test)
        score.append(accuracy_score(Y_test, Y_predict))
        i += 2

    plot_x = [3, 5, 7, 9]
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(plot_x, score, color='#548235', linestyle='-', marker='.', label='accuracy')
    plt.xlabel('K')
    # plt.ylabel('accuracy')
    plt.legend()
    # plt.savefig(os.path.dirname(__file__) + '/image/Classifier.png')
    plt.show()
    """

    # 生成图像
    plot_x = ['KNN', 'NB', 'SVM', 'RF', 'DR', 'LR']
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(plot_x, score, color='#548235', linestyle='-', marker='.', label='accuracy')
    plt.xlabel('Classifier')
    plt.ylabel('accuracy')
    for a, b in zip(plot_x, score):
        plt.text(a, b + 0.001, '%.2f' % b, ha='center', va='bottom')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(plot_x, run_time, color='#305496', linestyle='-', marker='.', label='run_time')
    plt.xlabel('Classifier')
    plt.ylabel('run_time')
    for a, b in zip(plot_x, run_time):
        plt.text(a, b + 0.001, '%.2f' % b, ha='center', va='bottom')
    plt.subplots_adjust(hspace=0.5)
    # plt.savefig(os.path.dirname(__file__) + '/image/Classifier.png')
    plt.show()


def get_yvec_xmat_vnames(target, df):

    yvec = df[target]

    # 将拥有n个不同数值的变量转换为n个0/1的变量，变量名字中有"_isDummy_"作为标注
    xmat = pd.get_dummies(df.loc[:, df.columns != target], prefix_sep = "_isDummy_")

    vnames = xmat.columns

    return yvec, xmat, vnames


if __name__ == '__main__':
    # 分析聚类数据，得到聚类表示意义
    # clusterAnalyze('Usage')

    # dataClassify()

    data = pd.read_excel(os.path.dirname(__file__) + '/data/classify_data.xlsx', sheet_name='Sheet6')
    data.head()
    # print(data.dtypes)

    data.IfUnlimited = data.IfUnlimited.astype(bool)
    data.IfCoalesce = data.IfCoalesce.astype(bool)
    data.IfContract = data.IfContract.astype(bool)
    data.IfSMSSilence = data.IfSMSSilence.astype(bool)
    data.Label = data.Label.astype(str)

    yvec, xmat, vnames = get_yvec_xmat_vnames("Label", data)
    dt = tree.DecisionTreeClassifier(criterion='entropy')
    dt.fit(xmat, yvec)

    dot_data = tree.export_graphviz(dt, feature_names=vnames, filled=True, class_names=yvec)
    graph = graphviz.Source(dot_data)

    # graph.save(os.path.dirname(__file__) + '/image/classify.gv')
    graph.view()

