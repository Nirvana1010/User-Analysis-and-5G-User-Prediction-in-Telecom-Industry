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
from sklearn.base import is_classifier # determine the desition tree is used for classfication / regression
from dtreeviz.colors import adjust_colors # determine color of decision tree
import seaborn as sns # determine color of decision tree
from matplotlib.colors import Normalize # RGB normalization
import matplotlib.cm as cm
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'  # Graphviz


UsageColumn = ['TotalFlow', '4GDuration', '4GFlow', 'TotalCallDuration', 'CallingDuration', 'TotalRevenue',
               'Registration']
ArrearageColumn = ['TotalArrearage', 'MaxArrearageTime', 'Registration']
features = ['IfUnlimited', 'IfCoalesce', 'IfContract', 'IfSMSSilence', 'TotalFlow', '4GDuration',
            '4GFlow', 'TotalCallDuration', 'CallingDuration', 'TotalRevenue', 'TotalArrearage',
            'MaxArrearageTime', 'Registration']


def clusterAnalyze(cluster_name):
    # load data
    global result
    global columns
    data = pd.read_excel(os.path.dirname(__file__) + '/data/classify_data.xlsx', sheet_name='Sheet3')
    data.head()

    num = 0
    # create dataframe
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

    # mean value for each cluster
    for i in range(num):
        temp = data.where(data[cluster_name] == i)
        means = []
        for col in columns:
            means.append(temp[col].mean())
        result.loc[i] = means

    # set precision
    result.round(2)
    result.to_excel(os.path.dirname(__file__) + '/data/Mean%s.xlsx' % cluster_name)
    print(result)


def dataClassify():
    # Sheet6 -> cluster datas
    data = pd.read_excel(os.path.dirname(__file__) + '/data/classify_data.xlsx', sheet_name='Sheet6')
    data.head()

    data_X = data[features]  # attributes
    data_Y = data['Clusters']  # labels

    m, n = data_X.shape

    score = []
    run_time = []

    # Classifiers: KNN、NB、SVM、RF、DT, LR
    KNN = neighbors.KNeighborsClassifier(n_neighbors=5, weights="distance")
    NB = naive_bayes.GaussianNB()
    SVM = svm.SVC()
    RF = ensemble.RandomForestClassifier()
    DR = tree.DecisionTreeClassifier(criterion='entropy')
    LR = linear_model.LogisticRegression()

    # split data into training and test set
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data_X, data_Y, train_size=0.7, random_state=0)

    # KNN
    start = time.time()
    KNN.fit(X_train, Y_train)
    Y_predict = KNN.predict(X_test)
    end = time.time()
    score.append(accuracy_score(Y_test, Y_predict))
    run_time.append(end-start)

    # Naive Bayes
    start = time.time()
    NB.fit(X_train, Y_train)
    Y_predict = NB.predict(X_test)
    end = time.time()
    score.append(accuracy_score(Y_test, Y_predict))
    run_time.append(end - start)

    # SVM
    start = time.time()
    SVM.fit(X_train, Y_train)
    Y_predict = SVM.predict(X_test)
    end = time.time()
    score.append(accuracy_score(Y_test, Y_predict))
    run_time.append(end - start)

    # Random Forest
    start = time.time()
    RF.fit(X_train, Y_train)
    Y_predict = RF.predict(X_test)
    end = time.time()
    score.append(accuracy_score(Y_test, Y_predict))
    run_time.append(end - start)

    # Decision Tree
    start = time.time()
    DR.fit(X_train, Y_train)
    Y_predict = DR.predict(X_test)
    end = time.time()
    score.append(accuracy_score(Y_test, Y_predict))
    run_time.append(end - start)

    # Logistic Regression
    start = time.time()
    LR.fit(X_train, Y_train)
    Y_predict = LR.predict(X_test)
    end = time.time()
    score.append(accuracy_score(Y_test, Y_predict))
    run_time.append(end - start)
    """

    # Determine K-value for KNN
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

    # comparison between classifiers
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

    # Dummy encoding
    xmat = pd.get_dummies(df.loc[:, df.columns != target], prefix_sep = "_isDummy_")

    vnames = xmat.columns

    return yvec, xmat, vnames


if __name__ == '__main__':
    # Clustering results analysis
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

