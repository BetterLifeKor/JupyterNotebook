# Day_03_05_sklearn.py
from sklearn import datasets, svm, random_projection
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import numpy as np
import pickle
from sklearn import model_selection
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


def not_used_1():
    iris = datasets.load_iris()
    print(type(iris))           # <class 'sklearn.utils.Bunch'>
    print(iris.keys())

    # ['data', 'target', 'target_names', 'DESCR', 'feature_names']
    print(iris['target_names'])
    print(iris['feature_names'])
    print(iris['data'])
    print(iris['target'])
    print(iris['DESCR'])

    print(type(iris['data']))   # <class 'numpy.ndarray'>


def outline(number):
    number = number.reshape(-1, 8)

    for row in number:
        for col in row:
            ch = 1 if col > 0 else ' '
            print(ch, end=' ')
        print()


def not_used_2():
    digits = datasets.load_digits()
    print(digits.keys())
    # ['data', 'target', 'target_names', 'images', 'DESCR']

    print(digits.data)
    print(digits.data.shape)        # (1797, 64)
    print(digits.data[0])
    print(digits.data[0].reshape(-1, 8))

    for i in range(10):
        print(digits.target[i])
        outline(digits.data[i])
        print('-' * 50)

    print(digits.images.shape)      # (1797, 8, 8)
    print(digits.data.shape)        # (1797, 64)


def not_used_3():
    digits = datasets.load_digits()

    clf = svm.SVC(gamma=0.001, C=100.)
    # fit = clf.fit(digits.data[:-1], digits.target[:-1])
    # print(clf)
    # print(fit)

    clf.fit(digits.data[:-1], digits.target[:-1])   # training

    pred = clf.predict(digits.data[-1:])
    print(pred)
    print(digits.target[-1])

    # ------------------------- #

    s = pickle.dumps(clf)
    clf2 = pickle.loads(s)
    print(clf2.predict(digits.data[-1:]))


def not_used_4():
    digits = datasets.load_digits()

    train_count = int(len(digits.data) * 0.8)

    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(digits.data[:train_count], digits.target[:train_count])

    y_hat = clf.predict(digits.data[train_count:])
    label = digits.target[train_count:]
    print(y_hat)
    print(label)
    print(y_hat == label)
    print(np.mean(y_hat == label))


def not_used_5():
    # 문제
    # iris 데이터셋을 svm에 적용해 봅니다.
    # 전체 데이터셋으로 학습하고, 처음 3개에 대해서 예측해 봅니다.
    iris = datasets.load_iris()

    clf = svm.SVC(gamma=0.001, C=100.)

    # 정수 레이블 -> 정수 결과
    clf.fit(iris.data, iris.target)
    print(clf.predict(iris.data[:3]))
    print(iris.target[:3])

    print(iris['target_names'])
    print(iris['target'])
    # print(iris['target_names'][iris['target']])
    print(iris.target_names[iris.target])

    # 문자열 레이블 -> 문자열 결과
    clf.fit(iris.data, iris.target_names[iris.target])
    print(clf.predict(iris.data[:3]))


def not_used_6():
    digits = datasets.load_digits()

    data = model_selection.train_test_split(digits.data, digits.target,
                                            train_size=0.7)
    print(len(data))

    train_x, test_x, train_y, test_y = data
    print(train_x.shape, test_x.shape)      # (1257, 64) (540, 64)
    print(train_y.shape, test_y.shape)      # (1257,) (540,)

    data = model_selection.train_test_split(digits.data, digits.target,
                                            train_size=1300)
    train_x, test_x, train_y, test_y = data
    print(train_x.shape, test_x.shape)      # (1300, 64) (497, 64)
    print(train_y.shape, test_y.shape)      # (1300,) (497,)

    clf = svm.SVC(gamma=0.001, C=100.)

    clf.fit(train_x, train_y)
    y_hat = clf.predict(test_x)

    print(y_hat)
    print(test_y)
    print(np.mean(y_hat == test_y))


iris = datasets.load_iris()

df = pd.DataFrame(iris.data,
                  columns=iris.feature_names)
print(df)

# scatter_matrix(df)
# scatter_matrix(df, c=iris.target)
scatter_matrix(df, c=iris.target, hist_kwds={'bins': 20})
plt.show()
