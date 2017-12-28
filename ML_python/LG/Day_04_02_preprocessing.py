# Day_04_02_preprocessing.py
from sklearn.preprocessing import (add_dummy_feature, Binarizer, Imputer,
                                   LabelBinarizer, LabelEncoder,
                                   MinMaxScaler)
import numpy as np


def use_add_dummy_feature():
    # 맨 앞의 칼럼에 1을 채워 줍니다. (bias)
    x = [[1, 0],
         [1, 0]]
    print(add_dummy_feature(x))

    x = [[3, 1, 0],
         [9, 1, 0]]
    print(add_dummy_feature(x))

    x = [[1, 0],
         [1, 0]]
    print(add_dummy_feature(x, value=7))

    # 2차원 데이터만 가능. (n_samples, n_features)
    # x = [1, 0]
    # print(add_dummy_feature(x))


def use_Binarizer():
    x =[[1., -1.,  2.],
        [2.,  0.,  0.],
        [0.,  1., -1.]]
    binarizer = Binarizer()
    binarizer.fit(x)
    print(binarizer.transform(x))

    binarizer = Binarizer(threshold=1.5)
    print(binarizer.transform(x))


def use_Imputer():
    # 4 = (1 + 7) / 2
    # 5 = (2 + 3 + 10) / 3
    x = [[1, 2],
         [np.nan, 3],
         [7, 10]]
    # strategy : mean, median, most_freqent
    imp = Imputer(strategy='mean', axis=0)
    imp.fit(x)
    print(imp.transform(x))

    x = [[np.nan, 2],
         [6, np.nan],
         [7, 6]]
    print(imp.transform(x))

    print(imp.statistics_)
    print(imp.missing_values)


def use_LabelBinarizer():
    x = [1, 2, 6, 4, 2]
    lb = LabelBinarizer()
    lb.fit(x)
    print(lb.transform(x))
    print(lb.classes_)

    lb2 = LabelBinarizer(sparse_output=True)
    lb2.fit(x)
    print(lb2.transform(x))

    lb3 = LabelBinarizer(neg_label=-1, pos_label=2)
    lb3.fit(x)
    print(lb3.transform(x))
    print('-' * 50)

    lb4 = LabelBinarizer()
    print(lb.fit_transform(['yes', 'no', 'no', 'yes']))

    x = ['yes', 'no', 'no', 'yes', 'cancel']
    lb5 = LabelBinarizer()
    lb5.fit(x)
    print(lb5.transform(x))
    print(lb5.classes_)
    print('-' * 50)

    inverse_x = lb5.transform(x)
    print(inverse_x)
    print(lb5.inverse_transform(inverse_x))


def use_LabelEncoder():
    x = [2, 1, 2, 6]
    le = LabelEncoder()
    le.fit(x)
    print(le.transform(x))
    print(le.classes_)

    inverse_x = le.transform(x)
    print(inverse_x)

    print(le.inverse_transform(inverse_x))
    print('-' * 50)

    x = ['paris', 'tokyo', 'paris', 'amsterdam']
    le2 = LabelEncoder()
    le2.fit(x)
    print(le2.classes_)

    inverse_x = le2.transform(x)
    print(inverse_x)
    print(le2.inverse_transform(inverse_x))
    print('-' * 50)

    print(le2.inverse_transform([0, 0, 1, 1, 2, 2]))


def use_MinMaxScaler():
    x = [[1., -1.,  2.],
         [2.,  0.,  0.],
         [0.,  1., -1.]]
    scaler = MinMaxScaler()
    scaler.fit(x)
    print(scaler.transform(x))
