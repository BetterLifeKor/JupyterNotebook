# Day_05_03_cross_validation.py
from sklearn.datasets import make_blobs, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     KFold, LeaveOneOut, ShuffleSplit,
                                     GroupKFold)
import numpy as np


def simple_test():
    x, y = make_blobs(random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    print('score :', logreg.score(x_test, y_test))
    # score : 0.88


def cross_validation():
    iris = load_iris()
    logreg = LogisticRegression()

    # 회귀 : KFold
    # 분류 : StratifiedKFold
    # cv는 기본 3번
    scores = cross_val_score(logreg, iris.data, iris.target)
    print('scores (3-folds) :', scores)
    # scores (3-folds) : [ 0.96078431  0.92156863  0.95833333]

    scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
    print('scores (5-folds) :', scores)
    print('  mean (5-folds) :', scores.mean())
    # scores (5-folds) : [ 1.          0.96666667  0.93333333  0.9         1.        ]
    #   mean (5-folds) : 0.96


def usage_KFold():
    iris = load_iris()

    # 분할 갯수. 반복 횟수는 분할 갯수와 동일.
    sp1 = KFold()
    for train_index, test_index in sp1.split(iris.data, iris.target):
        print(len(train_index), len(test_index))    # 100 50
    print()

    sp2 = KFold(n_splits=5)
    for train_index, test_index in sp2.split(iris.data, iris.target):
        print(len(train_index), len(test_index))    # 120, 30
    print()

    # fold 순서
    sp3 = KFold()
    for train_index, test_index in sp3.split(iris.data, iris.target):
        print(test_index[:10])
    print()
    # [0 1 2 3 4 5 6 7 8 9]
    # [50 51 52 53 54 55 56 57 58 59]
    # [100 101 102 103 104 105 106 107 108 109]

    sp4 = KFold(n_splits=5)
    for train_index, test_index in sp4.split(iris.data, iris.target):
        print(test_index[:10])
        # print(iris.data[test_index])      # 데이터를 가져오는 방법
        # print(iris.target[test_index])
    print()
    # [0 1 2 3 4 5 6 7 8 9]
    # [30 31 32 33 34 35 36 37 38 39]
    # [60 61 62 63 64 65 66 67 68 69]
    # [90 91 92 93 94 95 96 97 98 99]
    # [120 121 122 123 124 125 126 127 128 129]

    # KFold 내부구조 확인
    sp5 = KFold()
    folds = list(sp5.split(iris.data, iris.target))
    # print(folds)
    print(len(folds))       # 3. n_splits가 3이니까.
    # print(folds[0])
    print(len(folds[0]))    # 2. train과 test의 2개니까.

    train, test = folds[0]
    print(len(train))       # 100. 150개를 3개로 나눠서 2개 사용.
    print(len(test))        # 50.                     1개 사용.


def cv_detail():
    iris = load_iris()
    logreg = LogisticRegression()

    print('n_splits : 3')
    print(cross_val_score(logreg, iris.data, iris.target, cv=KFold()))
    # cv=3       : [ 0.96078431  0.92156863  0.95833333]
    # cv=KFold() : [ 0.  0.  0.]
    # 데이터가 50개씩 나눠져 있기 때문에 나쁜 결과.

    print('n_splits : 5')
    print(cross_val_score(logreg, iris.data, iris.target, cv=KFold(n_splits=5)))
    # n_splits : 5
    # [ 1.          0.93333333  0.43333333  0.96666667  0.43333333]

    print('n_splits : 3 shuffle')
    print(cross_val_score(logreg, iris.data, iris.target,
                          cv=KFold(shuffle=True, random_state=0)))
    # n_splits : 3 shuffle
    # [ 0.9   0.96  0.96]

    print('n_splits : loocv')
    loocv = cross_val_score(logreg, iris.data, iris.target,
                            cv=LeaveOneOut())
    # print(loocv)
    print('n_splits :', len(loocv))
    print('   score :', loocv.mean())
    # n_splits : 150
    #    score : 0.953333333333

    print('n_splits : 150')
    loocv = cross_val_score(logreg, iris.data, iris.target,
                            cv=KFold(n_splits=150))
    print(loocv)
    print('n_splits :', len(loocv))
    print('   score :', loocv.mean())
    # n_splits : 150
    #    score : 0.953333333333


def cv_shuffle_split():
    iris = load_iris()
    logreg = LogisticRegression()

    # sp = ShuffleSplit(test_size=0.5, train_size=0.5, n_splits=10,
    #                   random_state=0)
    # sp = ShuffleSplit(test_size=0.5, n_splits=10, random_state=0)
    sp = ShuffleSplit(train_size=0.5, n_splits=10, random_state=0)

    scores = cross_val_score(logreg, iris.data, iris.target, cv=sp)
    print(scores)
    print('mean :', scores.mean())
    # [ 0.84        0.93333333  0.90666667  1.          0.90666667  0.93333333
    #   0.94666667  1.          0.90666667  0.88      ]
    # mean : 0.925333333333

    # tr
    print('train size :', sp.train_size)
    print(' test size :', sp.test_size)
    # train size : None
    #  test size : 0.5
    # train size : 0.5
    #  test size : default


def usage_ShuffleSplit():
    iris = load_iris()

    sp1 = ShuffleSplit(train_size=0.6, test_size=0.4, n_splits=3)
    for train_index, test_index in sp1.split(iris.data, iris.target):
        print(len(train_index), len(test_index))    # 90 60
    print()

    # test_size를 전달하지 않으면 기본 15개 사용
    sp2 = ShuffleSplit(train_size=0.6, n_splits=3)
    for train_index, test_index in sp2.split(iris.data, iris.target):
        print(len(train_index), len(test_index))    # 90 15
    print()

    # train_size를 전달하지 않으면 나머지 전부 사용
    sp3 = ShuffleSplit(test_size=0.4, n_splits=3)
    for train_index, test_index in sp3.split(iris.data, iris.target):
        print(len(train_index), len(test_index))    # 90 60
    print()

    # 전체 테스트셋을 모아서 하나로 출력
    total = []
    sp4 = ShuffleSplit(train_size=100, test_size=50, n_splits=3)
    for train_index, test_index in sp4.split(iris.data, iris.target):
        print(test_index[:10])
        total += list(test_index)

    print(len(total))

    # 중복된 숫자들 발생.
    total_sorted = np.sort(total)
    print(total_sorted)


def group_kfold():
    logreg = LogisticRegression()
    x, y = make_blobs(n_samples=12, random_state=0)

    groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
    scores = cross_val_score(logreg, x, y,
                             groups, cv=GroupKFold(n_splits=3))
    print('scores')
    print(scores)
    # scores
    # [ 0.75        0.8         0.66666667]

    # 3개로 분할
    sp1 = GroupKFold()
    for train_indx, test_index in sp1.split(x, y, groups):
        print(train_indx, test_index)
    print()
    # groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
    # [ 0  1  2  7  8  9 10 11] [3 4 5 6]
    # [0 1 2 3 4 5 6] [ 7  8  9 10 11]
    # [ 3  4  5  6  7  8  9 10 11] [0 1 2]

    # 4개로 분할
    sp2 = GroupKFold(n_splits=4)
    for train_indx, test_index in sp2.split(x, y, groups):
        print(train_indx, test_index)
    print()
    # [ 0  1  2  7  8  9 10 11] [3 4 5 6]
    # [0 1 2 3 4 5 6 7 8] [ 9 10 11]
    # [ 3  4  5  6  7  8  9 10 11] [0 1 2]
    # [ 0  1  2  3  4  5  6  9 10 11] [7 8]

    # 종류가 4개밖에 없기 때문에 5개는 불가.
    # sp3 = GroupKFold(n_splits=5)
    # for train_indx, test_index in sp3.split(x, y, groups):
    #     print(train_indx, test_index)


group_kfold()
