# Day_04_06_preprocessing_svm.py
from sklearn import svm, model_selection, datasets, preprocessing


def show_accuracy(seed):

    cancer = datasets.load_breast_cancer()
    data = model_selection.train_test_split(cancer.data, cancer.target,
                                            random_state=seed)
    x_train, x_test, y_train, y_test = data

    clf = svm.SVC(C=100)
    clf.fit(x_train, y_train)
    print('original :', clf.score(x_test, y_test))

    # ----------------- #
    # 0~1 사이로 스케일링
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf.fit(x_train_scaled, y_train)

    print('  minmax :', clf.score(x_test_scaled, y_test))

    # ----------------- #
    # 평균 0, 분산 1로 스케일링
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf.fit(x_train_scaled, y_train)

    print('standard :', clf.score(x_test_scaled, y_test))


import random
for _ in range(10):
    show_accuracy(random.randrange(100))
    print('-' * 50)
