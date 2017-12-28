# Day_04_01_knn_regression.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import (model_selection, neighbors,
                     linear_model, preprocessing, metrics)

pd.set_option('display.width', 1000)

df = pd.read_csv('Data/winequality-red.csv', sep=';')
# print(df)

# pd.DataFrame.hist(df, figsize=(20, 9))
# plt.show()

all_quality = df.quality.values
print(all_quality)
print(df.quality.unique())

bad_quality = (all_quality <= 5)
print(bad_quality)

temp = df.groupby('quality').size()
print(temp)
print(' bad :', temp.iloc[:3].sum())
print('good :', temp.iloc[3:].sum())
print(' bad :', temp.loc[:5].sum())
print('good :', temp.loc[6:].sum())


def not_used():
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(all_quality)

    plt.subplot(1, 2, 2)
    plt.hist(bad_quality)
    plt.show()


# (1599, 12) --> (1599, 11) (1599,)
x = df.drop('quality', axis=1).values
y = (df.quality.values > 5)
print(x.shape, y.shape)

# 기본값 (75 : 25)
data = model_selection.train_test_split(x, y, random_state=42)
# data = model_selection.train_test_split(x, y, random_state=42,
#                             train_size=0.25, test_size=0.25)
train_x, test_x, train_y, test_y = data
print(train_x.shape, train_y.shape)     # (1199, 11) (1199,)
print(test_x.shape, test_y.shape)       # (400, 11) (400,)

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(train_x, train_y)
print('score :', knn.score(test_x, test_y))

y_hat = knn.predict(test_x)
# print(y_hat)
print('score :', np.mean(y_hat == test_y))

# ----------------------- #

# 데이터 전처리
x = preprocessing.scale(x)

data = model_selection.train_test_split(x, y, random_state=42)
train_x, test_x, train_y, test_y = data

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(train_x, train_y)
print('score :', knn.score(test_x, test_y))
