# Day_04_05_scaling.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   Normalizer, RobustScaler)

x, y = datasets.make_blobs(n_samples=50, centers=2,
                           cluster_std=1, random_state=4)
print(x.shape, y.shape)
print(x[:5])
print(y)

# 데이터를 오른쪽으로 이동
# x += 3

# Normalizer 클래스를 확인하기 위해.
x -= 3

plt.figure(1)
plt.scatter(x[:, 0], x[:, 1], c=y, s=60, edgecolors='black')

max_x = np.abs(x[:, 0]).max()
max_y = np.abs(x[:, 1]).max()

plt.xlim(-max_x-1, max_x+1)
plt.ylim(-max_y-1, max_y+1)
plt.title('original')

ax = plt.axes()
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# -------------------- #

plt.figure(2)

for i, scaler in enumerate([StandardScaler(), RobustScaler(),
                            MinMaxScaler(), Normalizer(norm='l2')]):
    x_fit = scaler.fit_transform(x)

    ax = plt.subplot(2, 2, i+1)
    plt.scatter(x_fit[:, 0], x_fit[:, 1], c=y, s=60, edgecolors='black')

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(type(scaler).__name__)

# for ax in plt.figure(2).axes:
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

plt.show()
