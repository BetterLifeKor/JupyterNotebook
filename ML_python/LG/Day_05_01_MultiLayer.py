# Day_05_01_MultiLayer.py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.keys())
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])

print('max by feature')
print(cancer.data.max(axis=0))      # 수직 방향. 열 우선.
print(cancer.target)                # 0 또는 1
print()

x_train, x_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state=0)
# 레이어 : 기본 100개
mlp1 = MLPClassifier(random_state=42)
mlp1.fit(x_train, y_train)

print('train :', mlp1.score(x_train, y_train))
print(' test :', mlp1.score(x_test, y_test))
# train : 0.906103286385
#  test : 0.881118881119

# ----------------------------------- #

# standardization.
mean_on_train = x_train.mean(axis=0)
std_on_train = x_train.std(axis=0)

x_train_scaled = (x_train - mean_on_train) / std_on_train
x_test_scaled = (x_test - mean_on_train) / std_on_train

# max_iter은 기본 200회.
mlp2 = MLPClassifier(random_state=0)
mlp2.fit(x_train_scaled, y_train)

print(' iter :', mlp2.max_iter)
print('train :', mlp2.score(x_train_scaled, y_train))
print(' test :', mlp2.score(x_test_scaled, y_test))
#  iter : 200
# train : 0.990610328638
#  test : 0.965034965035

# ----------------------------------- #

# alpha 기본 값은 0.0001.
mlp3 = MLPClassifier(random_state=0, max_iter=1000)
mlp3.fit(x_train_scaled, y_train)

print(' iter :', mlp3.max_iter)
print('alpha :', mlp3.alpha)
print('train :', mlp3.score(x_train_scaled, y_train))
print(' test :', mlp3.score(x_test_scaled, y_test))
#  iter : 1000
# alpha : 0.0001
# train : 0.992957746479
#  test : 0.972027972028

# ----------------------------------- #

mlp4 = MLPClassifier(random_state=0, max_iter=1000, alpha=1)
mlp4.fit(x_train_scaled, y_train)

print(' iter :', mlp4.max_iter)
print('alpha :', mlp4.alpha)
print('train :', mlp4.score(x_train_scaled, y_train))
print(' test :', mlp4.score(x_test_scaled, y_test))
#  iter : 1000
# alpha : 1
# train : 0.988262910798
#  test : 0.972027972028

# ----------------------------------- #

print(len(mlp4.coefs_))         # 2개
print(mlp4.coefs_[0].shape)     # (30, 100) weight
print(mlp4.coefs_[1].shape)     # (100, 1)  bias

# ----------------------------------- #

plt.figure(figsize=(20, 5))
ax = plt.gca()
im = ax.imshow(mlp4.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel('hidden units')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.1)
plt.colorbar(im, cax)
plt.show()
