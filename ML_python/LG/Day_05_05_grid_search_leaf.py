# Day_05_05_grid_search_leaf.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# 문제
# leaf.csv 파일을 사용해서
# svm을 적용한 그리드서치+교차검증을 구현해 보세요.
# 마지막에는 히트맵도 그려 봅니다.

def draw_heatmap(scores, param_grid):
    ax = plt.gca()
    img = ax.pcolor(scores, cmap='viridis')
    img.update_scalarmappable()
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    ax.set_xticks(np.arange(len(param_grid['gamma'])) + 0.5)
    ax.set_yticks(np.arange(len(param_grid['C'])) + 0.5)
    ax.set_xticklabels(param_grid['gamma'])
    ax.set_yticklabels(param_grid['C'])
    ax.set_aspect(1)

    for p, color, value in zip(img.get_paths(),
                               img.get_facecolors(), img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        c = 'k' if np.mean(color[:3]) > 0.5 else 'w'
        ax.text(x, y, '{:.2f}'.format(value), color=c,
                ha='center', va='center')


leaf = pd.read_csv('Data/leaf.csv')

le = LabelEncoder().fit(leaf.species)

label = le.transform(leaf.species)
leaf = leaf.drop(['id', 'species'], axis=1)

data = train_test_split(leaf, label,
                        train_size=0.7, test_size=0.3, random_state=42)
x_train, x_test, y_train, y_test = data

# 스케일링
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# ---------------------------- #

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(SVC(), param_grid=param_grid, cv=3)
grid_search.fit(x_train, y_train)

print('test score :', grid_search.score(x_test, y_test))
print('best score :', grid_search.best_score_)
print('best param :', grid_search.best_params_)
# [before]
# test score : 0.929292929293
# best score : 0.903318903319
# best param : {'C': 100, 'gamma': 1}

# [after]
# test score : 0.989898989899
# best score : 0.979797979798
# best param : {'C': 100, 'gamma': 0.001}

# ---------------------------- #

results = pd.DataFrame(grid_search.cv_results_)
scores  = np.array(results.mean_test_score).reshape(6, 6)

draw_heatmap(scores, param_grid)
plt.show()








