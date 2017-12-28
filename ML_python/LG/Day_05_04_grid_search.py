# Day_05_04_grid_search.py
from sklearn.datasets import load_iris
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV)
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.width', 1000)


def simple_grid_search(x_train, x_test, y_train, y_test):
    best_score, best_parameter = 0, {}
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            clf = SVC(gamma=gamma, C=C)
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)

            if best_score < score:
                best_score = score
                best_parameter = {'gamma': gamma, 'C': C}

    print('best score :', best_score)
    print('best param :', best_parameter)
    # best score : 1.0
    # best param : {'gamma': 0.01, 'C': 100}


def better_grid_search(x_total, x_test, y_total, y_test):
    x_train, x_valid, y_train, y_valid = train_test_split(x_total,
                                                          y_total,
                                                          random_state=0)
    print(x_train.shape, x_valid.shape, x_test.shape)
    # (84, 4) (28, 4) (38, 4)

    best_score, best_parameter = 0, {}
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            clf = SVC(gamma=gamma, C=C)
            clf.fit(x_train, y_train)
            score = clf.score(x_valid, y_valid)

            if best_score < score:
                best_score = score
                best_parameter = {'gamma': gamma, 'C': C}

    # clf = SVC(gamma=best_parameter['gamma'], C=best_parameter['C'])
    clf = SVC(**best_parameter)
    clf.fit(x_total, y_total)
    score = clf.score(x_test, y_test)

    print('test score :', score)
    print('best score :', best_score)
    print('best param :', best_parameter)
    # test score : 0.921052631579
    # best score : 0.964285714286
    # best param : {'gamma': 0.001, 'C': 100}


def cv_grid_search(x_train, x_test, y_train, y_test):
    best_score, best_parameter = 0, {}
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            clf = SVC(gamma=gamma, C=C)

            scores = cross_val_score(clf, x_train, y_train, cv=5)

            if best_score < scores.mean():
                best_score = scores.mean()
                best_parameter = {'gamma': gamma, 'C': C}

    clf = SVC(**best_parameter)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)

    print('test score :', score)
    print('best score :', best_score)
    print('best param :', best_parameter)
    # test score : 0.947368421053
    # best score : 0.974275362319
    # best param : {'gamma': 1, 'C': 1}


def grid_search_cv(x_train, x_test, y_train, y_test):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

    grid_search = GridSearchCV(SVC(), param_grid=param_grid)
    grid_search.fit(x_train, y_train)

    print('test score :', grid_search.score(x_test, y_test))
    print('best score :', grid_search.best_score_)
    print('best param :', grid_search.best_params_)
    # test score : 0.973684210526
    # best score : 0.964285714286
    # best param : {'C': 100, 'gamma': 0.01}

    return grid_search, param_grid


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


def draw_bad_heatmap(x_train, y_train):
    plt.figure(2, figsize=(12, 5))

    param_linear = {'C': np.linspace(1, 2, 6), 'gamma': np.linspace(1, 2, 6)}
    param_onelog = {'C': np.linspace(1, 2, 6), 'gamma': np.logspace(-3, 2, 6)}
    param_range  = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-7, -2, 6)}

    for i, param_grid in enumerate([param_linear, param_onelog, param_range]):
        grid_search = GridSearchCV(SVC(), param_grid, cv=5)
        grid_search.fit(x_train, y_train)
        scores = grid_search.cv_results_['mean_test_score'].reshape(6, 6)

        plt.subplot(1, 3, i+1)
        draw_heatmap(scores, param_grid)


def cv_pandas_heatmap(x_train, x_test, y_train, y_test):
    grid_search, param_grid = grid_search_cv(x_train, x_test,
                                             y_train, y_test)
    results = pd.DataFrame(grid_search.cv_results_)
    # print(results)
    print(results.head().T)

    scores = np.array(results.mean_test_score).reshape(6, 6)
    print(scores)

    plt.figure(1)
    draw_heatmap(scores, param_grid)

    plt.figure(2)
    draw_bad_heatmap(x_train, y_train)
    plt.tight_layout()
    plt.show()


# 비대칭 매개변수 그리드서치
def different_params(x_train, x_test, y_train, y_test):
    param_grid = [{'kernel': ['rbf'],
                   'C': [0.001, 0.01, 0.1, 1, 10, 100],
                   'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
                  {'kernel': ['linear'],
                   'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}]

    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    print('test score :', grid_search.score(x_test, y_test))
    print('best score :', grid_search.best_score_)
    print('best param :', grid_search.best_params_)
    # test score : 0.973684210526
    # best score : 0.973214285714
    # best param : {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}


iris = load_iris()
data = train_test_split(iris.data, iris.target, random_state=0)
x_train, x_test, y_train, y_test = data

# simple_grid_search(*data)
# better_grid_search(*data)
# cv_grid_search(*data)
# grid_search_cv(*data)
# cv_pandas_heatmap(*data)
different_params(*data)

# RandomizedSearchCV
