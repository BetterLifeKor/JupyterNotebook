# Day_02_02_matplotlib.py
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc, colors, cm
import numpy as np
import csv


def test_1():
    plt.plot([10, 20, 30, 40, 50])
    plt.show()


def test_2():
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    # plt.axis([0, 6, 0, 20])
    plt.xlim(0, 6)
    plt.ylim(0, 20)
    plt.show()


def test_3():
    # 문제
    # x의 범위가 (-10, 10)일 때의 x^2 그래프를 그려 보세요.
    # plt.plot(range(-10, 10), np.arange(-10, 10) ** 2)
    plt.plot(np.arange(-10, 10, 0.1), np.arange(-10, 10, 0.1) ** 2)
    plt.show()


def test_4():
    x = np.linspace(-10, 10, 100)
    y = np.sin(x)

    plt.plot(x, y)
    # plt.plot(x, y, marker='x')
    plt.plot(x, y, 'rx')
    plt.show()


def test_5():
    fig, ax = plt.subplots()
    ax.grid(True)

    x1 = np.arange(0.01, 2, 0.01)
    plt.plot(x1,  np.log(x1), 'r')
    plt.plot(x1, -np.log(x1), 'g')

    x2 = np.arange(0.01-2, 0, 0.01)
    plt.plot(x2,  np.log(-x2), 'b')
    plt.plot(x2, -np.log(-x2), 'k')

    plt.xlim(-2, 2)
    plt.show()


def test_6():
    def func(t):
        return np.exp(-t) * np.cos(2*np.pi*t)

    t1 = np.arange(0, 5, 0.1)
    t2 = np.arange(0, 5, 0.02)

    plt.figure(1)

    plt.subplot(221)
    plt.plot(t1, func(t1), 'bo')
    plt.plot(t2, func(t2), 'k')

    plt.figure(2)

    # plt.subplot(224)
    plt.subplot(2, 1, 2)
    plt.plot(t2, np.cos(2*np.pi*t2), 'r--')

    plt.show()


def test_7():
    # 문제
    # test_5()에서 사용했던 log 그래프를
    # 첫 번째 피겨에 각각 2개, 두 번째 피겨에 각각 2개씩 그립니다.
    # 각각 2개는 따로따로 그린다는 뜻입니다.
    fig = plt.figure(1)

    x1 = np.arange(0.01, 2, 0.01)

    plt.subplot(1, 2, 1)
    fig.gca().grid(True)
    plt.plot(x1,  np.log(x1), 'r')

    plt.subplot(1, 2, 2)
    fig.gca().grid(True)
    plt.plot(x1, -np.log(x1), 'g')

    fig = plt.figure(2)

    x2 = np.arange(0.01-2, 0, 0.01)

    plt.subplot(1, 2, 1)
    fig.gca().grid(True)
    plt.plot(x2,  np.log(-x2), 'b')

    plt.subplot(1, 2, 2)
    fig.gca().grid(True)
    plt.plot(x2, -np.log(-x2), 'k')

    plt.xlim(-2, 2)
    plt.show()


def test_8():
    means_men = (20, 35, 30, 35, 27)
    means_women = (25, 32, 34, 20, 25)

    n_group = len(means_men)
    index = np.arange(n_group)
    bar_width = 0.45
    opacity = 0.4

    plt.bar(index, means_men, bar_width, alpha=opacity,
            color='b', label='Men')
    plt.bar(index + bar_width, means_women, bar_width, alpha=opacity,
            color='g', label='Women')

    plt.xticks(index + bar_width / 2, ('A', 'B', 'C', 'D', 'E'))
    plt.tight_layout()
    plt.show()


def test_9():
    f = open('Data/2016_GDP.txt', 'r', encoding='utf-8')

    f.readline()

    names, money = [], []
    for row in csv.reader(f, delimiter=':'):
        # print(row)
        names.append(row[1])
        money.append(int(row[-1].replace(',', '')))

    f.close()
    # print(money)

    path = 'C:/Windows/Fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=path).get_name()
    print(font_name)
    rc('font', family=font_name)

    top10_names = names[:10]
    top10_money = money[:10]

    index = np.arange(10)

    # plt.bar(index, top10_money)
    # plt.bar(index, top10_money, color=colors.BASE_COLORS)
    plt.bar(index, top10_money, color=colors.TABLEAU_COLORS)
    # plt.bar(index, top10_money, color='rgb')
    # plt.bar(index, top10_money, color=['red', 'green', 'black'])
    # plt.bar(index, top10_money, color=colors.CSS4_COLORS)

    plt.xticks(index, top10_names)
    # plt.xticks(index, top10_names, rotation='vertical')
    # plt.xticks(rotation=270)
    # plt.xticks(rotation=-90)
    plt.xticks(rotation=60)

    plt.title('GDP 상위 10개국')
    # plt.subplots_adjust(left=0.5, right=0.8)
    # plt.subplots_adjust(top=0.8, bottom=0.2)
    plt.subplots_adjust(bottom=0.2)
    plt.show()


test_9()
