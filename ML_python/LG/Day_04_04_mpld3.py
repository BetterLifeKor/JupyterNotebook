# Day_04_04_mpld3.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mpld3


def sample_plot(df, subject):
    df.plot(kind='line', marker='p',
            color=['blue', 'red'], lw=3, ms=20, alpha=0.7)

    plt.title(subject)
    plt.text(s='blue line', x=1, y=2, color='blue')
    plt.text(s='red line', x=2.7, y=3, color='red')


def not_used_1():
    c1 = [1, 2, 3, 4]
    c2 = [1, 4, 2, 3]
    df = pd.DataFrame({'c1': c1, 'c2': c2})

    # sample_plot(df, 'base')
    #
    # plt.xkcd()
    # sample_plot(df, 'xkcd')
    # plt.show()

    sample_plot(df, 'D3.js')
    mpld3.show()


def not_used_2():
    plt.plot([3, 1, 4, 1, 5], 'ks-', mec='w', mew=5, ms=20)
    # plt.show()
    mpld3.show()


fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
N = 100

scatter = ax.scatter(np.random.normal(size=N),
                     np.random.normal(size=N),
                     c=np.random.random(size=N),
                     s=1000 * np.random.random(size=N),
                     alpha=0.3,
                     cmap=plt.cm.jet)
ax.grid(color='white', linestyle='solid')

ax.set_title("Scatter Plot (with tooltips!)", size=20)

labels = ['point {0}'.format(i + 1) for i in range(N)]
tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
mpld3.plugins.connect(fig, tooltip)

mpld3.show()
