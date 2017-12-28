# Day_02_04_pandas.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def use_Series():
    s = pd.Series([5, 1, 2, 9])
    print(s)

    print(s.values)
    print(s.index)
    print('-' * 50)

    s2 = pd.Series([5, 1, 2, 9],
                   index=['a', 'b', 'c', 'd'])
    print(s2)

    print(s2.values)
    print(s2.index)

    print(s2[0], s2[1], s2[-2], s2[-1])
    print(s2['a'], s2['b'])
    print('-' * 50)

    s3 = pd.Series({'a': 5, 'b': 1, 'c': 2, 'd': 9})
    print(s3)


def not_used():
    # 컴프리헨션
    print([i for i in range(5)])
    print([0 for i in range(5)])
    print([[i] for i in range(5)])
    a = list(range(10))
    print(a)
    print([i for i in a])
    print([i for i in a if i % 2 == 1])
    print([i*j for i in a if i % 2 == 1 for j in range(3)])


def use_names():
    baby_names = ['Bob', 'Jessica', 'Mary', 'John', 'Mel']

    np.random.seed(1)
    names = [baby_names[np.random.randint(5)] for _ in range(1000)]
    births = [np.random.randint(1000) for _ in range(1000)]

    # baby_set = [(baby_names[np.random.randint(5)], np.random.randint(1000))
    #             for _ in range(1000)]

    baby_set = list(zip(names, births))
    print(baby_set)

    df = pd.DataFrame(baby_set,
                      columns=['Name', 'Births'])
    print(df)
    print(df.index)
    print(df.values)
    print(df.columns)
    print('-' * 50)

    df.info()

    print(df.head())
    print(df.head(3))
    print('-' * 50)

    print(df.tail())
    print('-' * 50)

    print(df['Name'].unique())

    name_by = df.groupby('Name')
    print(name_by)
    print(name_by.sum())

    # name_by.sum().plot(kind='bar')
    # plt.show()

    print(name_by.size())

    name_by.size().plot()
    plt.ylim(0, 300)
    plt.show()


df = pd.DataFrame({'state': ['ohio', 'ohio', 'ohio', 'nevada', 'nevada', 'nevada'],
                   'year': [2000, 2001, 2002, 2000, 2001, 2002],
                   'population': [1.5, 1.7, 3.6, 2.4, 2.9, 2.8]})
print(df)
print('-' * 50)

print(df.index)

df.index = ['one', 'two', 'three', 'four', 'five', 'six']
print(df.index)
print(df)
print('-' * 50)

print(df['population'])
print(type(df['population']))
print(df['population'][2])
print(df['population']['three'])

print(df.population)
print(df.population[2])
print('-' * 50)

print(df.iloc[2])
print(df.ix[2])
print('-' * 50)

print(df.loc['three'])
print(df.ix['three'])
print('-' * 50)

print(df)
print(df.iloc[1:3])
print(df.loc['two':'four'])
