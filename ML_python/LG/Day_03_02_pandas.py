# Day_03_02_pandas.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/scores.csv')
print(df)
print(df.iloc[0])
print(df.ix[0])
print(df.loc[0])
print(df.iloc[:3])
print('-' * 50)

subjects = ['kor', 'eng', 'mat', 'bio']
# print(df['kor'])
print(df[subjects])
print(df[subjects[::-1]])
print(df.kor)
print('-' * 50)

print(df[subjects].sum())
print(df[subjects].sum(axis=0))
print(df[subjects].sum(axis=1))
print(df[subjects].sum().sum())

# 문제
# 과목별 평균, 학생별 평균을 출력해 보세요.
print(df[subjects].mean(axis=0))    # 과목 평균
print(df[subjects].mean(axis=1))    # 학생 평균
print('-' * 50)

df['sum'] = df[subjects].sum(axis=1)
# df.sum = df[subjects].sum(axis=1) # 동작 안함.
df['avg'] = df['sum'] / len(subjects)
# df['avg'] = df.sum / len(subjects)    # 에러
print(df)

print(df.sort_values('avg'))
print(df.sort_values('avg', ascending=False))
print('-' * 50)

sorted_df = df.sort_values('avg', ascending=False)
sorted_df.index = sorted_df.name
print(sorted_df)
print(sorted_df.index)
print(sorted_df.index.values)
print('-' * 50)

# del sorted_df.name        # error.
del sorted_df['name']
print(sorted_df)
print('-' * 50)


def not_used():
    # sorted_df['avg'].plot()
    # sorted_df['avg'].plot(kind='bar')
    sorted_df['avg'].plot(kind='bar', figsize=(8, 4))

    # x축 레이블 숨김
    ax = plt.axes()
    ax.xaxis.label.set_visible(False)

    plt.show()


print(df['class'] == 1)
c1 = df[df['class'] == 1]
c2 = df[df['class'] == 2]
print(c1)

# 과목 평균
mean_c1 = c1['sum'].mean() / 4
print(mean_c1)

# sorted_df[subjects].plot(kind='bar')
# sorted_df['kor'].plot(kind='bar')

# df[subjects].boxplot()
# 문제
# 1반과 2반 데이터를 boxplot으로 그려 보세요.
plt.figure(1)
c1[subjects].boxplot()
plt.figure(2)
c2[subjects].boxplot()

plt.show()
