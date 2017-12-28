# Day_03_06_pandas_baby.py
import pandas as pd
import matplotlib.pyplot as plt

# 문제
# yob1880.txt 파일로부터 아래 문제를 해결하세요.
# 1. 남자와 여자 이름 갯수
# 2. 남자와 여자 이름 갯수 합계
# 3. 남자 또는 여자 top5 막대 그래프

names = pd.read_csv('Data/yob1880.txt',
                    header=None,
                    names=['Name', 'Gender', 'Births'])
# print(names)

men = (names.Gender == 'M')
women = (names.Gender == 'F')

# print(men)
print(men.count())
print(men.sum(), women.sum())

# print(names[names.Gender == 'M'])
print(names[names.Gender == 'M'].count())
print(len(names[names.Gender == 'M']))
print('-' * 50)

# 1번
by_gender = names.groupby('Gender').size()
print(by_gender)

# 2번
print(names.groupby('Gender').sum())

# 3번
men_only = names[names.Gender == 'M']
print(men_only[:5])

top5 = men_only[:5]
print(top5)

top5.index = top5.Name
del top5['Name']

print(top5)
print('-' * 50)

# top5.plot(kind='bar')
# plt.show()

# 문제
# 남자와 여자 이름이 같은 데이터를 찾아 보세요.
by_count = names.groupby('Name').size()
print(by_count)

over_one = by_count.index[by_count > 1]
print(over_one)

by_names = names.pivot_table('Births', index='Name', columns='Gender')
print(by_names.head())
print('-' * 50)

print(by_names.loc[over_one])
