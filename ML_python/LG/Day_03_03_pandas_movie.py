# Day_03_03_pandas_movie.py

# 이 책 꼭 보세요!!
# 파이썬 라이브러리를 활용한 데이터 분석(수정보완판)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.width', 1000)


def get_movies():
    users = pd.read_csv('ml-1m/users.dat',
                        header=None, sep='::', engine='python',
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    movies = pd.read_csv('ml-1m/movies.dat',
                        header=None, sep='::', engine='python',
                        names=['MovieID', 'Title', 'Genres'])
    ratings = pd.read_csv('ml-1m/ratings.dat',
                        header=None, sep='::', engine='python',
                        names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    # print(movies)

    data = pd.merge(pd.merge(ratings, users), movies)
    # print(data.head())
    # print(data.columns)

    return data


def basic_usage():
    data = get_movies()

    t1 = data.pivot_table(values='Rating', columns='Gender')
    print(t1)

    t2 = data.pivot_table(values='Rating', columns='Gender', index='Age')
    print(t2)

    t2.index = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    print(t2)

    # 문제
    # 연령대별 성별 평점 결과를 막대 그래프로 그려 보세요.
    # t2.plot(kind='bar')
    # plt.show()

    # t3 = data.pivot_table(values='Rating', columns=['Gender', 'Occupation']) # error.
    t3 = data.pivot_table(values='Rating', index=['Gender', 'Occupation'])
    print(t3)

    t4 = data.pivot_table(values='Rating', index='Age', columns=['Gender', 'Occupation'])
    print(t4.head(-3))

    t5 = data.pivot_table(values='Rating', index='Age',
                          columns=['Gender', 'Occupation'], fill_value=0)
    print(t5.head(-3))

    t7 = data.pivot_table(values='Rating', index=['Gender', 'Age'])
    print(t7)
    print(t7.unstack())             # index -> column
    print(t7.unstack().unstack())
    print(t7.unstack().stack())     # column -> index

    t8 = data.pivot_table(values='Rating', index='Age', columns='Gender',
                          aggfunc='sum')
    print(t8)

    t9 = data.pivot_table(values='Rating', index='Age', columns='Gender',
                          aggfunc=[np.mean, np.sum])
    print(t9)

    t9_1 = data.pivot_table(values='Rating', index='Age', columns='Gender',
                          aggfunc=np.mean)
    print(t9_1)

    t9_2 = data.pivot_table(values='Rating', index='Age', columns='Gender',
                          aggfunc=np.sum)
    print(t9_2)

    print(pd.concat([t9_1, t9_2]))
    print(pd.concat([t9_1, t9_2], axis=1))


def groupby_usage(data):
    count_by_title = data.groupby('Title').size()
    print(count_by_title.head())
    print(count_by_title.sum())

    # temp = (count_by_title >= 500)
    # print(temp)

    count_by_title_500 = count_by_title[count_by_title >= 500]
    print(count_by_title_500.head())

    index_over_500 = count_by_title.index[count_by_title >= 500]
    print(index_over_500)
    print(index_over_500.values)

    return index_over_500


data = get_movies()

by_gender = pd.DataFrame.pivot_table(data, values='Rating',
                                     index='Title', columns='Gender')
# print(by_gender.head())
index_over_500 = groupby_usage(data)

# rating_500 = by_gender.ix[index_over_500]
rating_500 = by_gender.loc[index_over_500]
print(rating_500.head())

# top_female = rating_500.sort_values(by='F')
top_female = rating_500.sort_values(by='F', ascending=False)
print(top_female)
print(top_female.iloc[:5])
print(top_female.index[:5])

rating_500['Diff'] = rating_500['F'] - rating_500['M']
print(rating_500)

female_better = rating_500.sort_values(by='Diff')
print(female_better.head())

rating_500['Dist'] = (rating_500['F'] - rating_500['M']).abs()

far_off = rating_500.sort_values(by='Dist', ascending=False)
print(far_off.head())
print('-' * 50)

# rating_std = data.groupby('Title').std()
rating_std = data.groupby('Title')['Rating'].std()
print(rating_std.head())

rating_std_500 = rating_std.loc[index_over_500]
print(rating_std_500.head())
print(type(rating_std_500))

std_500_sorted = pd.Series.sort_values(rating_std_500)  # unbound method
std_500_sorted = rating_std_500.sort_values()           # bound method
print(std_500_sorted.head())
