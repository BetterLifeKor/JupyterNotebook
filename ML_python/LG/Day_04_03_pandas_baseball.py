# Day_04_03_pandas_baseball.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pd.set_option('display.width', 1000)
pd.options.display.width = 1000

filename = 'MLB World Series Champions_ 1903-2016.xlsx'
champs = pd.read_excel('world-series/' + filename, index_col=0)
print(champs.head())
# print(champs['Wins'])

total_teams = champs['Champion'].unique()
print(total_teams)
print(len(total_teams))
print('-' * 50)

# 문제
# 100승 이상 팀만 출력해 보세요.
over_100_index = (champs.Wins >= 100)
print(over_100_index.head())

over_100 = champs[over_100_index]
print(over_100.head())
print(over_100['Champion'].unique())
print('-' * 50)

# 문제
# 우승팀 전체의 평균 승률은 얼마입니까?
print(champs['WinRatio'].mean())

# 문제
# 뉴욕 양키스의 평균 승률을 구해 보세요.
ny_index = (champs['Champion'] == 'New York Yankees')
print(ny_index.head())

yankees = champs[ny_index]
print(yankees)
print(yankees.WinRatio.mean())

# 뉴욕 양키스가 우승한 최초 연도와 마지막 연도를 구해 보세요.
print(yankees.iloc[0])
print(yankees.iloc[-1])

print(yankees.index[0])
print(yankees.index[-1])
print('-' * 50)

# 문제
# 가장 많이 우승한 상위 5개 팀을 보여 주세요.
by_teams = champs.groupby('Champion').size()
print(by_teams)

sorted_teams = by_teams.sort_values(ascending=False)
print(sorted_teams)
print('-' * 50)

fifth = sorted_teams[4]
print(fifth)

top5_index = (sorted_teams >= fifth)
print(top5_index)

top5 = sorted_teams[top5_index]
print(top5)             # 횟수까지 표시
print(top5.index)       # 팀 이름만 표시
