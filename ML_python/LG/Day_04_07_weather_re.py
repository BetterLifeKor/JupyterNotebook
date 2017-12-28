# Day_04_07_weather_re.py
import requests
import re

url = 'http://www.kma.go.kr/weather/climate/past_cal.jsp?stn=108&yy=2017&mm=9&x=21&y=7&obs=1'
received = requests.get(url)
# print(received.text)

# 문제
# 평균기온, 최고기온, 최저기온의 3가지만 출력해 보세요.
tbody = re.findall(r'<tbody>(.+?)</tbody>',
                   received.text, re.DOTALL)
# print(tbody)
# print(tbody[0])

trs = re.findall(r'<tr>(.+?)</tr>',
                 tbody[0], re.DOTALL)

# 날짜 행과 데이터 행이 2개씩 배치됨
trs = [tr for tr in trs[1::2]]

days = []
for tr in trs:
    # print(tr)
    tds = re.findall(r'<td class="align_left">(.+?)</td>', tr)
    # print(tds)

    tds = [td for td in tds if td != '&nbsp;']
    # print(tds)

    days += tds

for day in days:
    # print(day)

    items = re.findall(r'[0-9]+.[0-9]', day)
    # print(items)
    print(items[:3])
