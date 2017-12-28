# Day_01_04_kma.py
import requests
import re
import csv

f = open('Data/kma.csv', 'w', encoding='utf-8', newline='')

url = 'http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108'
received = requests.get(url)
# print(received)
# print(received.text)

# 문제
# 기상청 사이트에서 가져온 데이터로부터
# province와 city, data를 출력해 보세요.

# .+ : 탐욕적
# .+? : 비탐욕적
# re.DOTALL : 여러 줄에 걸쳐 있을 때
locations = re.findall(r'<location wl_ver="3">(.+?)</location>',
                       received.text, re.DOTALL)
# print(len(locations))
# print(locations)

kma = []
for loc in locations:
    # print(loc)

    prov = re.findall(r'<province>(.+)</province>', loc)
    city = re.findall(r'<city>(.+)</city>', loc)
    print(prov, city)

    # area = re.findall(r'<province>(.+)</province>.+?<city>(.+)</city>',
    #                   loc, re.DOTALL)
    # print(area)
    # print(area[0][0], area[0][1])

    data = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)
    # print(len(data))

    for datum in data:
        mode = re.findall(r'<mode>(.+)</mode>', datum)
        tmEf = re.findall(r'<tmEf>(.+)</tmEf>', datum)
        wf   = re.findall(r'<wf>(.+)</wf>', datum)
        tmn  = re.findall(r'<tmn>(.+)</tmn>', datum)
        tmx  = re.findall(r'<tmx>(.+)</tmx>', datum)
        reli = re.findall(r'<reliability>(.+)</reliability>', datum)

        print('{},{},{},{},{},{},{},{}'.format(prov[0], city[1],
                                               mode[0], tmEf[0], wf[0],
                                               tmn[0], tmx[0], reli[0]))
        # print('{},{},{},{},{},{},{},{}'.format(prov[0], city[0],
        #                                        mode[0], tmEf[0], wf[0],
        #                                        tmn[0], tmx[0], reli[0]),
        #       file=f)
        kma.append([prov[0], city[0], mode[0], tmEf[0],
                    wf[0], tmn[0], tmx[0], reli[0]])


# print(*kma, sep='\n')
# csv.writer(f).writerows(kma)

f.close()

# ['제주도'] ['서귀포']
# A02,2017-11-24 00:00,구름많고 비,9,15,보통
# A02,2017-11-24 12:00,구름많음,9,15,낮음
# --->
# 제주도,서귀포,A02,2017-11-24 00:00,구름많고 비,9,15,보통
# 제주도,서귀포,A02,2017-11-24 12:00,구름많음,9,15,낮음
