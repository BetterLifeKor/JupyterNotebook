# Day_01_02_re.py
import re
import requests


def not_used_1():
    #       아이디       전화번호
    db = '''3412    Bob 123
    3834  Jonny 333
    1248   Kate 634
    1423   Tony 567
    2567  Peter 435
    3567  Alice 535
    1548  Kerry 534'''

    # print(db)

    # r : raw
    temp = re.findall(r'[0-9]', db)
    print(temp)

    numbers = re.findall(r'[0-9]+', db)
    print(numbers)

    # 문제
    # 아이디만 찾아 보세요.
    # 전화번호만 찾아 보세요.
    # ids = re.findall(r'[0-9]{4}', db)
    # ids = re.findall(r'[0-9]{3}', db)
    ids = re.findall(r'^[0-9]+', db, re.MULTILINE)
    print(ids)

    phones = re.findall(r'[0-9]+$', db, re.MULTILINE)
    print(phones)

    # 문제
    # 이름만 찾아 보세요.
    # names = re.findall(r'[A-z]+', db)
    # names = re.findall(r'[A-Za-z]+', db)
    names = re.findall(r'[A-Z][a-z]+', db)
    print(names)

    # 문제
    # T로 시작하는 이름만 찾아 보세요.
    # T로 시작하지 않는 이름만 찾아 보세요.
    tStart = re.findall(r'T[a-z]+', db)
    print(tStart)

    # tNot = re.findall(r'[^T][a-z]+', db)
    # tNot = re.findall(r'[ABCDEFGHIJKLMNOPQRSUVWXYZ][a-z]+', db)
    tNot = re.findall(r'[A-SU-Z][a-z]+', db)
    print(tNot)


def not_used_2():
    url = 'http://www.kma.go.kr/DFSROOT/POINT/DATA/top.json.txt'
    received = requests.get(url)
    print(received)
    print(received.text)

    text = received.content.decode('utf-8')
    print(text)
    print(type(text))

    # 문제
    # 수신 데이터로부터
    # 코드 번호와 도시 이름만 찾아 보세요.
    codes = re.findall(r'[0-9]+', text)
    print(codes)

    cities = re.findall(r'[가-힣]+', text)
    print(cities)

    items = zip(codes, cities)
    print(list(items))

    # [{"code":"11","value":"서울특별시"},{"code":"26","value":"부산광역시"},{"code":"27","value":"대구광역시"},{"code":"28","value":"인천광역시"},{"code":"29","value":"광주광역시"},{"code":"30","value":"대전광역시"},{"code":"31","value":"울산광역시"},{"code":"41","value":"경기도"},{"code":"42","value":"강원도"},{"code":"43","value":"충청북도"},{"code":"44","value":"충청남도"},{"code":"45","value":"전라북도"},{"code":"46","value":"전라남도"},{"code":"47","value":"경상북도"},{"code":"48","value":"경상남도"},{"code":"50","value":"제주특별자치도"}]

    print(re.findall(r'"([0-9]+)"', text))

    items = re.findall(r'{"code":"([0-9]+)","value":"(.+?)"}', text)
    print(items)
    print(len(items))


url = 'http://211.251.214.169:8080/SeatMate_sj/SeatMate.php?classInfo=1'
received = requests.get(url)
print(received)
# print(received.text)

text = received.content.decode('euc-kr')
# print(text)

# 문제
# 빈 자리가 몇 개인지 알려 주세요.
# 번호까지 알려주면 더 좋습니다.
seats = re.findall(r'padding-top:0px; ">([0-9]+)</div></TD>', text)
print(seats)
print(len(seats))
