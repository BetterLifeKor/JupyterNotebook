# Day_01_01_python.py

# ctrl + shift + f10

print(12, 3.14, True, 'hello')
print(type(12), type(3.14), type(True), type('hello'))

a, b = 7, 3
print(a, b)

# 산술 연산
print(a + b)
print(a - b)
print(a * b)
print(a / b)    # 실수 나눗셈
print(a ** b)   # 지수
print(a // b)   # 정수 나눗셈
print(a % b)

print('a' + 'b')
print('a' * 3)
print('-' * 50)

# 관계
print(a < b)
print(a <= b)
print(a > b)
print(a >= b)
print(a == b)
print(a != b)

print(int(a < b))
print(int(a > b))

age = 15
print(10 <= age <= 19)
print('-' * 50)

# 논리 연산 : and  or  not
print(True and True)
print(True and False)
print(False and True)
print(False and False)
print('-' * 50)

a = 3
if a % 2 == 1:
    print('홀수')
else:
    print('짝수')

if a < 0:
    print('음수')
elif a > 0:
    print('양수')
else:
    print('제로')

print('-' * 50)

for i in range(5):          # 종료
    print(i, end=' ')
print()

for i in range(0, 5):       # 시작, 종료
    print(i, end=' ')
print()

for i in range(0, 5, 1):    # 시작, 종료, 증감
    print(i, end=' ')
print()

for i in range(4, -1, -1):
    print(i, end=' ')
print()

for i in reversed(range(5)):
    print(i, end=' ')
print()
print('-' * 50)

# collection : list, tuple, set, dictionary
#               []    ()     {}   {}

a = [1, 3, 5]
print(a)
print(a[0], a[1], a[2])

a.append(7)
a += [9]
# a.extend([9])

# a.append([7])
# a += 9

for i in range(len(a)):
    print(i, a[i])

for i in a:         # iterable.
    print(i)

# 문제
# a를 거꾸로 뒤집어 보세요.
print(a)

for i in range(len(a)//2):
    # print(i, len(a)-1-i)
    a[i], a[len(a)-1-i] = a[len(a)-1-i], a[i]

print(a)
print('-' * 50)

a = (1, 3, 5)
print(a)
print(a[0], a[1], a[2])

for i in a:
    print(i)

# 튜플 : 상수 버전의 리스트
# a.append(7)   # error.
# a[0] = 99     # error.
print('-' * 50)

def dummy_1():
    pass

a = dummy_1()
print(a)

def dummy_2(a, b):
    if a < b:
        return a, b
    return b, a

m1, m2 = dummy_2(3, 7)
print(m1, m2)

m = dummy_2(3, 7)           # packing
print(m, m[0], m[1])

def dummy_3():
    return [1, 3, 5]

a = dummy_3()
print(a)

a1, a2, a3 = dummy_3()      # unpacking
print(a1, a2, a3)
print('-' * 50)

print(1, 2, 3, sep='**', end='\n\n')
print(1, 2, 3)

def f_1(a, b, c):
    print(a, b, c)

f_1(1, 2, 3)            # positional parameter
f_1(a=1, b=2, c=3)      # keyword parameter
f_1(1, b=2, c=3)
f_1(1, c=3, b=2)

def f_2(*args):         # 가변인자, packing
    print(args, *args)  # unpacking (force)

f_2()
f_2(1)
f_2(1, 2)

# 딕셔너리
# 영한 사전 : 영어 단어를 찾으면 한글 설명 나옴.
# 영어 단어 : key
# 한글 설명 : value

a = {'name': 'kim', 'age': 20, 30: 50}
a = dict(name='kim', age=20)
print(a)
print(a['name'], a['age'])

print(a.keys())
print(a.values())
print(a.items())

for k in a.keys():
    print(k, end=' ')
print()

for v in a.values():
    print(v, end=' ')
print()

# 문제
# items 함수를 for문에 적용해 보세요.
for kv in a.items():
    print(kv, kv[0], kv[1])

for k, v in a.items():
    print(k, v)
print('-' * 50)

for k in a:
    print(k, a[k])

s = 'hello'
for c in s:
    print(c, end=' ')
print()

for i, c in enumerate(s):
    print(i, c)

# 문제
# 딕셔너리의 items 함수와 enumerate를
# 연결해서 사용해 보세요.
for i in enumerate(a.items()):
    print(i)

for i, kv in enumerate(a.items()):
    print(i, kv)

for i, (k, v) in enumerate(a.items()):
    print(i, k, v)
