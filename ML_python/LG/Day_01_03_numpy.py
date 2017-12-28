# Day_01_03_numpy.py
import numpy as np


def not_used_1():
    print(np.arange(10))
    print(np.arange(10, 20))
    print(np.arange(10, 20, 2))
    print(np.arange(5, -5, -1))

    print(type(np.arange(10)))
    print('-' * 50)

    a = np.arange(6)
    print(a.shape, a.ndim, a.dtype)

    b = np.arange(6).reshape(2, 3)
    print(b.shape, b.ndim, b.dtype)
    print(b)

    # 문제
    # 변수 c에 2페이지 3행 4열짜리 배열을 만들고 결과를 확인해 보세요.
    c = np.arange(24).reshape(2, 3, 4)
    print(c)
    print(c.shape, c.ndim, c.dtype)

    print(a.itemsize, b.itemsize, c.itemsize)
    print(a.size, b.size, c.size)
    print('-' * 50)

    print(np.array([2, 3, 4]))
    print(np.array((2, 3, 4)))
    print(list(np.array((2, 3, 4))))

    # print(np.array(2, 3, 4))

    # 문제
    # np.array를 사용해서
    # 0~5 까지의 정수를 2행 3열 배열에 넣어 보세요. (3가지)
    print(np.array([0, 1, 2, 3, 4, 5]).reshape(2, 3))
    print(np.array([[0, 1, 2], [3, 4, 5]]))
    print(np.array([np.arange(3), np.arange(3, 6)]))
    print(np.array(range(6)).reshape(2, 3))

    print(np.arange(6).reshape(2, 3))
    print(np.arange(6).reshape(-1, 3))
    print(np.arange(6).reshape(2, -1))
    print('-' * 50)

    # 문제
    # 2차원 배열을 1차원 리스트로 변환해 보세요.
    a = np.arange(6).reshape(2, 3)
    print(list(a))
    print(list(a.reshape(a.size)))
    print(list(a.reshape(-1)))
    print('-' * 50)

    print(np.zeros((2, 5)))
    print(np.ones((2, 5)))
    print(np.empty((2, 5)))
    print(np.full((2, 5), 0.5))

    a = [[1, 2, 3], [4, 5, 6]]
    print(np.zeros_like(a))
    print(np.ones_like(a))
    print(np.empty_like(a))
    print(np.full_like(a, 0.5))

    print(np.zeros_like(a, dtype=np.float))
    print(np.ones_like(a, dtype=np.float32))
    print(np.empty_like(a, dtype=np.float64))
    print(np.full_like(a, 0.5, dtype=np.float16))

    print(np.zeros_like(a, dtype=np.float).dtype)

    print(np.arange(0, 2, 0.25))
    print(np.linspace(0, 2, 9))


a = np.arange(3)
print(a)
print(a + 1)            # broadcasting
print(a ** 2)
print(a > 1)
print(np.sin(a))        # universal funcion
print()

b = np.array([3, 4, 5])

print(a > b)            # vector operation
print(a + b)
print(a * b)
print()

c = np.arange(6).reshape(-1, 3)
c += 1
print(c)
print(c > 3)
print(np.logical_and(c > 0, c < 3))
print(c[c>3])
print()

# 문제
# c와 행렬 곱셈할 수 있는 배열을 만드세요.
# np.dot()
print(c)        # (2, 3)

d = np.arange(6).reshape(3, -1)
print(d)

print(np.dot(c, d))     # (2, 3) x (3, 2)
print(np.dot(d, c))     # (3, 2) x (2, 3)
print('-' * 50)

# 슬라이싱 (리스트, ndarray)
a = list(range(10))
print(a)
print(a[-1], a[-2])

print(a[3:7])       # range()

# 문제
# 앞쪽 절반을 출력해 보세요.
# 뒤쪽 절반을 출력해 보세요.
print(a[0:len(a)//2])
print(a[:len(a)//2])

print(a[len(a)//2:len(a)])
print(a[len(a)//2:])

# 문제
# 짝수 번째만 출력해 보세요.
# 홀수 번째만 출력해 보세요.
# 거꾸로 출력해 보세요.
print(a[::2])
print(a[1::2])
print(a[3:4])
print(a[3:3])
print(a[len(a)-1:0:-1])
print(a[len(a)-1:-1:-1])
print(a[-1:-1:-1])
print(a[::-1])
print('-' * 50)

a = np.arange(3)
b = np.arange(6)
c = np.arange(3).reshape(-1, 3)
d = np.arange(6).reshape(-1, 3)
e = np.arange(3).reshape(3, -1)

print(a)
print(b)
print(c)
print(d)
print(e)

print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)
print(e.shape)

# 검사
# 5개의 변수를 각각 2개씩 모두 더해 보세요.
# 그래서.. 안 되는 것은
# 왜 그런지 생각해 보세요.

# print(a + b)      # error.
# print(a + c)
# print(a + d)
# print(a + e)

# print(b + c)      # error.
# print(b + d)      # error.
# print(b + e)

# print(c + d)
# print(c + e)

# print(d + e)      # error.
