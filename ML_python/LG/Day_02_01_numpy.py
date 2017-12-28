# Day_02_01_numpy.py
import numpy as np

np.random.seed(1)

# print(np.random.random(2, 3))
print(np.random.random((2, 3)))
print(np.random.uniform(size=(2, 3)))
print(np.random.rand(2, 3))

print(np.random.randn(2, 3))
print('-' * 50)

a = np.random.choice(range(10), 12)
print(a)

b = a.reshape(-1, 4)
print(b)

print(np.sum(b))
print(np.sum(b, axis=0))    # 수직, 열(column)
print(np.sum(b, axis=1))    # 수평, 행(row)
print('-' * 50)

def func(y, x):
    return y*10 + x

a = np.fromfunction(func, shape=(5, 4), dtype=np.int32)
a = np.fromfunction(lambda y, x: y*10+x,
                    shape=(5, 4), dtype=np.int32)
print(a)
print(a[0])
print(a[0][0])
print(a[0, 0])          # fancy indexing
print(a[0:2, 0])
print(a[1:3, 1:3])

# 문제
# 거꾸로 출력해 보세요.
print(a[::-1])
print(a[::-1, ::-1])

# 문제
# 행과 열을 바꿔서 출력해 보세요. (반복문)
print(a.T)

for i in range(a.shape[-1]):
    print(a[:, i])
print('-' * 50)

a = np.arange(6).reshape(-1, 3)
print(a)

a[0] = 99
print(a)

a[:, 0] = 88
print(a)

a[:, ::2] = 77
print(a)

# 문제
# 대각선이 1로 채워진 5x5 행렬을 만드세요.
# 나머지는 0.
print(np.eye(5, 5))

b = np.zeros((5, 5))
# b[0, 0] = 1
# b[1, 1] = 1
b[range(5), range(5)] = 1
b[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]] = 1
print(b)
print('-' * 50)

# 문제
# 테두리가 1로 채워진 5x5 배열을 만드세요.
# 나머지는 0.
a = np.zeros((5, 5))
a[0], a[-1] = 1, 1
a[:, 0], a[:, -1] = 1, 1
print(a)

b = np.ones((5, 5))
b[1:-1, 1:-1] = 0
print(b)
print('-' * 50)

a = np.arange(10)
print(a)
print(a[[1, 4, 7]])
# print(a[1, 4, 7])

b = a.reshape(-1, 5)
print(b)
print(b[[0]])
print(b[[1, 0]])
print(b[[0, 1], [2, 3]])

b[[0, 1], [2, 3]] = 99
print(b)

c = b > 5
print(c)
print(b[c])
print('-' * 50)

a = np.array([3, 1, 2])
print(np.sort(a))
print(a)

b = np.argsort(a)
print(b)
print(a[b])

#             2  4  1  0  3
x = np.array([4, 3, 1, 5, 2])
y = np.argsort(x)
print(y)
print('-' * 50)

# onehot encoding
a = [1, 3, 0, 3]

# a를 아래처럼 출력해 보세요. (np.max)
# [[0 1 0 0]
#  [0 0 0 1]
#  [1 0 0 0]
#  [0 0 0 1]]
b = np.zeros([len(a), np.max(a)+1])
b[range(len(a)), a] = 1
print(b)

n = np.max(a)+1
b = np.eye(n, n)[a]
print(b)
