# Day_04_09_compare.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

np.set_printoptions(linewidth=200)


def make_wave(n_samples=100):
    rnd = np.random.RandomState(42)
    x = rnd.uniform(-3, 3, size=n_samples)

    noise = np.sin(4 * x) + x
    y = (noise + rnd.normal(size=len(x))) / 2

    # 사이킷런에서 2차원으로 전달하지 않으면 에러.
    return x.reshape(-1, 1), y


def regression_scikit(train_x, test_x, train_y, test_y):
    lr = LinearRegression().fit(train_x, train_y)

    print('[scikit]')
    print('{}, {}'.format(lr.coef_[0], lr.intercept_))

    print('train : {:.2f}'.format(lr.score(train_x, train_y)))
    print(' test : {:.2f}'.format(lr.score(test_x, test_y)))


def regression_tensor(train_x, test_x, train_y, test_y):
    def score(sess, hypothesis, x_holder, x_data, y_data):
        y_hat = sess.run(hypothesis, feed_dict={x_holder: x_data})

        y_mean = np.mean(y_data)
        u = np.sum((y_hat  - y_data) ** 2)  # 정답까지의 거리
        v = np.sum((y_mean - y_data) ** 2)  # 편차

        return 1 - u / v


    # 2차원을 1차원으로 변환
    train_x = train_x.reshape(-1)
    test_x = test_x.reshape(-1)

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = w * x + b
    cost = tf.reduce_mean((hypothesis - train_y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for _ in range(100):
        sess.run(train, feed_dict={x: train_x})

    ww, bb = sess.run([w, b])
    print('[tensor]')
    print('{}, {}'.format(ww[0], bb[0]))

    print('train : {:.2f}'.format(score(sess, hypothesis, x, train_x, train_y)))
    print(' test : {:.2f}'.format(score(sess, hypothesis, x, test_x, test_y)))

    sess.close()




x, y = make_wave(60)
print(x.shape, y.shape)

data = train_test_split(x, y, random_state=42)

# train_x, test_x, train_y, test_y = data
# regression_scikit(train_x, test_x, train_y, test_y)
regression_scikit(*data)
regression_tensor(*data)
