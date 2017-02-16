import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt('ex2x.dat')
y = np.loadtxt('ex2y.dat')

plt.plot(x, y, 'ro')
plt.xlabel('Age in years')
plt.ylabel('Height in meters')


def normal_equation(x, y):
    x = np.array([np.ones_like(x), x])
    inv = np.linalg.inv(np.dot(x, x.T))
    theta = np.dot(np.dot(inv, x), y)  # 2*1
    return theta


def plot_theta_line(x, theta):
    px = np.linspace(np.min(x[0]), np.max(x[0]), 50)
    px = np.array([np.ones_like(px), x])
    plt.plot(px[1], np.dot(theta.T, px))


def gradient_descend(x, y, alpha=0.01, iter_num=5000):
    x = np.array([np.ones_like(x), x])
    theta = np.random.randn(x.shape[0])
    for i in range(iter_num):
        grad = np.dot(np.dot(theta.T, x) - y, x.T) / x.shape[1]
        theta = theta - alpha * grad
        if i % 100 == 0:
            print theta
    return theta

plot_theta_line(x, normal_equation(x, y))
plot_theta_line(x, gradient_descend(x, y, 0.01, 5000))
plt.show()
