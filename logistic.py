import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

x = np.loadtxt('ex5Logx.dat', delimiter=',')
y = np.loadtxt('ex5Logy.dat', delimiter=',')


x0 = x[np.where(y == 0)]
x1 = x[np.where(y == 1)]
plt.scatter(x0[:,0], x0[:,1], marker='o', c='blue')
plt.scatter(x1[:,0], x1[:,1], marker='v', c='red')
plt.legend(['y=0', 'y=1'])

poly = PolynomialFeatures(degree=6)
x = poly.fit_transform(x)
(n, m) = x.shape
theta = np.zeros((m, 1))

max_iter = 15
lambdas = [0.03, 0.1, 0.3, 1.0]
J = np.zeros((len(lambdas),max_iter))

for (li,lam) in enumerate(lambdas):
    for i in range(max_iter):
        z=np.dot(x, theta)
        h=sigmoid(z)
        J[li][i] = (1.0/n)*np.sum(-y * np.log(h) - (1-y)*np.log(1-h))+lam/(2.0*n)*np.sum(theta**2)
        G = (lam/n)*theta
        L = (lam/n)*np.eye(m)
        G[0] = 0
        L[0] = 0
        # print h.shape, G.shape, L.shape, x.shape
        grad = ((1.0/n)*np.dot(x.T, h-y)) + G
        H = (1.0/m)*np.dot(x.T, np.dot(np.diag(h[:,0]), np.dot(np.diag((1-h)[:,0]), x))) + L
        theta = theta - np.dot(np.linalg.inv(H), grad)

plt.figure()
for (li,lam) in enumerate(lambdas):
    plt.plot(J[li], 'o--')
    plt.legend("lambda = %f" % lam)
plt.show()
