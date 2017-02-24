import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import sys

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
y = y.reshape((n, 1))

max_iter = 50
lambdas = [0.03, 0.1, 0.3, 1.0]
J = np.zeros((len(lambdas),max_iter))
Jmin = np.nan
thetaMin = np.zeros((m, 1))

for (li,lam) in enumerate(lambdas):
    theta = np.zeros((m, 1))
    for i in range(max_iter):
        z=np.dot(x, theta)
        h=sigmoid(z)
        J[li][i] = (1.0/n)*np.sum(-y * np.log(h) - (1-y)*np.log(1-h))+lam/(2.0*n)*np.sum(theta**2)
        G = (lam/n)*theta
        L = (lam/n)*np.eye(m)
        G[0] = 0
        L[0] = 0
        grad = ((1.0/n)*np.dot(x.T, h-y)) + G
        H = (1.0/m)*np.dot(x.T, np.dot(np.diag(h[:,0]), np.dot(np.diag((1-h)[:,0]), x))) + L
        theta = theta - np.dot(np.linalg.inv(H), grad)
        # print h.shape, G.shape, L.shape, x.shape, grad.shape, y.shape, H.shape
        if np.isnan(Jmin) or Jmin > J[li][i]:
            Jmin = J[li][i]
            thetaMin = theta

plt.figure()
legends = []
for (li,lam) in enumerate(lambdas):
    plt.plot(J[li][5:], 'o--')
    legends.append("lambda = %.2f" % lam)
plt.legend(legends)

u = np.linspace(-1., 1.5, 200)
v = np.copy(u)
z = np.zeros((200,200))

uv = np.array([u, v])
print uv
# for i in u:
#     for j in v:
#         z[i][j] = np.dot(poly.fit_transform([i,j]),thetaMin)
# print np.tile(thetaMin, (1,200))
# print np.repeat(thetaMin, 200).reshape((m, 200))
z = np.dot(poly.fit_transform(uv), np.tile(thetaMin, (1,200)))
# print z
#
# plt.figure()
# plt.contour(u, v, z)
# plt.show()
