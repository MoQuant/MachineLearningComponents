import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import random as rd

def SVM(X, y, C=10):
    def Objective(params):
        W = params[:-1]
        b = params[-1]
        A = 0.5*np.dot(W, W)
        B = C*sum([np.maximum(0, 1 - y[i]*(W.T @ X[i] + b)) for i in range(len(y))])
        return A + B
    w0 = np.ones(len(X[0]) + 1)
    res = minimize(Objective, w0)
    return res.x

def Plane(coef, X, N=30):
    beta = coef[:-1]
    alpha = coef[-1]

    ix = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), N)
    iy = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), N)

    x, y = np.meshgrid(ix, iy)
    z = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            hold = np.array([x[i, j], y[i, j]])
            z[i, j] = alpha + beta.T @ hold

    return x, y, z

rows = 40
cols = 2

X = np.random.randn(rows, cols)
y = np.array(list(map(lambda ue: -1 if ue == 0 else 1, [rd.randint(0, 1) for i in range(rows)])))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#for ix, classify in zip(X, y):
#    ax.scatter(ix[0], ix[1], ix[0] + ix[1] + classify, s=7, color='red' if classify == -1 else 'limegreen')

coef = SVM(X, y)

px, py, pz = Plane(coef, X)

ax.plot_surface(px, py, pz, color='black')

hX = np.random.randn(int(rows/2), cols)

alpha = coef[-1]
beta = coef[:-1]

for ix, iy in hX:
    iz = ix + iy
    classify = alpha + beta.T @ np.array([ix, iy])
    ax.scatter(ix, iy, iz + classify, color='blue' if classify < 0 else 'orange')
    plt.pause(0.1)
    

plt.show()
