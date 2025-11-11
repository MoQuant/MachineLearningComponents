# Vandermonde Interpolation

import numpy as np
import matplotlib.pyplot as plt

def Vandermonde(x, y):
    x, y = np.array(x), np.array(y)
    X = np.array([[i**j for j in range(len(x))] for i in x])
    return np.linalg.inv(X).dot(y)

y = [2, 4, 6, 1, 3, 9, 5, 8]
x = list(range(len(y)))


V = Vandermonde(x, y)

x0, x1 = np.min(x), np.max(x)
n = 100

dx = (x1 - x0)/(n - 1)

ux, uy = [], []

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(n):
    ex = x0 + i*dx
    ez = np.array([ex**k for k, j in enumerate(x)])
    yz = ez.T @ V

    ux.append(ex)
    uy.append(yz)

    ax.cla()
    ax.set_title('Vandermonde Interpolation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.scatter(x, y, color='red')
    ax.plot(ux, uy, color='blue')
    plt.pause(0.1)

plt.show()

