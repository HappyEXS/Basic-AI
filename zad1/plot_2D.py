import matplotlib.pyplot as plt
import numpy as np
import autograd
from scipy.misc import derivative

STEP = 0.05
STEPS = 30
X = [-100, 100]
ACCURACY = 1e-4


def func(x):
    return (x[0]+30)**2 + 2*x[1]**2

def plot_func(f):
    x = np.arange(X[0], X[1], 1)
    y = np.arange(X[0], X[1], 1)
    xx, yy = np.meshgrid(x, y, sparse=False)
    z = f((xx,yy))
    plt.contourf(x, y, z, levels=15)
    plt.colorbar()

def plot_points(f):
    # start_point = np.random.uniform(X[0], X[1], 2)
    start_point = np.random.uniform(50, 100, 2)
    print(start_point)
    points = np.array(gradient(f, start_point))
    plt.scatter(points[..., 0], points[..., 1], c='r')

def gradient(f, point, step=STEP, steps=STEPS):
    points = []
    points.append(point.copy())
    grad = autograd.grad(f)
    for i in range(steps):
        point -= grad(point) * step
        points.append(point.copy())
    return points

def main():
    plot_func(func)
    plot_points(func)
    plt.show()

if __name__ == '__main__':
    main()
