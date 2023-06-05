import numpy as np
import autograd
from scipy.misc import derivative
import timeit

STEP = 0.005
STEPS = 100000
ACCURACY = 1e-4
X = [-100, 100]
A = 100
N = 10

def end(f, point):
    # if point.max() <= ACCURACY:
    if f(point) <= ACCURACY:
        return True
    return False

def gradient(f, point, step=STEP, steps=STEPS):
    grad = autograd.grad(f)
    i = 0
    for i in range(steps):
        point -= grad(point) * step
        if end(f, point):
            # print(f"After {i+1} steps f(x) = {f(point)}")
            break
    return point, i+1, f(point)

def newton(f, point, step=STEP, steps=STEPS):
    grad = autograd.grad(f)
    hess = autograd.hessian(f)
    i = 0
    for i in range(steps):
        grad_eval = grad(point)
        hess_eval = hess(point)
        a = hess_eval + step*np.eye(point.size)
        b = grad_eval
        point = np.linalg.solve(a,np.dot(a,point) - b)
        if end(f, point):
            # print(f"After {i+1} steps f(x) = {f(point)}")
            break
    return point, i+1, f(point)

def newton_dynamic_step():
    pass

def main():
    # N = 20
    # start_point = np.random.uniform(X[0], X[1], N)
    # print(start_point)
    # print(func(start_point))
    # start_time = timeit.default_timer()
    # # end_point, steps, value = gradient(func, start_point)
    # end_point, steps, value = newton(func, start_point)
    # end_time = timeit.default_timer()
    # print(end_point)
    # print(end_time-start_time)

    N = 10
    a = [1, 10, 100]
    for A in a:
        start_point = np.random.uniform(X[0], X[1], N)
        start_time = timeit.default_timer()
        end_point, steps, value = gradient(func, start_point)
        end_time = timeit.default_timer()
        print("N =", N,f"A = {str(A).rjust(3)}", "Steps: ".rjust(3),steps, round(end_time-start_time, 4))
    N = 20
    for A in a:
        start_point = np.random.uniform(X[0], X[1], N)
        start_time = timeit.default_timer()
        end_point, steps, value = newton(func, start_point, 1)
        end_time = timeit.default_timer()
        print("N =", N,f"A = {str(A).rjust(3)}", "Steps: ".rjust(3),steps, round(end_time-start_time, 4))

    # N = 10
    # A = 10
    # print("Gradient:")
    # steps = [0.001, 0.005, 0.01]
    # for step in steps:
    #     start_point = np.random.uniform(X[0], X[1], N)
    #     start_time = timeit.default_timer()
    #     end_point, steps, value = gradient(func, start_point, step)
    #     end_time = timeit.default_timer()
    #     print(f"Step: = {str(step).rjust(3)}", "Steps: ".rjust(3),steps, round(end_time-start_time, 4))

    # print("Newton:")
    # steps = [0.001, 0.005, 0.01, 0.05, 0.1, 1]
    # for step in steps:
    #     start_point = np.random.uniform(X[0], X[1], N)
    #     start_time = timeit.default_timer()
    #     end_point, steps, value = newton(func, start_point, step)
    #     end_time = timeit.default_timer()
    #     print(f"Step: = {str(step).rjust(3)}", "Steps: ".rjust(3),steps, round(end_time-start_time, 4))


def func(x):
    return sum(pow(A,((i-1)/(N-1)))*x[i]**2 for i in range(N))


if __name__ == '__main__':
    main()
