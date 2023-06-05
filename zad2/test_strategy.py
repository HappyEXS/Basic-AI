from evolutionary_strategy import strategy_g, strategy_a, mutation_a
import numpy as np
import timeit
import math

X = [-100, 100]

def func1(D):
    def q(x):
        m = math.sqrt(sum(i**2 for i in x))
        return ((m**2 - D)**2) ** (1/8) + D ** (-1) * (m**2/2 + sum(x)) + 0.5
    return q

def func2(x):
    return sum(pow(10,((i-1)/(10-1)))*x[i]**2 for i in range(10))

def func3(x):
    return (x[0]+30)**2 + 2*x[1]**2

def first_population_g(Dim):
    P = []
    x1 = np.random.uniform(X[0], X[1], Dim)
    P.append(x1)
    return P

def first_population_a(Dim, sig):
    P = []
    x1 = np.random.uniform(X[0], X[1], Dim)
    P.append((x1, sig))
    return P

def experiment_gauss(q, Dim, mi, gamma, sig, Tmax):
    P0 = first_population_g(Dim)

    start_time = timeit.default_timer()
    x, o, t, sig = strategy_g(q, P0, mi, gamma, sig, Tmax)
    end_time = timeit.default_timer()

    # print(P0[0])
    # print(x)
    # print(sig)
    # print(f'Result = {round(o, 3)}'.ljust(14), f' Time: {round(end_time - start_time, 3)}'.ljust(13), f'Iterations: {t}')
    return o, end_time - start_time, t

def experiment_adaptation(q, Dim, mi, gamma, sig, Tmax):
    P0 = first_population_a(Dim, sig)

    start_time = timeit.default_timer()
    x, o, t = strategy_a(q, P0, mi, gamma, sig, Tmax)
    end_time = timeit.default_timer()

    # print(P0[0][1])
    # print(x[1])
    # print(f'Result = {round(o, 3)}'.ljust(14), f' Time: {round(end_time - start_time, 3)}'.ljust(13), f'Iterations: {t}')
    return o, end_time - start_time, t


def main():
    Dim = 10
    mi = 20
    Sig = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    Gamma = [mi, 2*mi, 4*mi, 7*mi, 10*mi, 15*mi]
    Seeds = [4231, 1524, 2132, 5532, 2020]
    Tmax = 2000
    Q = [func1(Dim), func2, func3]
    q = Q[0]

    print('Metoda logarytmicvzno-gaussowska')
    for sig in Sig:
        print('Sigma: ', sig)
        O = []
        Time = []
        T = []
        for seed in Seeds:
            np.random.seed(seed)
            o, time, it = experiment_gauss(q, Dim, mi, 7*mi, sig, Tmax)
            O.append(o)
            Time.append(time)
            T.append(it)
        print(f'Result = {round(sum(O)/len(O), 3)}'.ljust(14), f' Time: {round(sum(Time)/len(Time), 3)}'.ljust(13), f'Iterations: {sum(T)/len(T)}')

    print("Metoda samoadaptacji")
    for sig in Sig:
        print('Sigma: ', sig)
        O = []
        Time = []
        T = []
        for seed in Seeds:
            np.random.seed(seed)
            o, time, it = experiment_adaptation(q, Dim, mi, 7*mi, sig, Tmax)
            O.append(o)
            Time.append(time)
            T.append(it)
        print(f'Result = {round(sum(O)/len(O), 3)}'.ljust(14), f' Time: {round(sum(Time)/len(Time), 3)}'.ljust(13), f'Iterations: {sum(T)/len(T)}')

    sig = 0.5
    print('Metoda logarytmicvzno-gaussowska')
    for gamma in Gamma:
        print('Gamma: ', gamma)
        O = []
        Time = []
        T = []
        for seed in Seeds:
            np.random.seed(seed)
            o, time, it = experiment_gauss(q, Dim, mi, gamma, sig, Tmax)
            O.append(o)
            Time.append(time)
            T.append(it)
        print(f'Result = {round(sum(O)/len(O), 3)}'.ljust(25), f' Time: {round(sum(Time)/len(Time), 3)}'.ljust(13), f'Iterations: {sum(T)/len(T)}')

    print("Metoda samoadaptacji")
    for gamma in Gamma:
        O = []
        Time = []
        T = []
        print('Gamma: ', gamma)
        for seed in Seeds:
            np.random.seed(seed)
            o, time, it = experiment_adaptation(q, Dim, mi, gamma, sig, Tmax)
            O.append(o)
            Time.append(time)
            T.append(it)
        print(f'Result = {round(sum(O)/len(O), 3)}'.ljust(14), f' Time: {round(sum(Time)/len(Time), 3)}'.ljust(13), f'Iterations: {sum(T)/len(T)}')


if __name__ == '__main__':
    main()
