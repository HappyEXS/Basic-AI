import numpy as np
import math
import sys

'''
Oznaczenia w algorytmach:
R - tablica gamma osobników po reprodukcji
M - tablica gamma osobników po mutacji
Om - tablica ocen dla osobników mutacji
P - tablica mi osobników następnej populacji
O - tablica ocen populacji
x, o - najlepszy osobnik wraz ze swoją oceną

Algorytmy różnią się reprezentacją osobnika.
Dla metody z mutacją logarytmiczno-gaussowską:
x = [x0, x1, ..., xd]
Dla metody z mutacją samoadaptacyjną:
x = ([x0, x1, ..., xd], sigma)
'''

# Algorytm z mutacją logarytmiczno-gaussowska ================================
def strategy_g(q, P, mi, gamma, sigma, Tmax):
    t = 0
    while stop(t, Tmax):
        R = draw(P, gamma)
        M, sigma = mutation_g(R, sigma)
        Om = ratings_g(q, M)
        P, O = naxt_gen(M, Om, mi)
        x, o = best(P, O)
        t += 1
    return x, o, t, sigma

def mutation_g(R, sigma):
    '''Metoda mutacji logarytmiczno-gaussowskiej. '''
    M = []
    d = len(R[0])
    sig = gauss(sigma, d)
    for r in R:
        x = []
        for i in range(d):
            x.append(r[i] + np.random.normal(0, sig))
        M.append(x)
    return M, sig

def gauss(sigma, d):
    '''Zwraca sigmę dla nowej populacji. '''
    return sigma * math.exp((1/math.sqrt(d)) * np.random.normal(0, 1))

def ratings_g(q, P):
    '''Zwaraca listę ocen osobników po mutacji. '''
    return [q(x) for x in P]

# Algorytm z mutacją samoadaptacyjną =========================================
def strategy_a(q, P, mi, gamma, sigma, Tmax):
    t = 0
    o10 = sys.maxsize
    while stop(t, Tmax):
        R = draw(P, gamma)
        M = mutation_a(R)
        Om = ratings_a(q, M)
        P, O = naxt_gen(M, Om, mi)
        x, o = best(P, O)
        t += 1
        if t % 10 == 0:
            if abs(o10 - o) < 0.0001:
                break
            o10 = o
    return x, o, t

def mutation_a(R):
    '''Metoda mutacji samoadaptacyjnej. '''
    M = []
    d = len(R[0][0])
    for r in R:
        sig = adaptation(r[1], d)
        z = np.random.normal(0, 1, size=(1,d))[0]
        x = r[0] + sig*z
        M.append((x, sig))
    return M

def adaptation(sigma, d):
    '''Zwraca sigmę dla każdego osobnika. '''
    ksi = (1/math.sqrt(d)) * np.random.normal(0, 1)
    return sigma * math.exp(ksi)

def ratings_a(q, P):
    ''''Zwaraca listę ocen osobników po mutacji. '''
    return [q(x[0]) for x in P]

# Funkcje wspólne dla obydwu algorytmów ======================================
def naxt_gen(M, O, mi):
    '''Wybiera mi najlepszych osobników, które stworzą kolejną populację. '''
    OM = zip(O, M)
    try:
        OM_sorted = sorted(OM)
        next_gen = OM_sorted[0: mi]
        O, M = zip(*next_gen)
        return list(M), list(O)
    except ValueError:
        return M, O

def best(P, O):
    '''Populacja jest posortowana po ocenach, więc najlepszym osobnikiem
    jest ten pierwszy w liście. '''
    return P[0], O[0]

def stop(t, Tmax):
    '''Warunek zatrzymania algorytmu'''
    return t < Tmax

def draw(P, gamma):
    '''Losuje gamma osobników z populacji z powtórzeniami z równą wagą. '''
    R = []
    for i in range(gamma):
        j = np.random.randint(0, len(P))
        R.append(P[j])
    return R