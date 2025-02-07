from algorithms.bisection import CreateBisecSolver
import numpy as np

def class_ex_1(w):
    k = 1.0
    l = 1.0
    F = 0.25
    val = 2.0 * k * (np.sqrt(l ** 2.0 + w **2.0) - l) * w / np.sqrt(l ** 2.0 + w ** 2.0) - F
    return val

solver = CreateBisecSolver(class_ex_1, 1e-6)
a = 0
b = 3
c = solver(a,b)
print(f'The result is {c}')
