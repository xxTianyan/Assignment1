import numpy as np
from algorithms.newton_method import NewtonSolver

def func(x):
    return np.array([
        x[0]**2 + x[1]**2 - 4,  # Circle equation
        x[0] * np.exp(x[1]) - 1 # Exponential equation
    ])

def jac(x):
    return np.array([
        [2*x[0], 2*x[1]],        # Derivatives of f1
        [np.exp(x[1]), x[0]*np.exp(x[1])]  # Derivatives of f2
    ])

x0 = np.array([0, 1.0])  # Initial guess
root, info = NewtonSolver(func, x0, jac,plot=True)

print(f"Solution: {root}")
print(f"Converged: {info['converged']}, Iterations: {info['iterations']}")