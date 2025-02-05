# NewtonSolver

## Overview
`NewtonSolver` is a Python implementation of **Newton's method** for solving systems of nonlinear equations. It iteratively finds the roots of multivariate functions using the function's Jacobian matrix.

## Features
- Supports **multivariate** root-finding.
- Uses **direct solve (`np.linalg.solve`)** for efficient Newton steps.
- Handles **singular Jacobians** by using **least squares (`np.linalg.lstsq`)**.
- Provides **detailed convergence information**.
- Allows customizable **tolerance (`tol`)** and **maximum iterations (`maxiter`)**.

## Installation
No additional installation is required. The function only depends on **NumPy**, which is included in most Python distributions. You can install NumPy using:
```sh
pip install numpy
```

## Usage
### Function Signature
```python
NewtonSolver(func, x, jac, tol=1.48e-8, maxiter=50)
```

## Example Usage
```python
import numpy as np

def func(x):
    return np.array([x[0]**2 + x[1]**2 - 4, x[0] * np.exp(x[1]) - 1])

def jac(x):
    return np.array([[2*x[0], 2*x[1]], [np.exp(x[1]), x[0]*np.exp(x[1])]])

x0 = np.array([1.0, 1.0])  # Initial guess
root, info = NewtonSolver(func, x0, jac)

print("Root:", root)
print("Convergence Info:", info)
```

## Error Handling
The function includes various **input validations**:
- Ensures `func` and `jac` are **callable**.
- Checks if `x` is a **1D NumPy array**.
- Ensures `tol > 0` and `maxiter >= 1`.
- Handles **singular Jacobians** by switching to **least squares solutions**.

## Convergence Criteria
- The solver stops when the **residual norm** is less than or equal to `tol`.
- If the method reaches `maxiter` iterations without meeting `tol`, it **terminates without full convergence**.


