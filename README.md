# ME700 Course Code

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![OS](https://img.shields.io/badge/os-ubuntu%20%7C%20macos%20%7C%20windows-blue)

[![codecov](https://codecov.io/gh/xxTianyan/ME700/branch/main/graph/badge.svg)](https://codecov.io/gh/xxTianyan/ME700)
![Tests](https://github.com/xxTianyan/ME700/actions/workflows/ci.yml/badge.svg)

## Description
This repository contains all the numerical algorithms that learned in the BU course ME700. 

## Installation
Firstly, clone or download the repository to your local environment:
```sh
# Clone the repository
git clone https://github.com/xxTianyan/ME700.git
cd ME700
```
Create a virtual environment, here we use conda:
```sh
# Create a new conda environment and activate it
conda create -n numerical python=3.12
conda activate numerical
```
Ensure that pip is using the most up to date version of setuptools:
```sh
pip install --upgrade pip setuptools wheel
```
Create an editable install of the bisection method code (note: you must be in the correct directory):
```sh
pip install -e .
```
Test that the code is working with pytest:
```sh
pytest
```
In order to see the tutorial, you need to install jupyter notebook:
```sh
pip install jupyter notebook
```

## NewtonSolver

### Overview
`NewtonSolver` is a Python implementation of Newton's method for solving nonlinear systems of equations. It iteratively refines an initial guess to find a root where the function evaluates to zero. The solver supports multivariate functions and includes optional visualization of convergence behavior.

### Features
- Solves multivariate nonlinear equations using Newton's method
- Accepts user-defined function and Jacobian
- Allows setting tolerance and maximum iterations
- Provides convergence information
- Supports optional convergence plot

### Usage
#### Function Signature
```python
NewtonSolver(func, x, jac, tol=1.48e-8, maxiter=50, plot=False)
```

#### Parameters
- `func` (*callable*): The function \( F(x) \) whose root is to be found. Should return a NumPy array.
- `x` (*array-like*): Initial guess for the root, given as a 1D NumPy array.
- `jac` (*callable*): Function computing the Jacobian matrix \( J(x) \).
- `tol` (*float*, optional): Convergence tolerance. Defaults to `1.48e-8`.
- `maxiter` (*int*, optional): Maximum number of iterations. Defaults to `50`.
- `plot` (*bool*, optional): If `True`, plots the residual convergence. Defaults to `False`.

#### Returns
The function returns:
- `x` (*numpy.ndarray*): Estimated root.
- `info` (*dict*): Dictionary containing:
  - `'converged'` (*bool*): Whether the solver converged.
  - `'iterations'` (*int*): Number of iterations performed.
  - `'final_residual'` (*float*): Final residual norm.

### Example Usage
```python
import numpy as np
from algorithms.newton_method import NewtonSolver

def func(x):
    return np.array([x[0]**2 + x[1] - 4, x[0] + x[1]**2 - 4])

def jac(x):
    return np.array([[2*x[0], 1], [1, 2*x[1]]])

x0 = np.array([2.0, 2.0])
root, info = NewtonSolver(func, x0, jac, plot=True)

print("Root:", root)
print("Convergence Info:", info)
```

### Error Handling
- Ensures `func` and `jac` are callable.
- Requires `x` to be a 1D NumPy array.
- Validates `tol` to be positive and `maxiter` to be at least 1.
- Handles singular Jacobian matrices by applying a fallback method.


## Bisection Method

### Overview
The repository also includes an implementation of the **Bisection Method**, a root-finding algorithm that repeatedly bisects an interval and selects the subinterval containing the root. 

### Features
- Implements the **Bisection Method** for root-finding
- Checks for correct interval selection
- Allows configurable tolerance

### Usage

#### Function Signature
```python
CreateBisecSolver(f: Callable[[float], float], tolerance: float) -> Callable[[float, float], float]
```

#### Parameters
- `f` (*callable*): The function \( f(x) \) whose root is to be found.
- `tolerance` (*float*): The stopping criterion; the algorithm halts when the function value is smaller than this threshold.

#### Returns
- A solver function that takes an interval `[a, b]` and returns the estimated root.

### Example Usage
```python
import numpy as np
from algorithms.bisection import CreateBisecSolver

def func(x):
    return x**2 - 4

solver = CreateBisecSolver(func, tolerance=1e-6)
root = solver(1, 3)
print("Root:", root)
```

### Error Handling
- Ensures `func` is callable.
- Validates `tolerance` is greater than zero.
- Checks that the input interval contains a sign change.


## ElastoPlastic Material Model

A Python class implementing 1D elasto-plastic material behavior with isotropic and kinematic hardening models.

### Features
- **Two Hardening Models**:
  - Isotropic Hardening (Yield surface expansion)
  - Kinematic Hardening (Bauschinger effect)
- **Stress-Strain Calculation**:
  - Elastic predictor-plastic corrector algorithm
  - Automatic yield surface updates
- **Visualization**:
  - Built-in stress-strain curve plotting
  - PNG output support

### Example Usage
```python
from elastoplastic import ElastoPlastic, generate_strain_path

# Create material
material = ElastoPlastic(
    E=200000,       # Young's modulus (MPa)
    H=10000,        # Hardening modulus (MPa)
    Yi=400,         # Initial yield stress (MPa)
    mode='isotropic' # 'isotropic' or 'kinematic'
)

# Generate strain path
strain_path = generate_strain_path(
    max_strain=0.06, 
    n_steps=1000,
    plot=True
)

# Apply loading
material.apply_loading(strain_path)

# Plot results
material.plot_curve(save_path='stress_strain.png')
```

## License
This function is open-source and can be freely modified and distributed.

## Author
Tianyan Xu




























