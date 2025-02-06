import numpy as np
import pytest
from algorithms.newton_method import NewtonSolver

# ===================================================================
# Test 1: Basic Functionality Test (Single-Variable Equation)
# ===================================================================
def test_newton_basic():
    """Test Newton's method for solving the single-variable equation x^2 - 4 = 0"""
    def func(x):
        return np.array([x[0]**2 - 4])  # Roots are x=2 or x=-2

    def jac(x):
        return np.array([[2 * x[0]]])   # Jacobian is 2x

    # Initial guess x=3, should converge to x=2
    x_initial = np.array([3.0])
    x_sol, info = NewtonSolver(func, x_initial, jac, tol=1e-6, maxiter=50)

    # Verify the result
    assert info['converged'] == True
    assert np.allclose(x_sol, [2.0], atol=1e-6)
    assert info['final_residual'] < 1e-6
    assert info['iterations'] < 10  # Newton's method should converge quickly

# ===================================================================
# Test 2: Multivariable Equation Test
# ===================================================================
def test_newton_multivariate():
    """
    Test solving a system of multivariable equations:
    f0 = x0^2 - 1
    f1 = x1^2 - 4
    Solutions include [1, 2] or [-1, -2], etc.
    """
    def func(x):
        return np.array([
            x[0]**2 - 1,
            x[1]**2 - 4
        ])

    def jac(x):
        return np.array([
            [2*x[0], 0],
            [0, 2*x[1]]
        ])

    x_initial = np.array([2.0, 3.0])
    x_sol, info = NewtonSolver(func, x_initial, jac, tol=1e-8)

    assert info['converged'] == True
    assert np.allclose(x_sol, [1.0, 2.0], atol=1e-8)
    assert info['final_residual'] < 1e-8

# ===================================================================
# Test 3: Initial Residual Already Satisfies Tolerance (No Iteration Needed)
# ===================================================================
def test_initial_convergence():
    """Initial guess is already close to the solution, residual is below tolerance"""
    def func(x):
        return np.array([x[0] - 2.0])  # root is x=2

    def jac(x):
        return np.array([[1.0]])

    x_initial = np.array([2.0])  # Initial residual is 0
    x_sol, info = NewtonSolver(func, x_initial, jac)

    assert info['converged'] == True
    assert info['iterations'] == 0
    assert info['final_residual'] <= 1.48e-8
    assert jac(x_initial) == np.array([[1.0]])

# ===================================================================
# Test4: triggering maxiter when convergence is not possible
# ===================================================================
def test_maxiter_reached():
    """Test triggering maxiter when convergence is not possible"""
    def func(x):
        return np.array([x[0]**2 + 1])  # No real solution (never converges)

    def jac(x):
        return np.array([[2 * x[0]]])

    x_initial = np.array([1.0])
    x_sol, info = NewtonSolver(func, x_initial, jac, maxiter=5)

    assert info['converged'] == False
    assert info['iterations'] == 5
    assert info['final_residual'] > 1.48e-8

# ===================================================================
# Test 5: Invalid Input Handling
# ===================================================================
def test_invalid_inputs():
    """Test invalid parameter inputs (such as non-callable functions, invalid tol/maxiter, etc.）"""
    def func(x): return np.array([0])
    def jac(x): return np.array([[1]])

    # Case 1: func is not callable
    with pytest.raises(TypeError):
        NewtonSolver("not a function", [1], jac)

    # Case 2: jac is not callable
    with pytest.raises(TypeError):
        NewtonSolver(func, [1], "not a function")

    # Case 3: x is not a 1D array
    with pytest.raises(ValueError):
        NewtonSolver(func, [[1, 2], [3, 4]], jac)

    # Case 4: tol <= 0
    with pytest.raises(ValueError):
        NewtonSolver(func, [1], jac, tol=-1e-8)

    # Case 5: maxiter < 1
    with pytest.raises(ValueError):
        NewtonSolver(func, [1], jac, maxiter=0)

# ===================================================================
# Test 6: Least Squares Fallback for Singular Jacobian
# ===================================================================
def test_singular_jacobian():
    """Test fallback to least squares solution when the Jacobian is singular"""
    def func(x):
        return np.array([x[0]**3 - 8])  # root is x=2

    def jac(x):
        # At x=0, the Jacobian is [[0]] (singular)
        return np.array([[3 * x[0]**2]])

    # Initial guess x=0, Jacobian is singular
    x_initial = np.array([0.0])
    x_sol, info = NewtonSolver(func, x_initial, jac, maxiter=20)

    # Check if it eventually converges to the solution x=2
    assert info['converged'] == True
    assert np.allclose(x_sol, [2.0], atol=1e-6)