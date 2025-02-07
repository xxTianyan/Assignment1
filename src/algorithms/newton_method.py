import numpy as np
import matplotlib.pyplot as plt

def NewtonSolver(func, x, jac, tol=1.48e-8, maxiter=50, plot=False):
    '''
    Newton's method for solving roots of multivariate equations.
    ------------------------------------------------------------------
    Parameters:
        func (callable): Objective function F(x) returning a numpy array.
        x (array-like): Initial guess for the root.
        jac (callable): Jacobian matrix function J(x).
        tol (float): Tolerance for termination.
        maxiter (int): Maximum number of iterations.
    ------------------------------------------------------------------
    Returns:
        x (numpy.ndarray): Estimated root.
        info (dict): Convergence info containing:
            - 'converged' (bool): Whether the solver converged.
            - 'iterations' (int): Number of iterations performed.
            - 'final_residual' (float): Final residual norm.
    '''
    # check the input
    if not callable(func) or not callable(jac):
        raise TypeError("func and jac must be callable")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    if tol <= 0:
        raise ValueError(f"Tolerance must be positive, got {tol}")
    if maxiter < 1:
        raise ValueError(f"maxiter must be >=1, got {maxiter}")

    converged = False
    Fx = func(x)
    residual = np.linalg.norm(Fx)
    if residual <= tol:
        return x, {'converged': True, 'iterations': 0, 'final_residual': residual}
    convergence = []

    for it in range(maxiter):
        Jx = jac(x)
        try:
            delta_x = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            if np.linalg.norm(Jx) < 1e-8:
                delta_x = -0.1 * Fx  # 采用固定步长
            #   delta_x = np.linalg.lstsq(Jx, -Fx, rcond=None)[0]
            
        x += delta_x
        Fx = func(x)
        residual = np.linalg.norm(Fx)
        convergence.append(residual)
        if residual <= tol:
            converged = True
            break
    
    if plot == True:
        fig, ax = plt.subplots()
        ax.plot(range(len(convergence)), convergence, '-o')
        ax.axhline(y=0, color='darkred', linestyle='--', linewidth=2)
        ax.set_xlabel('Iteration times')
        ax.set_ylabel('Residual')
        plt.show()

    info = {
        'converged': converged,
        'iterations': it + 1,
        'final_residual': residual
    }
    return x, info


