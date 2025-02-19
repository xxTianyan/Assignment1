"""
This python file contains biscection method and
all the basic functions that is needed for bissection methos.
Created by Tianyan XU, 1/28/2025
"""

import numpy as np
from typing import Callable

def mean(a:float, b:float) -> float:
    '''
    Calculate and return the mean value for two given float numbers.
    '''
    return (a+b) / 2.0

def computeDis(a:float, c:float) -> float:
    '''
    Calculate the absolute value of the difference between two given floats.
    '''
    val = c-a
    return np.abs(val)

def isClose(a:float, tolerance:float) -> bool:
    '''
    Check if the value is less than the specified tolerance.
    '''

    if np.abs(a) < tolerance:
        return True
    else:
        return False


def correctSign(a:float, b:float) -> None:
    '''
    Determine whether the given two floating-point numbers have the same sign.
    If two numbers have the same sign, it will assett a warning.
    '''
    val = a*b

    assert val < 0, f'{a} and {b} have the same sign or one of them is 0 value.'

def diffSign(a:float, b:float) -> bool:
    val = a*b
    if val < 0:
        return True


def CreateBisecSolver(f: Callable[[float], float], tolerance:float) -> Callable[[float, float], float]:
    '''
    Accepts a given callable function and tolerance and returns a bisection solver.
    The solver takes different intervals and find the 0 point between them.
    Parameters:
    -----------
    f : Callable[[float], float]
        The function for which the root is to be found.
    tolerance : float
        The stopping criterion for the algorithm; must be greater than zero.

    Returns:
    --------
    Callable[[float, float], float]
        A function that performs the bisection method on the interval `[a, b]` and returns the root.
    '''

    # tolerance should larger than zero
    assert tolerance > 0, f'Tolerance t={tolerance} should larger than 0.' 

    def solver(a:float, b:float) -> float:

        left = f(a)
        right = f(b)
        mid = np.inf

        # determine whether the interval is valid
        if isClose(right, tolerance):
            return b
        if isClose(left, tolerance):
            return a
        correctSign(right, left)

        while not isClose(mid, tolerance):
            # find mid point c
            c = mean(a,b)
            # check convergence
            mid = f(c)
            if isClose(mid, tolerance):
                return c
            else:
                if diffSign(left, mid):
                    b = c
                    right = f(b)
                else:
                    a = c
                    left = f(a)
    return solver
