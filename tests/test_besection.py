import unittest
import numpy as np
from algorithms.bisection import mean, computeDis, isClose, correctSign, diffSign, CreateBisecSolver

class TestBisectionMethods(unittest.TestCase):
    
    def test_mean(self):
        self.assertAlmostEqual(mean(2, 4), 3.0)
        self.assertAlmostEqual(mean(-3, 3), 0.0)
        self.assertAlmostEqual(mean(5.5, 2.5), 4.0)
    
    def test_computeDis(self):
        self.assertAlmostEqual(computeDis(5, 2), 3.0)
        self.assertAlmostEqual(computeDis(-3, 3), 6.0)
        self.assertAlmostEqual(computeDis(7.5, 7.5), 0.0)
    
    def test_isClose(self):
        self.assertTrue(isClose(0.0001, 0.001))
        self.assertFalse(isClose(0.01, 0.001))
        self.assertTrue(isClose(0.0, 0.001))
    
    def test_correctSign(self):
        with self.assertRaises(AssertionError):
            correctSign(3, 4)
        with self.assertRaises(AssertionError):
            correctSign(-2, -5)
        with self.assertRaises(AssertionError):
            correctSign(0, 5)
    
    def test_diffSign(self):
        self.assertTrue(diffSign(-3, 3))
        self.assertFalse(diffSign(2, 5))
        self.assertFalse(diffSign(-1, -4))
    
    def test_bisection_solver(self):
        def test_function(x):
            return x**3 - x - 2  # Root around x = 1.521
        
        solver = CreateBisecSolver(test_function, tolerance=1e-6)
        root = solver(1, 2)
        self.assertAlmostEqual(root, 1.521, places=3)
        
        def linear_function(x):
            return x - 5  # Root at x = 5
        
        solver = CreateBisecSolver(linear_function, tolerance=1e-6)
        root = solver(0, 10)
        self.assertAlmostEqual(root, 5.0, places=6)

if __name__ == '__main__':
    unittest.main()
