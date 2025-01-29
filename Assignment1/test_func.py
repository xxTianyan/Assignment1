import functions as f
import numpy as np

def test_mean():
    """
    Test the correctness of mean function
    """
    x = 10
    y = 20
    assert np.isclose(f.mean(x,y), 15)
    
    x = 11
    y = 11.5
    assert np.isclose(f.mean(x,y), 11.25)

    x = 0
    y = 0
    assert np.isclose(f.mean(x,y), 0)

    x = -100
    y = -99
    assert np.isclose(f.mean(x,y), -99.5)

    x = 1e-2
    y = 1e3
    assert np.isclose(f.mean(x,y), 500.005)


def test_computeDis():
    '''
    Test the computeDis function.
    '''
    x = 100
    y = -100
    assert np.isclose(f.computeDis(x,y), 200)

    x = 0.000000001
    y = 999999
    assert np.isclose(f.computeDis(x,y), 999998.999999999)

    x = 1/2
    y = 1/3
    assert np.isclose(f.computeDis(x,y), 1/6)

def test_correctSign():
    '''
    Test correctSign function
    '''
    x = 1
    y = 1
    # f.correctSign(x, y)

    x = 0
    y = 1
    # f.correctSign(x, y)

    x = -1
    y = 10
    f.correctSign(x, y)

def test_isClose():
    '''
    Test isClose function.
    '''
    t = 0
    a = 1
    re = f.isClose(a, t)
    assert re == False

    t = 1e-9
    a = 1e-10
    re = f.isClose(a, t)
    assert re == True

    t = -1
    a = 0.001
    f.isClose(a, t)
    assert re == True

    t = 1e-3
    a = 1e-4
    re = f.isClose(a,t)
    assert re == True

    t=1e-4
    a=-0.1
    f.isClose(a,t)
    re = f.isClose(a,t)
    assert re == False


def test_bisection():
    def func(x):
        return x**2 - 2
    solver = f.CreateBisecSolver(func, tolerance=1e-6)
    a = -3
    b = 0
    re = solver(a, b)
    assert f.isClose(re+np.sqrt(2), 1e-6)

    a = 4
    b = 1
    re = solver(a,b)
    assert f.isClose(re-np.sqrt(2), 1e-3)


if __name__ == '__main__':

    test_mean()
    test_computeDis()
    test_correctSign()
    test_isClose()
    test_bisection()

    