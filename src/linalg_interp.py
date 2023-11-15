#Author : Marc Laberge

def guess_iter_solve():
    """
    Using iterative Gauss-Seidel to solve a linear system
    
    Parameters
    ----------
    A:
        An array containing the coefficient matrix
    b:
        An array containing the right-hand-side vector(s)
    x0:
        An array containing initial guess(es)
        Defult guess is 0 or None
        When a value is entered it should be the same shape as b
        Raise error if A,b,x0 shapes arn't compatable
    tol:
        A float of felative error tolerance or stopping criterial
        Defult parameter = 1e-8
        Raise RuntimeWarning if the solution doesn't converge after a specified number of iterations
    alg:
        A string flag for the algorithm to be used
        Option for 'seidel' or 'jacobi' iteration algorithm should be used
        Raise error if neither option above is entered

    Returns
    -------
    
    numpy.ndarray:
        Should be the shape as b
    """
    pass

def spline_function():
    """
    A function that uses 2 given vectors x and y to generate a spline function

    Parameters
    ----------
    xd:
        An array of float data increasing in value
    yd:
        An array of float data with the same shape as xd
    order:
        Optional integer with values (1,2 or 3) 
        Defult of 3

    Returns
    -------

        Takes one parameter (float or an array of floats) and returns the interpolated y value(s)

    Raises
    ------
        -If the flattened arrays of xd and yd do not have the same length (The number of independant variables is not the same as dependant variables)
        -Repeated values in xd (The number of unique independant variables is not equal to the number of dependant variables)
        -If xd values are not in increasing order
        -If the order is something other than 1,2 or 3
        -If the imput parameter is outside of the range of xd

    """
    pass
