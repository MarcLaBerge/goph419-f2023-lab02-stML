#Author : Marc Laberge

import numpy as np

NONE = 0
MAX_ITERATIONS = 100


def guess_iter_solve(A, b, x0, tol, alg):
    """
    Using iterative Gauss-Seidel to solve a linear system
    
    Parameters
    ----------
    A:
        An array containing the coefficient matrix
    b:
        An array containing the right-hand-side vector(s)
    x0: (optional)
        An array containing initial guess(es)
        Defult guess is 0 or None
        When a value is entered it should be the same shape as b
        Raise error if A,b,x0 shapes arn't compatable
    tol: (optional)
        A float of felative error tolerance or stopping criterial
        Defult parameter = 1e-8
        Raise RuntimeWarning if the solution doesn't converge after a specified number of iterations
    alg: (optional)
        A string flag for the algorithm to be used
        Option for 'seidel' or 'jacobi' iteration algorithm should be used
        Raise error if neither option above is entered

    Returns
    -------
    
    numpy.ndarray:
        Should be the shape as b
    """

    #Default
    x0 = NONE
    tol = 1e-8
    alg = 'seidel'


    #Raising Errors
        #array like check
    A = np.array(A, dtype = float)
    b = np.array(b, dtype = float)

    #Checking if A is 2D
    m = len(A) #returns number of rows
    ndim = len(A.shape)
    if ndim != 2: #has to be a 2d array
        raise ValueError(f"A has {ndim} dimensions" + ", should be 2")

    #Checking that A is square
    if A.shape[1] != m: #Checking that the amounto of columns in row 1 is the same as the amount of rows
        raise ValueError(f"A has {m} rows and {A.shape[1]} cols" + ", should be square")
    
    #Checking if b is 1D 
    ndimb = len(b.shape)
    if ndimb != 1 or ndimb != 2:
        raise ValueError(f"b has {ndimb} dimensions" + ", should be 1D or 2D")
    
    #Checking that A and b are compatable
    n = len(b) #amount of rows
    if n != m:
        raise ValueError(f"A has {m} rows, B has {n} values" + ", dimensions incompatible")
    
    #Checking that 'seidel' or 'jacobi' is selected for alg
    s = 'seidel'
    sLow = s.lower()
    if "seidel" in sLow:
        alg = 'seidel'
        print("You selected %s" %alg)
    elif "jacobi" in sLow:
        alg = 'jacobi'
        print("You selected %s" %alg)
    else:
        raise ValueError("Entered a string other than 'seidel' or 'jacobi'")



    #Creating x0
    if x0 == NONE:
        x0 = np.zeros((n,len(b[1])), dtype = int)#.reshape((len(b[0]),m))
        #This should give x0 whatever shape b is
    











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
