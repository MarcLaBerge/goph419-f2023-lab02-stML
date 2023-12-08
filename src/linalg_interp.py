#Author : Marc Laberge

import numpy as np

NONE = 0
MAX_ITERATIONS = 100


def guess_iter_solve(A, b, x0 = None, tol = 1e-8, alg = 'seidel'):
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
    
    # Initialize an inital guess of zeros, if no intial guess provided and check that x and b have the same shape
    if not x0:
        x = np.zeros_like(b)
    else:
        x = np.array(x0,dtype=float)
        if x.shape != b.shape:
            raise ValueError(f"X has shape {x.shape}, b has shape {b.shape}"+ ", should be same length") 
    
    #Checking that 'seidel' or 'jacobi' is selected for alg
    s = alg.strip()
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
        x0 = np.zeros_like(b)
        #This should give x0 whatever shape b is
    else:
        #Checking if x0 is an array (np.array)
        x0 = np.array(x0, dtype=float)
        #Check that x0 is the same shape as b
        if x0.shape != b.shape:
            x0 = np.array_like(b)
        #Checking that x0 is has 1 or 2 Dimensions
        if x0.ndim not in [1,2]:
            raise ValueError(f"x0 has {x0.ndim} dimensions, should be 1D or 2D")
        #Checking that x0 has the same amount of rows as A and b
        if len(x0) != m:
            raise ValueError(f"x0 has {len(x0)} rows while A and b have {m} rows, systems are incompatitble")        

    
    #Seidel algorithm
        #iterations
        i = 0
        #Approxiamte relative error
        eps_a = 2*tol

        #Normalize matrix (coefficient and b vector)
        ADiagonal = np.diag(1.0/np.diag(A))
        bStar = ADiagonal @ b
        AStar = ADiagonal @ A
        A_s = AStar - np.eye(m)

    while np.max(eps_a) > tol and i < MAX_ITERATIONS:
        if alg == 'jacobi':
            x_old = np.array(x0)
            x0 = bStar - (AStar @ x_old)
        elif alg == 'seidel':
            x_old = np.array(x0)
            for i, j in enumerate(A):
                x0[i,:] = bStar[i:(i+1),:] - AStar[i:(i+1),:] @ x0
        # Error for each calculation
        num = x0 - x_old
        eps_a = np.linalg.norm(num) / np.linalg.norm(x0)
        i += 1
        if i >= MAX_ITERATIONS:
            raise RuntimeWarning(f"No convergence after {MAX_ITERATIONS} iterations, returning last updated x vector")
    
    return (x)



def spline_function(xd, yd, order = 3):
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
    #Check that values entered are compatable (arrays)
    xd = np.array(xd, dtype = float)
    yd = np.array(yd, dtype = float)

    #Check that xd and yd have the same length
    n = len(yd)
    m = len(xd)
    if (n != m):
        raise ValueError(f"The length of xd {m} is not equal to the length of yd {n}, to be compatible, xd and yd must be the same shape")
    
    u = np.unique(xd)
    k = len(u)
    if k != m:
        raise ValueError(f"The number of indipendent values {k} is not equal to the number of dependent variables {n}")

    #Check that the order is either 1, 2, or 3
    if order not in [1, 2, 3]:
        raise ValueError(f"invalid order {order}, must be 1, 2, 3")
    
    #Check that the values of xd are in the proper order
    if (all (xd[i] >= xd[i+1] for i in range(m-1))):
        raise ValueError(f"values of xd must be in increasing order")

    #b = f2 - f1 / x2 - x1
    #y is the f and x is the x
    diff_x = np.diff(xd)
    diff_y = np.diff(yd)
    div_dif_1 = diff_y / diff_x #First oder divided difference = b

    if order == 1:
        def spline_1(x):
            """
            Linear spline function

            Inputs
            --------
            x:
                float or array-like of floats
       
            Returns
            --------
            The interpolated value of y (or f in the notes) or multiple of their values
            """
            # ai = yi or fi
            a = yd[:-1]
            # b = first order divided difference
            b = div_dif_1[:-1]
            for xi in x:
                i = np.array([np.nonzero(xd >= xi)[0][0] - 1 for xi in x])
                i = np.where(i < 0, 0, i)
                # s = fi + (b)(x - xi)
                y = a[i-1] + b[i-1] * (x - xd[i-1])
            return y
        return spline_1
    
    elif order == 2:
        def spline_2(x):
            """
            Quadratic Spline Function
            
            Inputs
            --------
            x:
                float or array-like floats
                
            Returns
            --------
            Interpolated value of y (or f in the notes) or multiple of their values

            Unknows - a, b, c
            """
            # We need to solve for the unknowns... make a system
            # RHS of the system
            N = m - 1
            rhs = np.zeros(N)
            rhs[1:] = np.diff(div_dif_1, axis = 0)
            # set up coefficient matrix
            A = np.zeros((N, N))
            # values of first and last rows of A
            A[0,0:2] = [1, -1]
            A[1:,:-1] += np.diag(diff_x[:-1])
            A[1:,1:] += np.diag(diff_x[1:])
            # determine coefficients
            c = np.linalg.solve(A, rhs)
            # c = gauss_iter_solver(A, rhs)
            # first order divided difference - (c(solved by function)*(x(i+1)-xi)
            b = div_dif_1 - (c * diff_x)
            a = yd[:-1]
            # calculate spline functions
            for xi in x:
                # determine indexing intervals where spline function will interpolate between points
                i = np.array([np.nonzero(xd >= xi)[0][0] - 1 for xi in x])
                i = np.where(i < 0, 0, i)
                # spline function over at index i
                # s = ai + (bi)(x - xi) + ci(x-xi)^2
                y = a[i] + b[i] * (x - xd[i]) + c[i] * (x - xd[i]) ** 2
            return y
        return spline_2
    
    elif order == 3:
        def spline_3(x):
            """
            Cubic Spline Function
            
            Inputs
            --------
            x:
                float or array-like floats
                
            Returns
            --------
            Interpolated value of y (or f in the notes) or multiple of their values

            """
            # Linear system to solve for the unknowns
            # Setting up the right hand side
            N = m
            div_dif_2 = np.diff(div_dif_1)
            rhs = np.zeros(N)
            rhs[1:-1] = 3 * div_dif_2
            # set up coefficient matrix
            A = np.zeros((N, N))
            A[1, 0] = diff_x[0]
            A[-2, -1] = diff_x[-1]
            A[0,:3] = [-diff_x[1], (diff_x[0] + diff_x[1]), -diff_x[-2]]
            A[-1,-3:] = [-diff_x[-1], (diff_x[-1] + diff_x[-2]), -diff_x[-2]]
            A[1:-1,:-2] += np.diag(diff_x[:-1])
            A[1:-1,1:-1] += np.diag(2 * (diff_x[:-1] + diff_x[1:]))
            A[1:-1,2:] += np.diag(diff_x[1:])
            # calculate coefficients (solving for unknown "c")
            c = np.linalg.solve(A, rhs)
            #c = gauss_iter_solver(A, rhs)
            #Use c to then solve for the other unknowns (formulas in the slides)
            d = np.diff(c) / (diff_x * 3)
            b = div_dif_1 - diff_x * (c[:-1] + c[1:] * 2) / 3
            # get indexes for spline function interpolation
            i = np.array([np.nonzero(xd >= xi)[0][0] - 1 for xi in x])
            i = np.where(i < 0, 0, i)
            #Once we find the unknowns then we can get y (or f)
            y = np.array([(yd[i] + b[i] * (xi - xd[i]) + c[i] * (xi - xd[i]) ** 2 + d[i] * (xi - xd[i]) ** 3) for i, xi in zip(i, x)])
            return y
        return spline_3
    
            

