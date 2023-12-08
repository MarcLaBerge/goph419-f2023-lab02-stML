# Author: Marc Laberge

import numpy as np
import matplotlib.pyplot as plt
from linalg_interp import spline_function
from scipy.interpolate import UnivariateSpline

def test_spline_function():
    #Testing the result of our splice function vs a premade function
    #First find evenly spaces intervals

    x = np.linspace(-10, 10, 10)

    #Create some example functions
    #Linear
    linear = 0.5 * x
    #Quadratic
    quadratic = 4 * (x ** 2) + 5 * x +10
    #Cubic
    cubic = 2 * (x ** 3) + 4 * (x ** 2) + 5 * x + 10

    #Use interpolation for find the y values (spline function)
    #Linear
    f1 = spline_function(x, linear, order = 1)
    y1 = f1(x)

    #Quadratic
    f2 = spline_function(x, quadratic, order = 2)
    y2 = f2(x)

    #Cubic
    f3 = spline_function(x, cubic, order = 3)
    y3 = f3(x)

    #Plotting each value
    fig, axs = plt.subplots(nrows= 1, ncols = 3, figsize = (16, 12))
    fig.suptitle('Spline Interpolation vs precreated spline function')
    

    axs[0].plot(x, linear, 'bo', label = 'linear spline')
    axs[0].plot(x, y1, 'b', label = 'premade linear spline')
    axs[0].legend()

    axs[1].plot(x, quadratic, 'bo', label = 'quadractic spline')
    axs[1].plot(x, y2, 'b', label = 'premade quadractic spline')
    axs[1].legend()

    axs[2].plot(x, cubic, 'bo', label = 'cubic spline')
    axs[2].plot(x, y3, 'b', label = 'premade cubic spline')
    axs[2].legend()
    



    
    """
    plt.plot(x, linear, 'ro', label = 'linear function')
    plt.plot(x, y1, 'r', label = 'linear interpolation')

    plt.plot(x, quadratic, 'gx', label = 'quadratic function')
    plt.plot(x, y2, 'g', label = 'quadratic interpolation')

    plt.plot(x, cubic, 'mD', label = 'cubic function')
    plt.plot(x, y3, 'm', label = 'cubic interpolation')

    plt.title('Spline Function Test for Linear, Quadratic, and Cubic Functions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    """
    plt.savefig('figures/spline_test1')
if __name__ == "__main__":
    test_spline_function()