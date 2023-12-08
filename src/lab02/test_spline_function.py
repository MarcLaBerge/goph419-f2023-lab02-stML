# Author: Marc Laberge

import numpy as np
import matplotlib.pyplot as plt
from linalg_interp import spline_function
from scipy.interpolate import UnivariateSpline

def test_spline_function():
    #Testing the result of our splice function vs the expected functions
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
    axs[0].plot(x, y1, 'b', label = 'linear equation')
    axs[0].legend()

    axs[1].plot(x, quadratic, 'bo', label = 'quadractic spline')
    axs[1].plot(x, y2, 'b', label = 'quadratic equation')
    axs[1].legend()

    axs[2].plot(x, cubic, 'bo', label = 'cubic spline')
    axs[2].plot(x, y3, 'b', label = 'cubic equation')
    axs[2].legend()
    
    # check spline_function(order = 3) against scipy.interpolate.UnivariateSpline() function
    xd = np.linspace(1, 10, 50)
    yd = np.exp(xd)

    f_scipy = UnivariateSpline(xd, yd, k = 3, s = 0, ext = 'raise')
    y_scipy = f_scipy(xd)

    f_spline = spline_function(xd, yd, order = 3)
    y_spline = f_spline(xd)

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16,12))

    axs[0].plot(xd, yd, 'ko', label = 'data')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    axs[1].plot(xd, yd, 'ko', label = 'data')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')

    axs[0].plot(xd, y_scipy, 'b', label = 'scipy interpolation')
    axs[1].plot(xd, y_spline, 'm', label = 'GOPH419 spline_function')

    fig.suptitle('Scipy Univariate Spline Function vs GOPH419 Spline Function')
    fig.set_label('y')
    axs[0].legend()
    axs[1].legend()

    plt.savefig('figures/spline_test2')
    plt.savefig('figures/spline_test1')
    
if __name__ == "__main__":
    test_spline_function()