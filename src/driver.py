# Author: Marc Laberge



import numpy as np
import matplotlib.pyplot as plt
from linalg_interp import spline_function


def main():
    #Import the data

    #Starting with water
    water_data = np.loadtxt('data/water_density_vs_temp_usgs.txt')

    # set x and y values from columns in the file
    xWater = water_data[:, 0]
    yWater = water_data[:, 1]

    # Get 100 evenly spaced out points using linspace
    x_water = np.linspace(np.min(xWater), np.max(xWater), 100)

        #Find the interpolation using the first order spline function
    f_water1 = spline_function(xWater, yWater, order = 1)
    y_water1 = f_water1(x_water)


    # set up all subplots
    fig, axs = plt.subplots(nrows= 2, ncols = 3, figsize = (16, 12))
    fig.suptitle('Spline Interpolation for Water and Air Density Data')


    # plot first order water density interpolation. top right [0][0] position
    axs[0][0].plot(x_water, y_water1, 'b--', label = 'linear spline')
    axs[0][0].plot(xWater, yWater, 'ko', label = 'discrete data')
    axs[0][0].set_ylabel('Water density (g/cm^3)')
    axs[0][0].set_xlabel('Temperature (C)')
    axs[0][0].set_title('First Order')
    axs[0][0].legend()


        #Find the interpolation using 2nd order spline function
    f_w2 = spline_function(xWater, yWater, order = 2)
    y_water2 = f_w2(x_water)


    # plot second order water density interpolation. second top position [0][1]
    axs[0][1].plot(x_water, y_water2, 'b--', label = 'quadratic spline')
    axs[0][1].plot(xWater, yWater, 'ko', label = 'discrete data')
    axs[0][1].set_ylabel('Water density (g/cm^3)')
    axs[0][1].set_xlabel('Temperature (C)')
    axs[0][1].set_title('Second Order')
    axs[0][1].legend()


        #Find the interpolation using 3rd order spline function
    f_w3 = spline_function(xWater, yWater, order = 3)
    y_water3 = f_w3(x_water)


    # plot third order water density interpolation. third top position [0][2]
    axs[0][2].plot(x_water, y_water3, 'b--', label = 'cubic spline')
    axs[0][2].plot(xWater, yWater, 'ko', label = 'discrete data')
    axs[0][2].set_ylabel('Water density (g/cm^3)')
    axs[0][2].set_xlabel('Temperature (C)')
    axs[0][2].set_title('Third Order')
    axs[0][2].legend()
##########################################################################################
    #Import the air density data

    air_density_data = np.loadtxt('data/air_density_vs_temp_eng_toolbox.txt')

    # set up x and y values, from columns in the file
    xAir = air_density_data[:, 0]
    yAir = air_density_data[:, 1]

    #Get 100 evenly spaced out points using linspace
    x_air = np.linspace(np.min(xAir), np.max(xAir), 100)

        #Find the interpolation using the 1st order spline function
    f_a1 = spline_function(xAir, yAir, order = 1)
    y_air1 = f_a1(x_air)

    # plot first order air density interpolation
    axs[1][0].plot(x_air, y_air1, 'k--', label = 'linear spline')
    axs[1][0].plot(xAir, yAir, 'ro', label = 'discrete data')
    axs[1][0].set_ylabel('Air density (kg/m^3)')
    axs[1][0].set_xlabel('Temperature (C)')
    axs[1][0].set_title('First Order')
    axs[1][0].legend()

        #Find the interpolation using the 2nd order spline function
    f_a2 = spline_function(xAir, yAir, order = 2)
    y_air2 = f_a2(x_air)

    # plot second order air density interpolation
    axs[1][1].plot(x_air, y_air2, 'k--', label = 'quadratic spline')
    axs[1][1].plot(xAir, yAir, 'ro', label = 'discrete data')
    axs[1][1].set_ylabel('Air density (kg/m^3)')
    axs[1][1].set_xlabel('Temperature (C)')
    axs[1][1].set_title('Second Order')
    axs[1][1].legend()

        #Find the interpolation using the 3rd order spline function
    f_a3 = spline_function(xAir, yAir, order = 3)
    y_air3 = f_a3(x_air)

    # plot third order air density interpolation
    axs[1][2].plot(x_air, y_air3, 'k--', label = 'cubic spline')
    axs[1][2].plot(xAir, yAir, 'ro', label = 'discrete data')
    axs[1][2].set_ylabel('Air density (kg/m^3)')
    axs[1][2].set_xlabel('Temperature (C)')
    axs[1][2].set_title('Third Order')
    axs[1][2].legend()

    plt.savefig('figures/density_vs_temperature_graphs')

if __name__ == "__main__":
    pass