# Author: Marc Laberge

from lab02.linalg_interp import gauss_iter_solve

import numpy as np

#Testing the Gauss solver

#Create a new solver 

def test_gauss_iter_solve():
    #Creating an example A matrix
    A = np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9],
    ])
    #Creating an example b matrix
    b = np.array([10,11,12])
    #Creating an example guess
    testGuess = np.array([1,1,1])

    #Testing Jacobi with an initial guess
    sol_jac = gauss_iter_solve(A, b, x0 = testGuess, alg = 'jacobi')
    #Using function from numpy
    sol_by_np = np.linalg.solve(A, b)
    #Print the result... see if they're the same
    print(f"The Jacobi solution with an initial guess: {sol_jac} \nThe numpy solution using np.linalg.solve: {sol_by_np}\n")

    #Testing Jacobi withOUT an initial guess
    sol_jac = gauss_iter_solve(A, b, x0 = None, alg = 'jacobi')
    #Using function from numpy
    sol_by_np = np.linalg.solve(A, b)
    #Print the result... see if they're the same
    print(f"The Jacobi solution withOUT an initial guess: {sol_jac} \nThe numpy solution using np.linalg.solve: {sol_by_np} \n")

    #Testing Siedel with an initial guess
    sol_sied = gauss_iter_solve(A, b, x0 = testGuess, alg = 'seidel')
    #Using function from numpy
    sol_by_np = np.linalg.solve(A,b)
    print(f"The Siedel soltuion with an initial guess: {sol_sied} \nThe numpy solution using np.linalg.solve: {sol_by_np} \n")

    #Testing Siedel withOUT an initial guess
    sol_seid = gauss_iter_solve(A, b, x0 = None, alg = 'seidel')
    #Using function from numpy
    sol_by_np = np.linalg.solve(A,b)
    print(f"The Siedel solution withOUt an initial guess: {sol_seid} \nThe numpy solution using np.linalg.solve: {sol_by_np} \n")

    #Testing a RHS vector to get the result A^-1
    b = np.eye(len(A))
    x = gauss_iter_solve(A,b)
    #Using function from numpy
    sol_x = np.linalg.solve(A,b)
    #Print the result... see if they're the same
    print(f"The Siedel solution: {x} \nThe numpy solution using np.linalg.solve: {sol_x}")
    #To make sure we got the right A^-1
    I = A @ x
    print("[A][A^-1] = \n {np.round(I, decimals = 0)}")

if __name__ == "__main__":
    test_gauss_iter_solve()


