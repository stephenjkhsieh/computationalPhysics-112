"""

Functions to solve linear systems of equations.

Kuo-Chuan Pan
2024.05.05

"""
import numpy as np

def solveLowerTriangular(L,b):
    """
    Solve a linear system with a lower triangular matrix L.

    Arguments:
    L -- a lower triangular matrix
    b -- a vector

    Returns:
    x -- the solution to the linear system
    """
    n  = len(b)
    x  = np.zeros(n)
    bs = np.copy(b)
    bs = bs.astype(float)
    
    # TODO: implement the algorithm
    for i in np.arange(n):
        x[i] = bs[i] / L[i,i]
        bs[i+1:] -= np.dot(x[i], L[i+1:,i])
    return x


def solveUpperTriangular(U,b):
    """
    Solve a linear system with an upper triangular matrix U.

    Arguments:
    U -- an upper triangular matrix
    b -- a vector

    Returns:
    x -- the solution to the linear system

    """
    n  = len(b)
    x  = np.zeros(n)
    bs = np.copy(b)
    bs = bs.astype(float)
    
    # TODO: implement the algorithm
    for i in np.arange(n-1,-1,-1):
        x[i] = bs[i] / U[i,i]
        bs[:i] -= np.dot(x[i], U[:i,i])
    
    return x


def lu(A):
    """
    Perform LU decomposition on a square matrix A.

    Arguments:
    A -- a square matrix

    Returns:
    L -- a lower triangular matrix
    U -- an upper triangular matrix

    """
    n  = len(A)
    L  = np.identity(n)
    U  = np.zeros((n,n))
    # M  = np.zeros((n,n))
    As = np.copy(A)

    # TODO
    for k in np.arange(n):
        if As[k,k] == 0:
            raise ValueError(f"A[{k},{k}] is singular")
        
        for i in np.arange(k+1,n):
            L[i,k] = As[i,k] / As[k,k]

        for j in np.arange(k+1,n):
            for i in np.arange(k+1,n):
                As[i,j] -= L[i,k] * As[k,j]
        
        for i in np.arange(n):
            # L[i, :i] = M[i, :i]
            U[i, i:] = As[i, i:]

    return L, U


def lu_solve(A,b):
    """
    Solve a linear system with a square matrix A using LU decomposition.

    Arguments:
    A -- a square matrix
    b -- a vector

    Returns:
    x -- the solution to the linear system

    """

    x = np.zeros(len(b))

    l, u = lu(A)

    # TODO
    # L y = b
    y = solveLowerTriangular(l, b)

    # U x = y
    x = solveUpperTriangular(u, y)

    return x