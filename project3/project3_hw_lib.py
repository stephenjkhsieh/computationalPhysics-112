# %%
import numpy as np
from numba import njit, prange
from scipy.sparse import dia_array
import matplotlib.pyplot as plt

def generate_1_4_1_dia(usize=4):
    ex = np.ones(usize)
    data = np.array([-ex, 4 * ex, -ex])
    offsets = np.array([-1, 0, 1])
    A = dia_array((data, offsets), shape=(usize, usize)).toarray()
    return A

def generate_the_laplace_matrix_with_size(N=4):
    """
    assume sqrt(N) is an integer.

    """
    nsq = N*N
    A   = np.zeros((nsq,nsq))
    u   = generate_1_4_1_dia(N)

    d_matrix = generate_1_4_1_dia(N)
    o_matrix = -np.identity(N)
    z_matrix = np.zeros((N,N))

    init_matrix_kernel(N,A,u,d_matrix,o_matrix,z_matrix)
    return A

@njit(parallel=True)
def init_matrix_kernel(N,A,u,d_matrix,o_matrix, z_matrix):
    for i in prange(N):
        for j in range(N):
            if u[i,j] == 4:
                # the elemen is another 1_4_1 matrix
                suba = d_matrix
            elif u[i,j] == -1:
                suba = o_matrix
            else:
                suba = z_matrix
            
            for i2 in range(N):
                for j2 in range(N):
                    ii = i2 + i*N
                    jj = j2 + j*N
                    A[ii,jj] = suba[i2,j2]
    return A

def convert_solution(x):
    usize = np.sqrt(len(x))
    u = x.reshape(int(usize),int(usize)).transpose()
    return u

def generate_rho(N, xmin=-1, xmax=1, ymin=-1,ymax=1):
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)

    xx,yy  = np.meshgrid(x,y, indexing='ij')

    r1 = (xx + 1.5)**2 + yy**2
    r2 = (xx - 1.5)**2 + yy**2
    rho  = np.exp(-5/4*r1**2) + 3/2*np.exp(-r2**2)
    return xx,yy, rho

def generate_the_rhs_vector_with_size(N=4, rho=None, dx=1, dy=1):
    if rho is None:
        b = np.zeros(N*N)
        b[len(b)//2] = 1
        b = -b * dx*dy
    else:
        # b = np.zeros(N*N)
        b = rho.flatten()
        b = -b * dx*dy
        
        # Cos boundary condition
        b[-N:] += np.cos(3*np.pi*np.arange(1,N+1)/N)
        b[:N] -= np.cos(3*np.pi*np.arange(1,N+1)/N)
        b[::N] -= np.cos(3*np.pi*np.arange(1,N+1)/N)
        b[N-1::N] += np.cos(3*np.pi*np.arange(1,N+1)/N)
    return b

def generate_mash(N=4, buff=1, xmin=-1, xmax=1, ymin=-1, ymax=1):
    """
    Generate 2D mesh grid for solving Poisson equation.

    Parameters
    ----------
    N : int
        Number of grid points in x and y directions.
    buff : int
        Number of ghost cells around the domain.
    xmin : float
        Minimum value of x.
    xmax : float
        Maximum value of x.
    ymin : float
        Minimum value of y.
    ymax : float
        Maximum value of y.
    
    Returns
    -------
    xx : 2D numpy array
        Mesh grid in x direction.
    yy : 2D numpy array
        Mesh grid in y direction.
    dx : float
        Grid spacing in x direction.
    dy : float
        Grid spacing in y direction.
    u : 2D numpy array
        Initial guess for solution.
    """
    x = np.linspace(xmin, xmax, N+2*buff)
    y = np.linspace(ymin, ymax, N+2*buff)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xx, yy = np.meshgrid(x, y, indexing='ij')
    u = np.zeros((N+2*buff, N+2*buff))
    # # seed 210
    # np.random.seed(210)
    # u = -np.random.rand(N+2*buff, N+2*buff)
    return xx, yy, dx, dy, u

@njit(parallel=True)
def jacobi(u, uold, nx, ny, dx, dy, rho):
    """
    Solve Laplace equation using Jacobi method.

    Parameters
    ----------
    u : 2D numpy array
        Initial solution.
    uold : 2D numpy array
        Old solution.
    nx : int
        Number of grid points in x direction.
    ny : int
        Number of grid points in y direction.

    Returns
    -------
    u : 2D numpy array
        Updated solution.
    """

    for i in prange(1, nx-1):
        for j in range(1, ny-1):
            u[i, j] = 0.25 * (uold[i - 1, j] + uold[i + 1, j] + uold[i, j - 1] + uold[i, j + 1] - dx*dy*rho[i-1,j-1])
    return u

@njit(parallel=True)
def gauss_seidel(u, uold, nx, ny, dx, dy, rho):
    """
    Solve Laplace equation using Gauss-Seidel method.

    Parameters
    ----------
    u : 2D numpy array
        Initial solution.
    uold : 2D numpy array
        Old solution.
    nx : int
        Number of grid points in x direction.
    ny : int
        Number of grid points in y direction.

    Returns
    -------
    u : 2D numpy array
        Updated solution.
    """

    for i in prange(1, nx-1):
        for j in range(1, ny-1):
            u[i, j] = 0.25 * (u[i - 1, j] + u[i, j - 1] + uold[i + 1, j] + uold[i, j + 1] - dx*dy*rho[i-1,j-1])
    return u

def sor(u, uold, nx, ny, dx, dy, rho, w):
    """
    Solve Laplace equation using Successive Over-Relaxation (SOR) method.

    Parameters
    ----------
    u : 2D numpy array
        Initial solution.
    uold : 2D numpy array
        Old solution.
    nx : int
        Number of grid points in x direction.
    ny : int
        Number of grid points in y direction.
    w : float
        Relaxation parameter.

    Returns
    -------
    u : 2D numpy array
        Updated solution.
    """

    u = gauss_seidel(u, uold, nx, ny, dx, dy, rho)
    u = w * u + (1 - w) * uold
    return u

def relax(u, method='jacobi', tolerance=1e-4, maxiter=1e6, **kwargs):
    """
    Relax the solution using Jacobi, Gauss-Seidel, or SOR method.

    Parameters
    ----------
    u : 2D numpy array
        Initial solution.
    method : str
        Relaxation method ('jacobi', 'gauss-seidel', or 'sor').
    tolerance : float
        Convergence tolerance.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    u : 2D numpy array
        Relaxed solution.
    iters : 1D numpy array
        Number of iterations.
    errors : 1D numpy array
        Errors at each iteration.
    """

    nx, ny = u.shape
    n = nx * ny
    iters = np.arange(int(maxiter))
    errors = np.zeros(int(maxiter))

    # Cos boundary condition
    u[-1] = np.cos(3*np.pi*np.arange(1,nx+1)/nx)
    u[0] = -np.cos(3*np.pi*np.arange(1,nx+1)/nx)
    u[:,-1] = np.cos(3*np.pi*np.arange(1,ny+1)/ny)
    u[:,0] = -np.cos(3*np.pi*np.arange(1,ny+1)/ny)

    # # zero boundary condition
    # u[-1] = 0
    # u[0] = 0
    # u[:,-1] = 0
    # u[:,0] = 0

    if method == 'jacobi':
        iter_func = jacobi
    elif method == 'gauss-seidel':
        iter_func = gauss_seidel
    elif method == 'sor':
        iter_func = sor
    else:
        raise ValueError('Invalid method')
    
    for it in range(int(maxiter)):
        uold = u.copy()
        u = iter_func(u, uold, nx, ny, **kwargs)
        errors[it] = np.linalg.norm((u - uold)) / np.sqrt(n)
        if errors[it] < tolerance:
            break
    return u, iters[:it+1], errors[:it+1]

def plot_solution(xx,yy,u):
    plt.figure(1, figsize=(6,6))
    plt.pcolormesh(xx,yy,u)
    plt.axis('equal')
    plt.colorbar()
    plt.contour(xx,yy, u, 5, colors="w")
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.xticks(np.arange(-5,6,1))
    plt.yticks(np.arange(-5,6,1))
    plt.show()
    plt.close()
