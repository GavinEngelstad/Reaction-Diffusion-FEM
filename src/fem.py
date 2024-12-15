'''
Functions to solve a reaction diffusion system

To solve one using these, you should
    1. Create the stiffness and damping matricies for the triangulation
    with `build_matricies`. This is unique to the triangulation, so it
    can be reused
    2. Solve the system with `solve_fem`. This iterates until a specified
    end time
'''

## load some packages
from scipy import sparse
import numpy as np


## build the matricies for a trinagulation
def _areas_2d(pts_T):
    # areas for trinagules on a 2d plane
    xi, yi = pts_T[:, :, 0], pts_T[:, :, 1]  # get i, j is rolled -1, k is rolled 1
    A_T = (xi * np.roll(yi, -1, axis=1) - xi * np.roll(yi, 1, axis=1)).sum(axis=1) / 2

    return A_T


def _areas_3d(pts_T):
    # areas for trinagules on a 3d surface
    # directly implements cross product formula
    row_sum = lambda i, j: (pts_T[:, :, i] * np.roll(pts_T[:, :, j], -1, axis=1) - pts_T[:, :, i] * np.roll(pts_T[:, :, j], 1, axis=1)).sum(axis=1)
    A_T = np.sqrt(row_sum(0, 1)**2 + row_sum(1, 2)**2 + row_sum(2, 0)**2) / 2

    return A_T


def _ii_stiffness_2d(pts_T, A_T):
    # implments the euqation for the stiffness matrix for psi_i**2 in 2d
    xi, yi = pts_T[:, :, 0], pts_T[:, :, 1]  # get i, j is rolled -1, k is rolled 1
    stiff_data = (((np.roll(xi, -1, axis=1) - np.roll(xi, 1, axis=1))**2 + (np.roll(yi, 1, axis=1) - np.roll(yi, -1, axis=1))**2) / 4 / A_T[:, None]).ravel()

    return stiff_data


def _grad_psi(xi, xj, xk, yi, yj, yk, zi, zj, zk):
    # gradient of psi
    ## differences
    ij = np.array([xi - xj, yi - yj, zi - zj])
    jk = np.array([xj - xk, yj - yk, zj - zk])

    ## lambda
    lamb = (ij * jk).sum(axis=0) / (jk**2).sum(axis=0)
    w = ij - lamb * jk
    magw_sqr = (w**2).sum(axis=0)
    grad_psi = w / magw_sqr

    ## return
    return grad_psi


def _ii_stiffness_3d(pts_T, A_T):
    # implments the euqation for the stiffness matrix for grad psi_i**2 in 3d
    xi, yi, zi = pts_T[:, :, 0], pts_T[:, :, 1], pts_T[:, :, 2]  # get i, j is rolled -1, k is rolled 1
    grad_psi = _grad_psi(xi, np.roll(xi, -1, axis=1), np.roll(xi, 1, axis=1), yi, np.roll(yi, -1, axis=1), np.roll(yi, 1, axis=1), zi, np.roll(zi, -1, axis=1), np.roll(zi, 1, axis=1))
    stiff_data = (A_T[:, None] * (grad_psi**2).sum(axis=0)).ravel()

    return stiff_data


def _ij_stiffness_2d(pts_T, A_T, i, j, k):
    # implements the equation for the stiffness matrix for grad psi_i * grad psi_j in 2d
    xi, yi = pts_T[:, i].T.swapaxes(1, 2)  # get xs and ys
    xj, yj = pts_T[:, j].T.swapaxes(1, 2)
    xk, yk = pts_T[:, k].T.swapaxes(1, 2)
    stiff_data = (((yj - yk) * (yk - yi) + (xk - xj) * (xi - xk)) / 4 / A_T[:, None]).ravel()

    return stiff_data


def _ij_stiffness_3d(pts_T, A_T, i, j, k):
    # implements the equation for the stiffness martic for grad psi_i * grad psi_j in 3d
    xi, yi, zi = pts_T[:, i].T.swapaxes(1, 2)  # get xs and ys
    xj, yj, zj = pts_T[:, j].T.swapaxes(1, 2)
    xk, yk, zk = pts_T[:, k].T.swapaxes(1, 2)
    grad_psi_i = _grad_psi(xi, xj, xk, yi, yj, yk, zi, zj, zk)
    grad_psi_j = _grad_psi(xj, xk, xi, yj, yk, yi, zj, zk, zi)
    stiff_data = (A_T[:, None] * (grad_psi_i * grad_psi_j).sum(axis=0)).ravel()

    return stiff_data


def build_matricies(pts, tris):
    '''
    pts: Verticies in the triangulation
    tris: Indicies of pts for the triangulation
    '''
    ## setup
    pts_T = pts[tris]
    n_pts, dim = pts.shape
    
    ## areas
    match dim:
        case 2:
            A_T = _areas_2d(pts_T)
            ii_stiffness = _ii_stiffness_2d
            ij_stiffness = _ij_stiffness_2d
        case 3:
            A_T = _areas_3d(pts_T)
            ii_stiffness = _ii_stiffness_3d
            ij_stiffness = _ij_stiffness_3d
        case _:
            raise ValueError('Use only 2 or 3 dimensional points.')

    ## i, i pairs
    rows = tris.ravel()
    cols = tris.ravel()
    damp_data = np.repeat(A_T, 3) / 6  # formulas
    stiff_data = ii_stiffness(pts_T, A_T)

    ## i, j pairs
    i = np.array([0, 0, 1, 1, 2, 2])  # i, j pairs we need to work with
    j = np.array([1, 2, 0, 2, 0, 1])
    k = 3 - i - j  # other vertex, i + j + k = 3 always
    rows = np.hstack((rows, tris[:, i].ravel()))
    cols = np.hstack((cols, tris[:, j].ravel()))
    damp_data = np.hstack((damp_data, np.repeat(A_T, 6) / 12))  # formulas
    stiff_data = np.hstack((stiff_data, ij_stiffness(pts_T, A_T, i, j, k)))

    ## make into a matrix
    # when you make a sparse matrix, if i, j is in there twice it adds the data
    # abuse this to avoid adding accross triangles
    damp_mat = sparse.csr_array((damp_data, (rows, cols)), shape=(n_pts, n_pts))
    stiff_mat = sparse.csr_array((stiff_data, (rows, cols)), shape=(n_pts, n_pts))

    return damp_mat, stiff_mat



## --- Solve FEM ---
def solve_fem(r, gamma, t_max, dt, u0, damp_mat, stiff_mat, return_all=False):
    '''
    r: Reaction function. Takes n_pts by N matrix of u values and returns n_pts
    by N matrix of reactions
    gamma: Diffusion coefficeints. Should be N of them
    t_max: Maximium time we calculate to
    dt: Time step each iteration
    u0: Inital condition
    damp_mat: Damping matrix
    stiff_mat: Stiffness matrix
    return_all: Return the progression at each time step. Uses a lot of memory
    '''
    ## setup 
    n_pts, N = u0.shape
    num_iter = int(np.ceil(t_max / dt) + 1)
    u = 1 * u0
    if return_all:
        u_all = np.empty((num_iter, n_pts, N))
        u_all[0] = u0

    ## loop, explicit euler
    lhs = [damp_mat + dt * g * stiff_mat for g in gamma]  # lhs is constant the whole time
    for i in range(1, num_iter):
        # setup problem
        ri = r(u)  # reaction
        rhs = damp_mat @ (u + dt * ri)

        # solve problem
        for n in range(N):
            u[:, n] = sparse.linalg.cg(lhs[n], rhs[:, n], x0=u[:, n])[0]
        
        # store it
        if return_all:
            u_all[i] = u
        
    if return_all:
        return u_all
    return u


## --- Evaluate Function ---
def _psi_T(x, y, xj, xk, yj, yk, A_T):
    ## evaluate x and y on the plane in trinagle T
    psi = (xj * yk - xk * yj + (yj - yk) * x[..., None] + (xk - xj) * y[..., None]) / 2 / A_T

    return psi


def _u_func(x, y, tris, A_T, xi, xj, xk, yi, yj, yk, u, tol):
    ## u_func implemented in a for loop
    # setup
    uxy = np.empty_like(x)

    # loop
    for i in np.ndindex(uxy.shape):
        psi_i = _psi_T(x[i], y[i], xj, xk, yj, yk, A_T)
        psi_j = _psi_T(x[i], y[i], xk, xi, yk, yi, A_T)
        psi_k = _psi_T(x[i], y[i], xi, xj, yi, yj, A_T)

        ## get triangle point is in
        lb = 0 - tol
        ub = 1 + tol
        tri_idx = ((lb <= psi_i) & (psi_i <= ub) & (lb <= psi_j) & (psi_j <= ub) & (lb <= psi_k) & (psi_k <= ub)).argmax()
        psi = np.stack((psi_i[tri_idx], psi_j[tri_idx], psi_k[tri_idx]), axis=-1)
        ui = u[tris[tri_idx]]

        # caluate result
        if (psi < lb).any() or (psi > ub).any():
            uxy[i] = np.nan
        else:
            uxy[i] = (psi * ui).sum()
    
    return uxy        


def u_func(x, y, pts, tris, u, tol = 1e-10, max_size=100000):
    '''
    Evaluates u at x, y
    x: x coordinate to evaluate at
    y: y coordinate to evaluate at
    pts: Points of the u values
    tris: Trinagle indicies
    u: U values at each pt
    tol: Error tolerance for convexity, just use the default  
    max_size: If bigger than this, uses a for loop instead of numpy arrays
    for memory reasons, just use the default
    '''
    ## setup
    pts_T = pts[tris]

    ## evaluate function at each point
    A_T = _areas_2d(pts_T)
    (xi, xj, xk), (yi, yj, yk) = pts_T.T
    if pts_T.shape[0] > max_size:  # size check before we extend the arrats
        return _u_func(x, y, tris, A_T, xi, xj, xk, yi, yj, yk, u, tol)
    psi_i = _psi_T(x, y, xj, xk, yj, yk, A_T)
    psi_j = _psi_T(x, y, xk, xi, yk, yi, A_T)
    psi_k = _psi_T(x, y, xi, xj, yi, yj, A_T)

    ## get triangle point is in
    lb = 0 - tol
    ub = 1 + tol
    tri_idx = ((lb <= psi_i) & (psi_i <= ub) & (lb <= psi_j) & (psi_j <= ub) & (lb <= psi_k) & (psi_k <= ub)).argmax(axis=-1)
    ij = tuple(np.meshgrid(*[np.arange(i) for i in x.shape]))[::-1] + (tri_idx,)
    psi = np.stack((psi_i[ij], psi_j[ij], psi_k[ij]), axis=-1)
    u = u[tris[tri_idx]]

    ## caluate result
    uxy = (psi * u).sum(axis=-1)
    uxy[(psi < lb).any(axis=-1) | (psi > ub).any(axis=-1)] = np.nan
    
    return uxy
