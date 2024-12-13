'''
Functions to make trinagulations of various shapes

The supported shapes currently include:
- Rectangle
- Disk
- Maze
- The surface of a sphere
- The surface of a torus

Each shape takes various shape-specific paramaters as well as an `N`. `N`
approximately decribes the number of points per length 1 distance along the
triangular mesh
'''

## --- Config --- 
# load some packages
from scipy import spatial
import networkx as nx
import numpy as np
import stripy  # triangulate the sphere


## --- 2d Shapes ---
def tri_rectangle(a, b, N):
    '''
    a: Width of rectangle
    b: Height of rectangle
    '''
    ## setup
    Nx = int(a * N)
    Ny = int(b * N)

    ## get points
    # make sure boundaries are included on both sides
    i = np.hstack((np.arange(0, Nx * 2, 2), Nx * 2 - 1))
    j = np.hstack((0, np.arange(1, Nx * 2, 2)))
    x1 = np.linspace(0, a, Nx * 2)[i]  # offset rhs
    x2 = np.linspace(0, a, Nx * 2)[j]  # offset lhs
    y1 = np.linspace(0, b, Ny)[::2]
    y2 = np.linspace(0, b, Ny)[1::2]
    x1, y1 = np.meshgrid(x1, y1)
    x2, y2 = np.meshgrid(x2, y2)
    n1 = x1.size
    pts = np.array((np.hstack((x1.ravel(), x2.ravel())), np.hstack((y1.ravel(), y2.ravel())))).T

    ## get triangles
    tris = spatial.Delaunay(pts).simplices
    # # bottom left corners
    # i = np.arange((Nx + 1) * int(np.ceil((Ny / 2)))).reshape((-1, Nx + 1))[:, :-1].ravel()[:, None]
    # end = len(i)-Nx if Ny % 2 == 1 else len(i)
    # tris = np.vstack((
    #         i[:end] + np.array([[0, n1, n1 + 1]]),
    #         i[:end] + np.array([[0, n1 + 1, 1]]),
    #         i[Nx:] + np.array([[0, n1 - Nx, n1 - Nx - 1]]),
    #         i[Nx:] + np.array([[0, n1 - Nx, 1]]),
    #     ))

    return pts, tris


def tri_circle(r, N):
    '''
    r: radius
    '''
    ## setup
    Nr = int(np.ceil(N * r))
    rads = r * np.linspace(0, 1, Nr)
    thetas = [np.linspace(0, 2 * np.pi, int(r * 2 * np.pi * N + 2))[:-1] for r in rads]

    ## points
    x = np.hstack([rads[i] * np.sin(thetas[i]) for i in range(Nr)])
    y = np.hstack([rads[i] * np.cos(thetas[i]) for i in range(Nr)])
    pts = np.array((x, y)).T

    ## triangles
    tris = spatial.Delaunay(pts).simplices
    # # basic idea: make triangle out of a (this doesnt work (idk why))
    # #   1. inital point
    # #   2. point clockwise of it
    # #   3. point in an adjacent layer with an angle between it
    # npts = 1  # number of points before layer 
    # tris = []
    # for r in range(1, Nr):  # for each layer
    #     ln = thetas[r - 1].size
    #     n = thetas[r].size

    #     # shared bits at current row
    #     i = np.arange(n)
    #     j = (i + 1) % n
    #     k = np.searchsorted(thetas[r - 1], thetas[r], side='left') % ln - ln
    #     tris.append(npts + np.array([i, j, k]).T)
    #     if r != Nr - 1:
    #         pn = thetas[r + 1].size
    #         k = np.searchsorted(thetas[r + 1], thetas[r], side='right') % pn + n
    #         tris.append(npts + np.array([i, j, k]).T)

    #     # iterate the number of vistied points
    #     npts += n
    # tris = np.vstack(tris)

    return pts, tris


def tri_maze(m, n, a, N, seed=None):
    '''
    m: # of columns in maze
    n: # of rows in maze
    a: size of each box in the maze (maze size is (2 * m - 1) * a by (2 * n - 1) * a)
    '''
    ## setup
    Na = int(np.ceil(N * a)) + 1

    ## random maze
    G = nx.random_spanning_tree(nx.grid_2d_graph(m, n), seed=seed)  # the maze we use
    maze = np.zeros((m * 2 - 1, n * 2 - 1))  # store it in a numpy array
    i, j = np.arange(0, m * 2, 2), np.arange(0, n * 2, 2)  # indicies of "nodes"
    i, j = np.meshgrid(i, j)
    maze[i, j] = 1
    i, j = np.array(G.edges).sum(axis=-2).T  # indicies of "edges"
    maze[i, j] = 1

    ## points
    # points on "nodes" go first so edges can connect them
    xg, yg = np.linspace(0, a, Na), np.linspace(0, a, Na)
    xg, yg = np.meshgrid(xg, yg)
    # xg, yg = xg.ravel(), yg.ravel()
    ig, jg = np.arange(0, m * 2, 2), np.arange(0, n * 2, 2)  # indicies of "nodes"
    ig, jg = np.meshgrid(ig, jg)
    ig, jg = ig.ravel(), jg.ravel()
    # points on "edges" go next
    ie, je = np.array(G.edges).sum(axis=-2).T  # indicies of "edges"
    ish = ie % 2 == 1  # vertical rows have 1 here
    ive, jve = ie[~ish], je[~ish]
    ihe, jhe = ie[ish], je[ish]
    pts = np.vstack((
            np.array(((xg.ravel() + ig[:, None] * a).ravel(), (yg.ravel() + jg[:, None] * a).ravel())).T,
            np.array(((xg[1:-1].ravel() + ive[:, None] * a).ravel(), (yg[1:-1].ravel() + jve[:, None] * a).ravel())).T,
            np.array(((xg[:, 1:-1].ravel() + ihe[:, None] * a).ravel(), (yg[:, 1:-1].ravel() + jhe[:, None] * a).ravel())).T,
        ))

    ## trinagules
    # trinagulate it
    tris = spatial.Delaunay(pts).simplices
    # remove edges not in maze
    center = (pts[tris].mean(axis=-2) / a).astype(int)  # centerpoint of trinagle
    tris = tris[maze[center[:, 0], center[:, 1]] == 1]

    return pts, tris


## --- 3d Shapes ---
def tri_sphere(r, refinement_level):
    '''
    r: radius
    refinement_level: refinement level 
    '''
    icoR = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=refinement_level)
    pts = icoR.points * r
    tris = icoR.simplices

    return pts, tris


def tri_torus(r, R, N):
    ## setup 
    Nu = N * R
    Nv = N * r

    ## paramterization
    u = np.linspace(0, 2 * np.pi, Nu + 1)[:-1]
    v = np.linspace(0, 2 * np.pi, Nv + 1)[:-1]
    u, v = np.meshgrid(u, v)
    u += np.tile([0, np.pi / (Nu + 1)], (N + 1) // 2)[:N, None]
    u = u.ravel()
    v = v.ravel()

    ## triangulation
    i, j = np.meshgrid(np.arange(Nu), np.arange(Nv, step=2))
    i, j = i.ravel(), j.ravel()
    iup, jup, jdown = (i + 1) % Nu, (j + 1) % Nv, (j - 1) % Nv
    tris = np.vstack((
            np.array([Nu * j + i, Nu * j + iup, Nu * jup + i]).T, np.array([Nu * jup + iup, Nu * j + iup, Nu * jup + i]).T,
            np.array([Nu * j + i, Nu * j + iup, Nu * jdown + i]).T, np.array([Nu * jdown + iup, Nu * j + iup, Nu * jdown + i]).T,
        ))

    ## points
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    pts = np.array([x, y, z]).T

    return pts, tris
