"""simulate Percolations on different graphs"""
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi

# Fixing random state for reproducibility
# np.random.seed(19680801)

# Standard coupling of edge percolation in a w by h rectangle of square lattice
# width w
# height h
# total number of edges is h*(w+1) vertical ones and (h+1)*w horizontal
# ones, edges are labeled with vertex, the vertex (i,j) is associate to
# two edges, the right one (i,j)--(i+1,j) is recorded in the element
# x[i,j,0],  the up one (i,j)--(i,j+1) is recorded in x[i,j,1].


def sample_unif_x(w: int, h: int) -> np.ndarray:
    return np.random.random((w + 1, h + 1, 2))


def compute_spin(x: np.ndarray, p: float) -> np.ndarray:
    """Return a numpy array with 0 or 1 values"""
    return np.where(x < p, 0, 1)


def simu_perco_square(w: int, h: int, p: float) -> np.ndarray:
    """
    This function will give a percolation sample in a rectangle of height h,
    width w, parameter p, the output is two array X[(h+1,w,0)], the
    horizontal edges and X[(h,w+1,1)] the vertical edges. order the
    horizontal edges by (i,j,0); i=1..w; j= 1..h+1; from the bottom left
    corner, i coordinate go to the right and j coordinate go to the top.
    order the vertical edges by (i,j,1); i=1..w+1; j=1..h, same coordinate
    as the horizontals.
    """
    return compute_spin(sample_unif_x(w, h), p)


def find_all_cluster(x: np.ndarray, w: int, h: int) -> np.array:
    """
    compute the clusters of a percolation sample x of width w
    and height h, on the square lattice,
    return a list cluster[i,j]=k where
    k indicate to which cluster the site (i,j) belongs to.
    
    order the vertices by order(i,j)=i+1+(w+1)j, that is left to right,
    bottom to top. the variable order record the next unvisited vertex
    """

    cluster = np.zeros((w + 1, h + 1), dtype=int)
    visited = np.full((w + 1, h + 1), False)
    k = 0  # cluster index
    myvertex = 1
    stack = []
    # as long as we havent treated the last myvertex, continue
    while myvertex < (w + 1) * (h + 1):
        # put the next site in myvertex in to the stack if the site is
        # unvisited, otherwise myvertex ++
        iv = (myvertex - 1) % (w + 1)
        jv = (myvertex - 1) // (w + 1)
        if not visited[iv, jv]:
            stack.append([iv, jv])
            k += 1  # increment cluster index
        else:
            myvertex += 1

        while stack:
            # pop the current myvertex from the stack and set its cluster label
            # to k and mark as visited
            i, j = stack.pop(0)
            cluster[i, j] = k
            visited[i, j] = True
            # check all of its 4 neighbors, if neighbor is unvisited and
            # connected to current site,
            # then set its cluster label to k and marked visited and
            # push this site into stack, otherwise do nothing
            # check the left neighbor, first coordinate must >0 to have a left
            # neighbor
            if i > 0 and not visited[i - 1, j] and x[i - 1, j, 0] == 1:
                cluster[i - 1, j] = k
                visited[i - 1, j] = True
                stack.append([i - 1, j])
            # check the right neighbor, first coordinate must <w to have a
            # right neighbor
            if i < w and not visited[i + 1, j] and x[i, j, 0] == 1:
                cluster[i + 1, j] = k
                visited[i + 1, j] = True
                stack.append([i + 1, j])
            # check the up neighbor, second coordinate must <h to have such a
            # neighbor
            if j < h and not visited[i, j + 1] and x[i, j, 1] == 1:
                cluster[i, j + 1] = k
                visited[i, j + 1] = True
                stack.append([i, j + 1])
            # check the down neighbor, second coordinate must >0
            if j > 0 and not visited[i, j - 1] and x[i, j - 1, 1] == 1:
                cluster[i, j - 1] = k
                visited[i, j - 1] = True
                stack.append([i, j - 1])

    return cluster


def get_largest_cluster(cluster: np.ndarray) -> tuple:
    """Return index and size of largest cluster"""
    counts = np.bincount(cluster.reshape(-1))
    largest_cluster = np.argmax(counts)
    largest_cluster_size = counts[largest_cluster]
    return largest_cluster, largest_cluster_size


def plot_figure(w, h, sample, cluster):
    """Plot clusters"""

    largest_cluster, largest_cluster_size = get_largest_cluster(cluster)

    # Compute space step
    dx = 1. / max(w, h)

    # Create figures
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'datalim')
    ax.set_title(f'{(w, h)} grid')
    ax.grid(True)

    for i in range(w + 1):
        for j in range(h + 1):
            # Plot horizontal edge
            color = 'r' if cluster[i, j] == largest_cluster else 'b'
            if i <= w - 1 and sample[i, j, 0] == 1:
                ax.plot([i * dx, (i + 1) * dx], [j * dx, j * dx], color)
            # Plot vertical edge
            if j <= h - 1 and sample[i, j, 1] == 1:
                ax.plot([i * dx, i * dx], [j * dx, (j + 1) * dx], color)
    # Add text
    plt.text(0.8, 1.,
             s=f'{largest_cluster_size}-vertex cluster',
             transform=ax.transAxes,
             bbox={'boxstyle': 'square', 'ec': 'r', 'fc': 'w'})
    plt.show()


def compute_clusters(w, h, p=0.5):
    """Compute clusters in a random (w, h)-grid"""
    sample = simu_perco_square(w, h, p)
    cluster = find_all_cluster(sample, w, h)
    return sample, cluster


def main(w, h, p):
    """Compute and plot clusters"""
    sample, cluster = compute_clusters(w, h, p)
    plot_figure(w, h, sample, cluster)


if __name__ == '__main__':
    main(22, 15, p=0.5)
