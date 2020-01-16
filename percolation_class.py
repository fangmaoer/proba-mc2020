"""simulate Percolations on different graphs"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
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


class PercolationRect:

    grid_type = 'rectangular'

    def __init__(self, w: int, h: int, p: float):
        self.w = w
        self.h = h
        self.p = p
        self.sample = self._simu_perco()
        self.cluster = None
        self.title = f'Percolation on a {self.grid_type} {(w, h)}-grid'

    def _simu_perco(self) -> np.ndarray:
        """
        This function will give a percolation sample in a rectangle
        of height h, width w, parameter p,
        the output is two arrays:
        - sample[h+1, w, 0], the horizontal edges,
        - sample[h, w+1, 1], the vertical edges.
        Order the horizontal edges by (i,j,0); i=1..w; j= 1..h+1;
        from the bottom left corner,
        i coordinate go to the right and j coordinate go to the top.
        Order the vertical edges by (i,j,1); i=1..w+1; j=1..h,
        same coordinate as the horizontals.
        """
        return np.where(np.random.random((self.w + 1, self.h + 1, 2)) < self.p,
                        0, 1)

    def compute_clusters(self):
        """
        Compute the clusters of a percolation sample x of width w
        and height h, on the square lattice,
        return a list cluster[i, j] = k where
        k indicate to which cluster the site (i,j) belongs to.

        Order the vertices by order[i, j] = i + 1 + (w + 1)*j,
        that is left to right, bottom to top.
        The variable order record the next unvisited vertex
        """
        w = self.w
        h = self.h
        self.cluster = np.zeros((w + 1, h + 1), dtype=int)
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
                # pop the current myvertex from the stack and set its cluster
                # label to k and mark as visited
                i, j = stack.pop(0)
                self.cluster[i, j] = k
                visited[i, j] = True
                # check all of its 4 neighbors, if neighbor is unvisited and
                # connected to current site,
                # then set its cluster label to k and marked visited and
                # push this site into stack, otherwise do nothing
                # check the left neighbor, first coordinate must >0 to have
                # a left neighbor
                if i > 0 and not visited[i - 1, j] and \
                   self.sample[i - 1, j, 0] == 1:
                    self.cluster[i - 1, j] = k
                    visited[i - 1, j] = True
                    stack.append([i - 1, j])
                # check the right neighbor, first coordinate must be < w
                # to have a right neighbor
                if i < w and not visited[i + 1, j] and \
                   self.sample[i, j, 0] == 1:
                    self.cluster[i + 1, j] = k
                    visited[i + 1, j] = True
                    stack.append([i + 1, j])
                # check the up neighbor, second coordinate must be < h
                # to have such a neighbor
                if j < h and not visited[i, j + 1] and \
                   self.sample[i, j, 1] == 1:
                    self.cluster[i, j + 1] = k
                    visited[i, j + 1] = True
                    stack.append([i, j + 1])
                # check the bottom neighbor, second coordinate must be > 0
                if j > 0 and not visited[i, j - 1] and \
                   self.sample[i, j - 1, 1] == 1:
                    self.cluster[i, j - 1] = k
                    visited[i, j - 1] = True
                    stack.append([i, j - 1])

    def get_largest_cluster(self) -> tuple:
        """Return index and size of largest cluster"""
        flat_cluster = self.cluster.reshape(-1)
        true_clusters = np.extract(flat_cluster > 0, flat_cluster)
        counts = np.bincount(true_clusters)
        largest = np.argmax(counts)
        size = counts[largest]
        return largest, size

    def create_figure(self, show_grid=True):
        """Create empty figure"""
        fig, ax = plt.subplots()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_aspect('equal', 'datalim')  # x and y scales are equal
        fig.suptitle(self.title)
        ax.grid(show_grid)
        return ax

    @staticmethod
    def add_text(ax, text, color='r'):
        """Add text to figure"""
        plt.text(0.8, 1.,
                 s=text,
                 transform=ax.transAxes,
                 bbox={'boxstyle': 'square', 'ec': color, 'fc': 'w'})

    def plot_clusters(self):
        """Plot clusters using matplotlib"""
        w = self.w
        h = self.h
        largest_cluster, largest_cluster_size = self.get_largest_cluster()

        # Compute space step
        dx = 1. / max(w, h)

        # Create figure
        ax = self.create_figure()

        for i in range(w + 1):
            for j in range(h + 1):
                color = 'r' if self.cluster[i, j] == largest_cluster else 'b'
                # Plot horizontal edge
                if i <= w - 1 and self.sample[i, j, 0] == 1:
                    ax.plot([i * dx, (i + 1) * dx], [j * dx, j * dx], color)
                # Plot vertical edge
                if j <= h - 1 and self.sample[i, j, 1] == 1:
                    ax.plot([i * dx, i * dx], [j * dx, (j + 1) * dx], color)
        self.add_text(ax, f'{largest_cluster_size}-vertex cluster')

    def __repr__(self):
        """Return a string be output with print(self)"""
        s = f'sample:\n{self.sample}\n'
        s += f'cluster:\n{self.cluster}\n'
        s += f'largest_cluster:\n{self.get_largest_cluster()}'
        return s

    def is_crossed(self):
        """Return True if percolation crosses from left to right boundary"""
        left_boundary_clusters = np.extract(self.cluster[0] > 0,
                                            self.cluster[0])
        right_boundary_clusters = np.extract(self.cluster[-1] > 0,
                                             self.cluster[-1])
        return np.in1d(left_boundary_clusters, right_boundary_clusters).any()


class PercolationHex(PercolationRect):
    """
    The hexagonal lattice, percolation by cells, we take a w times h hexagonal
    lattice, label the hexagonal cells by its center, counts from left bottom,
    Cell centers are labeled (0,0) ... (h-1,w-1)
    """

    grid_type = 'hexagonal'

    def _simu_perco(self) -> np.ndarray:
        """
        This function will give a percolation sample in a hexagonal rectangle
        of height h, width w, parameter p, the output is a array X[(w,h)]
        where X[i,j] is the sample uniform number at the cell (i,j).
        """
        return np.where(np.random.random((self.w, self.h)) < self.p, 0, 1)

    def compute_clusters(self):
        """
        compute the clusters of a percolation sample x of width w
        and height h, on the hexagonal lattice,
        return a list cluster[i,j] = k where
        k indicates to which cluster the site (i,j) belongs to.
        order the vertices by order(i,j)=i+1+wj, that is left to right,
        bottom to top. the variable myvertex record the next unvisited vertex
        """
        w = self.w
        h = self.h
        self.cluster = np.zeros((w, h), dtype=int)
        x = self.sample
        visited = np.full((w, h), False)
        k = 0  # cluster index
        myvertex = 1
        stack = []
        # as long as we havent treated the last myvertex, continue
        while myvertex < w * h + 1:
            # put the next site in myvertex in to the stack if the site is
            # unvisited, otherwise myvertex ++
            iv = (myvertex - 1) % w
            jv = (myvertex - 1) // w
            if not visited[iv, jv] and x[iv, jv] == 1:
                stack.append([iv, jv])
                k += 1  # increment cluster index
            else:
                myvertex += 1

            while stack:
                # pop the current myvertex from the stack and set its cluster
                # label to k and mark as visited
                i, j = stack.pop(0)
                self.cluster[i, j] = k
                visited[i, j] = True
                # check all of its six neighbors, if neighbor is unvisited and
                # connected to current site,
                # then set its cluster label to k and marked visited and
                # push this site into stack, otherwise do nothing
                # check the 12clock neighbor
                if j < h-1 and not visited[i, j+1] and x[i, j+1] == 1:
                    self.cluster[i, j+1] = k
                    visited[i, j+1] = True
                    stack.append([i, j+1])
                # check the 2clock neighbor
                if i < w-1 and not visited[i+1, j] and x[i+1, j] == 1:
                    self.cluster[i+1, j] = k
                    visited[i+1, j] = True
                    stack.append([i+1, j])
                # check the 4clock neighbor
                if i < w-1 and j > 0 and not visited[i+1, j-1] \
                   and x[i+1, j-1] == 1:
                    self.cluster[i+1, j-1] = k
                    visited[i+1, j-1] = True
                    stack.append([i+1, j-1])
                # check the 6clock neighbor
                if j > 0 and not visited[i, j-1] and x[i, j-1] == 1:
                    self.cluster[i, j-1] = k
                    visited[i, j-1] = True
                    stack.append([i, j-1])
                # check the 8clock neighbor
                if i > 0 and not visited[i-1, j] and x[i-1, j] == 1:
                    self.cluster[i-1, j] = k
                    visited[i-1, j] = True
                    stack.append([i-1, j])
                # check the 10clock neighbor
                if i > 0 and j < h-1 and not visited[i-1, j+1] \
                   and x[i-1, j+1] == 1:
                    self.cluster[i-1, j+1] = k
                    visited[i-1, j+1] = True
                    stack.append([i-1, j+1])

    def plot_clusters(self, add_cluster_id=False):
        """Plot clusters using matplotlib"""
        w = self.w
        h = self.h
        largest_cluster, largest_cluster_size = self.get_largest_cluster()
        ax = self.create_figure(show_grid=False)

        for i in range(w):
            for j in range(h):
                if self.sample[i, j] == 0:
                    color = 'lightblue'
                else:
                    if self.cluster[i, j] == largest_cluster:
                        color = 'lightcoral'
                    else:
                        color = 'sandybrown'
                # Polygon coordinates
                x = (i * 3**0.5 / 2)
                y = (i / 2 + j)
                hex = RegularPolygon((x, y), numVertices=6, radius=3**0.5/3,
                                     orientation=np.radians(30),
                                     facecolor=color, alpha=1.,
                                     edgecolor='grey')
                ax.add_patch(hex)
                if add_cluster_id:
                    ax.text(x, y, self.cluster[i, j], ha='center', va='center')

        self.add_text(ax, f'{largest_cluster_size}-cell cluster',
                      color='lightcoral')
        ax.autoscale()


def percolation_vs_p(w: int, h: int, nsim=40, n_p=40):
    """
    Plot the probability of crossing as a function of p
    by running nsim simulations
    """
    p = np.linspace(0., 1., n_p)  # 40-value array between 0 and 1

    def crossing_probability(Percolation, p: float):
        """Return a probability of crossing"""
        sum = 0
        for i in range(nsim):
            perco = Percolation(w, h, p)
            perco.compute_clusters()
            if perco.is_crossed():
                sum += 1
        return sum / nsim

    def get_crossing_probabilities(Percolation):
        """
        Return a n_p array of crossing probability for given Percolation type
        """
        print(f"Computing crossing probabilities for {Percolation.grid_type} "
              "percolation")
        crossing_prob = np.zeros_like(p)
        for i in range(len(p)):
            crossing_prob[i] = crossing_probability(Percolation, p[i])
        return crossing_prob

    crossing_prob_rect = get_crossing_probabilities(PercolationRect)
    crossing_prob_hex = get_crossing_probabilities(PercolationHex)

    fig, ax = plt.subplots()
    fig.suptitle('Probability of crossing as a function of $p$')
    ax.set_xlabel('$p$')
    ax.set_ylabel('probability')
    ax.grid()
    plt.plot(p, crossing_prob_rect, '-o', label='Rectangular percolation')
    plt.plot(p, crossing_prob_hex, '-x', label='Hexagonal percolation')
    ax.legend()
    ax.set_title(f"{nsim} simulations on a {w} x {h} grid", fontsize=10)


if __name__ == '__main__':

    percorect = PercolationRect(20, 10, 0.5)
    percorect.compute_clusters()
    # print(percorect)
    percorect.plot_clusters()

    percohex = PercolationHex(20, 20, 0.5)
    percohex.compute_clusters()
    # print(percohex)
    # print('is_crossed:', percohex.is_crossed())
    percohex.plot_clusters(add_cluster_id=False)

    # Compute percolation probabilities
    percolation_vs_p(20, 20, nsim=50)
    plt.show()
