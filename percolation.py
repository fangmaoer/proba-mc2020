"""simulate Percolations on different graphs"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Patch
from matplotlib.lines import Line2D
from math import sqrt, pi
import progressbar

# Fixing random state for reproducibility
# np.random.seed(19680801)

# Standard coupling of edge percolation in a w by h rectangle of square lattice
# width w
# height h
# total number of edges is h*(w+1) vertical ones and (h+1)*w horizontal
# ones, edges are labeled with vertex, the vertex (i,j) is associate to
# two edges, the right one (i,j)--(i+1,j) is recorded in the element
# x[i,j,0],  the up one (i,j)--(i,j+1) is recorded in x[i,j,1].


class Percolation:

    grid_type = ''
    largest_cluster_color = ''
    other_clusters_color = ''

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.p = None
        self.rand_array = self._get_rand_array()
        self.sample = None
        self.cluster = None
        self.suptitle = f'Percolation on a {self.grid_type} grid'

    def _get_rand_array(self):
        """Return a 2d random array"""
        pass

    def _get_sample(self, p: float) -> np.ndarray:
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
        return np.where(self.rand_array > p, 0, 1)

    def compute_clusters(self, p: float):
        """
        Compute the clusters of a percolation sample x of width w
        and height h
        """
        pass

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
        fig.suptitle(self.suptitle)
        ax.grid(show_grid)
        return ax

    def set_title(self, ax):
        """Set ax title"""
        ax.set_title(f'{self.w} x {self.h}-grid, $p = {self.p}$', fontsize=10)

    def set_legend(self, ax, largest_cluster_size):
        """Set figure legend"""
        pass

    def plot_clusters(self):
        """Plot clusters using matplotlib"""
        pass

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

    def find_p_cross(self):
        """
        Find minimum p value that produces a crossing
        Use a simple dichotomy algorithm
        """

        # initial values
        a = 0.
        b = 1.
        err = 1.

        while err > 1e-3:
            p = 0.5 * (a + b)
            self.compute_clusters(p)
            if self.is_crossed():
                b = p
            else:
                a = p
            err = abs(a - b)

        return p

    def plot(self, p: int):
        """Compute percolation for a given p and plot it"""
        self.compute_clusters(p)
        self.plot_clusters()


class PercolationRect(Percolation):

    grid_type = 'rectangular'
    largest_cluster_color = 'r'
    other_clusters_color = 'b'

    def _get_rand_array(self):
        """Return a 2d random array"""
        return np.random.random((self.w + 1, self.h + 1, 2))

    def compute_clusters(self, p: float):
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
        self.p = p
        self.sample = self._get_sample(p)
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

    def set_legend(self, ax, largest_cluster_size):
        """Set figure legend"""
        handles = []
        handles.append(Line2D([0], [0], color=self.largest_cluster_color,
                              label='largest cluster '
                                    f'({largest_cluster_size} cells)'))
        handles.append(Line2D([0], [0], color=self.other_clusters_color,
                              label='other clusters'))
        ax.legend(handles=handles)

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
                if self.cluster[i, j] == largest_cluster:
                    color = self.largest_cluster_color
                else:
                    color = self.other_clusters_color
                # Plot horizontal edge
                if i <= w - 1 and self.sample[i, j, 0] == 1:
                    ax.plot([i * dx, (i + 1) * dx], [j * dx, j * dx],
                            color=color)
                # Plot vertical edge
                if j <= h - 1 and self.sample[i, j, 1] == 1:
                    ax.plot([i * dx, i * dx], [j * dx, (j + 1) * dx],
                            color=color)

        self.set_title(ax)
        self.set_legend(ax, largest_cluster_size)


class PercolationHex(Percolation):
    """
    The hexagonal lattice, percolation by cells, we take a w times h hexagonal
    lattice, label the hexagonal cells by its center, counts from left bottom,
    Cell centers are labeled (0,0) ... (h-1,w-1)
    """

    grid_type = 'hexagonal'
    largest_cluster_color = 'lightcoral'
    other_clusters_color = 'sandybrown'

    def _get_rand_array(self):
        """Return a 2d random array"""
        return np.random.random((self.w, self.h))

    def compute_clusters(self, p: float):
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
        self.p = p
        self.sample = self._get_sample(p)
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

    def set_legend(self, ax, largest_cluster_size):
        """Set figure legend"""
        handles = []
        handles.append(Patch(color=self.largest_cluster_color,
                             label='largest cluster '
                                   f'({largest_cluster_size} cells)'))
        handles.append(Patch(color=self.other_clusters_color,
                             label='other clusters'))
        ax.legend(handles=handles)

    def plot_clusters(self, add_cluster_id=False):
        """Plot clusters using matplotlib"""
        w = self.w
        h = self.h
        largest_cluster, largest_cluster_size = self.get_largest_cluster()
        ax = self.create_figure(show_grid=False)

        for i in range(w):
            for j in range(h):
                if self.sample[i, j] == 0:
                    color = 'white'
                else:
                    if self.cluster[i, j] == largest_cluster:
                        color = self.largest_cluster_color
                    else:
                        color = self.other_clusters_color
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

        self.set_title(ax)
        self.set_legend(ax, largest_cluster_size)
        ax.autoscale()


class PercolationRectDual(PercolationRect):
    """A class to plot the percolation graph and its dual"""

    grid_type = 'rectangular'

    def __init__(self, n: int):
        self.n = n
        self.p = None
        self.rand_array = self._get_rand_array()
        self.suptitle = f'Percolation on a {self.grid_type} grid'
        self.ax = None
        self.handles = []

    def _get_rand_array(self):
        """Return a 2d random array"""
        rand_array = np.random.random((self.n, self.n, 2))
        rand_array[0, :, 1] = 0.
        rand_array[-1, :, 1] = 0.
        return rand_array

    def plot_graph(self, p, graph_type='initial'):
        """Plot graph using matplotlib"""
        n = self.n
        self.handles = []  # Reinitialize legend handles
        self.sample = self._get_sample(p)
        self.ax = self.create_figure()
        self.ax.set_title(f'{self.n} x {self.n - 1}-grid, $p = {p}$',
                          fontsize=10)

        if graph_type == 'initial':
            InitialGraph(self).plot()
        elif graph_type == 'dual':
            DualGraph(self).plot()
        elif graph_type == 'both':
            InitialGraph(self).plot()
            DualGraph(self).plot()

        self.ax.legend(handles=self.handles)


class Graph:
    """An abstract class to plot a graph"""

    color = ''
    legend = ''

    def __init__(self, percolation):
        self.n = percolation.n
        self.ax = percolation.ax
        self.sample = percolation.sample
        self.handles = percolation.handles
        self.dx = 1. / self.n

    def plot(self):
        """plot all edges"""
        for i in range(self.n):
            for j in range(self.n):
                self.plot_ij(i, j)
        self.set_legend()

    def plot_ij(self, i: int, j: int):
        pass

    def plot_edge(self, x0, x1, y0, y1):
        """plot a single edge"""
        self.ax.plot([x0, x1], [y0, y1], '-o', color=self.color)

    def set_legend(self):
        """Set figure legend"""
        self.handles.append(Line2D([0], [0], color=self.color,
                            label=self.legend))


class InitialGraph(Graph):
    """A class to plot an initial graph"""

    color = 'b'
    legend = 'Initial graph'

    def plot_ij(self, i: int, j: int):
        n = self.n
        dx = self.dx
        # Plot horizontal edge
        if self.sample[i, j, 0] == 1:
            x0, x1 = i * dx, (i + 1) * dx
            y0, y1 = j * dx, j * dx
            self.plot_edge(x0, x1, y0, y1)
        # Plot vertical edge
        if i > 0 and j < n - 1 and self.sample[i, j, 1] == 1:
            x0, x1 = i * dx, i * dx
            y0, y1 = j * dx, (j + 1) * dx
            self.plot_edge(x0, x1, y0, y1)


class DualGraph(Graph):
    """A class to plot a dual graph"""

    color = 'r'
    legend = 'Dual graph'

    def plot_ij(self, i: int, j: int):
        n = self.n
        dx = self.dx
        # Plot vertical edge
        if self.sample[i, j, 0] == 0:
            x0, x1 = (i + 0.5) * dx, (i + 0.5) * dx
            y0, y1 = (j - 0.5) * dx, (j + 0.5) * dx
            self.plot_edge(x0, x1, y0, y1)
        # Plot horizontal edge
        if i > 0 and j < n - 1 and self.sample[i, j, 1] == 0:
            x0, x1 = (i - 0.5) * dx, (i + 0.5) * dx
            y0, y1 = (j + 0.5) * dx, (j + 0.5) * dx
            self.plot_edge(x0, x1, y0, y1)


def percolation_vs_p(w: int, h: int, nsim=40, n_p=50):
    """
    Plot the probability of crossing as a function of p
    by running nsim simulations
    """
    p_values = np.linspace(0., 1., n_p)  # n_p-value array between 0 and 1

    def plot_crossing_probability(ax, Percolation) -> np.ndarray:
        """
        Plot crossing probabilities of a percolation of type Percolation
        """

        print(f"Computing crossing probabilities for {Percolation.grid_type} "
              "percolation")
        cross_proba = np.zeros_like(p_values)
        for i in progressbar.progressbar(range(nsim)):
            perco = Percolation(w, h)
            p_cross = perco.find_p_cross()
            cross_proba += np.where(p_values < p_cross, 0, 1)

        cross_proba /= nsim
        ax.plot(p_values, cross_proba, '-',
                label=f'{Percolation.grid_type} percolation')

    fig, ax = plt.subplots()
    fig.suptitle('Probability of crossing as a function of $p$')
    ax.set_xlabel('$p$')
    ax.set_ylabel('probability')
    ax.grid()
    plot_crossing_probability(ax, PercolationRect)
    plot_crossing_probability(ax, PercolationHex)
    ax.legend()
    ax.set_title(f"{nsim} simulations on a {w} x {h} grid", fontsize=10)


if __name__ == '__main__':

    percorect = PercolationRect(20, 10)
    percorect.compute_clusters(p=0.5)
    # print(percorect)
    percorect.plot_clusters()

    percohex = PercolationHex(20, 20)
    percohex.compute_clusters(p=0.5)
    # print(percohex)
    # print('is_crossed:', percohex.is_crossed())
    percohex.plot_clusters(add_cluster_id=False)

    # Compute percolation probabilities
    percolation_vs_p(20, 20, nsim=200, n_p=100)

    percorectdual = PercolationRectDual(5)
    percorectdual.plot_graph(p=0.5, graph_type='both')

    plt.show()
