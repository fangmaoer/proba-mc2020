"""
Solve OT problem
"""

import numpy as np
import matplotlib.pyplot as plt
import ot


class EarthMovers2D:

    def __init__(self, n: int):
        self.n = n  # number of samples
        self.p = None
        self._set_positions()
        # Cost matrix
        self.M = None
        # OT matrix
        self.T = None
        # sample are uniform
        self.a = []
        self.b = []

    def _set_positions(self):
        # source position
        self.xs = np.random.random_sample((self.n, 2))
        # target position
        self.xt = np.random.random_sample((self.n, 2))

    def _compute_loss_matrix(self):
        """Return loss matrix"""
        self.M = (ot.dist(self.xs, self.xt,
                          metric='sqeuclidean'))**(self.p / 2)

    def get_ot_matrix(self):
        """Return optimal transport matrix"""
        self._compute_loss_matrix()
        return ot.emd(self.a, self.b, self.M)

    def get_wasserstein_distance(self) -> float:
        """Return Wasserstein_distance"""
        return np.sum(self.T * self.M)

    def compute_ot(self):
        """Solve OT problem"""
        self.T = self.get_ot_matrix()

    def get_distances(self):
        """Return a 1D-array of the distances"""
        return np.extract(self.T / self.T.max() > 1e-8, self.M)

    def create_figure(self, suptitle: str):
        """Create empty figure"""
        fig, ax = plt.subplots()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_aspect('equal', 'datalim')  # x and y scales are equal
        fig.suptitle(suptitle)
        return ax

    def plot_ot(self, p=1., plot_points=True):
        """A 2D plot of the OT problem"""
        self.p = p
        xs = self.xs
        xt = self.xt
        ax = self.create_figure(suptitle='Source and target distributions')

        self.compute_ot()

        max_distance = self.get_distances().max()
        # inspired by plot2D_samples_mat()
        mx = self.T.max()
        for i in range(self.n):
            for j in range(self.n):
                if self.T[i, j] / mx > 1e-8:
                    color_scale = 1 - self.M[i, j] / max_distance
                    c = [color_scale, color_scale, color_scale]
                    ax.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]], c=c)
        if plot_points:
            ax.plot(xs[:, 0], xs[:, 1], 'ob', label='Source samples')
            ax.plot(xt[:, 0], xt[:, 1], 'or', label='Target samples')
            ax.legend(loc=0)
        wd = self.get_wasserstein_distance()
        ax.set_title(f"$p = {self.p}$ - Wasserstein distance: {wd:f}",
                     fontsize=10)

    def plot_distance_histogram(self, p=1., bins=10):
        """Plot an histogram of distance"""
        self.p = p
        self.compute_ot()
        distances = self.get_distances()
        fig, ax = plt.subplots()
        plt.hist(distances, bins=bins)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Number of matchings")
        ax.set_title(f"Histogram of distance ($p = {self.p}$)")


class EarthMovers1D(EarthMovers2D):

    def _set_positions(self):
        # source and target positions
        self.xs = np.empty((self.n, 2))
        self.xt = np.empty((self.n, 2))
        # source
        self.xs[:, 0] = np.random.random_sample((self.n, ))
        self.xs[:, 1] = 0.
        # target
        self.xt[:, 0] = np.random.random_sample((self.n, ))
        self.xt[:, 1] = 1.

    def _compute_loss_matrix(self):
        """Return loss matrix"""
        self.M = (ot.dist(self.xs, self.xt, metric='sqeuclidean')
                  - 1)**(self.p / 2)


if __name__ == '__main__':
    em = EarthMovers2D(500)
    em.plot_ot(p=1., plot_points=False)

    em1D = EarthMovers1D(50)
    em1D.plot_ot(p=1.00001)

    em1000 = EarthMovers2D(1000)
    em1000.plot_distance_histogram(bins=20)

    plt.show()
