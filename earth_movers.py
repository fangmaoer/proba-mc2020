"""
Solve OT problem
"""

import numpy as np
import matplotlib.pyplot as plt
import ot


class EarthMovers:

    def __init__(self, n):
        self.n = n  # number of samples
        # source position
        self.xs = np.random.random_sample((self.n, self.n))
        # target position
        self.xt = np.random.random_sample((self.n, self.n))
        # Cost matrix
        self.M = None
        # OT matrix
        self.T = None
        # sample are uniform
        self.a = []
        self.b = []

    def _compute_loss_matrix(self):
        """Return loss matrix"""
        M = ot.dist(self.xs, self.xt)
        M /= M.max()
        self.M = M

    def get_ot_matrix(self):
        """Return optimal transport matrix"""
        if self.M is None:
            self._compute_loss_matrix()
        return ot.emd(self.a, self.b, self.M)

    def get_wasserstein_distance(self) -> float:
        """Return Wasserstein_distance"""
        if self.M is None:
            self._compute_loss_matrix()
        if self.T is None:
            self.compute_ot()
        return np.sum(self.T * self.M)

    def compute_ot(self):
        """Solve OT problem"""
        self.T = self.get_ot_matrix()

    def get_distances(self):
        """Return a 1D-array of the distances"""
        if self.T is None:
            self.compute_ot()
        d = np.zeros((self.n, ))
        mx = self.T.max()
        k = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.T[i, j] / mx > 1e-8:
                    d[k] = self.M[i, j]
                    k += 1
        return d

    def create_figure(self, suptitle: str):
        """Create empty figure"""
        fig, ax = plt.subplots()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_aspect('equal', 'datalim')  # x and y scales are equal
        fig.suptitle(suptitle)
        return ax

    def plot_ot(self):
        """A 2D plot of the OT problem"""
        xs = self.xs
        xt = self.xt
        ax = self.create_figure(suptitle='Source and target distributions')

        if self.T is None:
            self.compute_ot()

        # inspired by plot2D_samples_mat()
        mx = self.T.max()
        for i in range(self.n):
            for j in range(self.n):
                if self.T[i, j] / mx > 1e-8:
                    ax.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                            alpha=self.T[i, j] / mx, c=[.5, .5, 1])

        ax.plot(xs[:, 0], xs[:, 1], 'ob', label='Source samples')
        ax.plot(xt[:, 0], xt[:, 1], 'or', label='Target samples')
        ax.legend(loc=0)
        wd = self.get_wasserstein_distance()
        ax.set_title(f"Wasserstein distance: {wd:f}", fontsize=10)

    def plot_distance_histogram(self, bins=10):
        """Plot an histogram of distance"""
        distances = self.get_distances()
        fig, ax = plt.subplots()
        plt.hist(distances, bins=bins)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Number of matchings")
        ax.set_title("Histogram of distance")


if __name__ == '__main__':
    em = EarthMovers(50)
    em.plot_ot()

    em1000 = EarthMovers(2000)
    print(f"Wasserstein distance: {em1000.get_wasserstein_distance():f}")
    em1000.plot_distance_histogram(bins=20)

    plt.show()
