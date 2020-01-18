"""
Exemple taken from
https://pot.readthedocs.io/en/stable/auto_examples/plot_OT_2D_samples.html
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
        self.M = self._compute_loss_matrix()
        # sample are uniform
        self.a = []
        self.b = []

    def _compute_loss_matrix(self):
        """Return loss matrix"""
        M = ot.dist(self.xs, self.xt)
        M /= M.max()
        return M

    def get_ot_matrix(self):
        """Return optimal transport matrix"""
        return ot.emd(self.a, self.b, self.M)

    def get_wasserstein_distance(self) -> float:
        """Return Wasserstein_distance"""
        return ot.emd2(self.a, self.b, self.M)

    #def get_distances(self):


    def create_figure(self, suptitle: str):
        """Create empty figure"""
        fig, ax = plt.subplots()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_aspect('equal', 'datalim')  # x and y scales are equal
        fig.suptitle(suptitle)
        return ax

    def plot_ot(self):

        xs = self.xs
        xt = self.xt
        ax = self.create_figure(suptitle='Source and target distributions')

        # using a uniform distribution on samples
        G = self.get_ot_matrix()

        # inspired by plot2D_samples_mat()
        mx = G.max()
        for i in range(self.n):
            for j in range(self.n):
                if G[i, j] / mx > 1e-8:
                    ax.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                            alpha=G[i, j] / mx, c=[.5, .5, 1])

        ax.plot(xs[:, 0], xs[:, 1], 'ob', label='Source samples')
        ax.plot(xt[:, 0], xt[:, 1], 'or', label='Target samples')
        ax.legend(loc=0)
        wd = self.get_wasserstein_distance()
        ax.set_title(f"Wasserstein distance: {wd:f}", fontsize=10)


if __name__ == '__main__':
    em = EarthMovers(10)
    em.plot_ot()
    plt.show()

    em500 = EarthMovers(500)
    print(f"Wasserstein distance: {em500.get_wasserstein_distance():f}")
