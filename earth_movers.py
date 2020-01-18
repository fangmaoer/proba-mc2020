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

    def compute_loss_matrix(self):
        """Return loss matrix"""
        # loss matrix
        M = ot.dist(self.xs, self.xt)
        M /= M.max()
        return M

    def create_figure(self, title):
        """Create empty figure"""
        fig, ax = plt.subplots()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_aspect('equal', 'datalim')  # x and y scales are equal
        ax.set_title(title)
        return ax

    def plot_ot(self):

        xs = self.xs
        xt = self.xt
        M = self.compute_loss_matrix()
        ax = self.create_figure(title='Source and target distributions')

        # using a uniform distribution on samples
        G = ot.emd(a=[], b=[], M=M)

        # inspired by plot2D_samples_mat()
        mx = G.max()
        for i in range(xs.shape[0]):
            for j in range(xt.shape[0]):
                if G[i, j] / mx > 1e-8:
                    ax.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                            alpha=G[i, j] / mx, c=[.5, .5, 1])

        ax.plot(xs[:, 0], xs[:, 1], 'ob', label='Source samples')
        ax.plot(xt[:, 0], xt[:, 1], 'or', label='Target samples')
        ax.legend(loc=0)


if __name__ == '__main__':
    em = EarthMovers(50)
    em.plot_ot()
    plt.show()
