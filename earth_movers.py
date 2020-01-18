"""
Exemple taken from
https://pot.readthedocs.io/en/stable/auto_examples/plot_OT_2D_samples.html
"""

import numpy as np
import matplotlib.pyplot as plt
import ot
import ot.plot


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

    def plot_ot(self):

        M = self.compute_loss_matrix()

        # using a uniform distribution on samples
        G0 = ot.emd(a=[], b=[], M=M)
        ot.plot.plot2D_samples_mat(self.xs, self.xt, G0, c=[.5, .5, 1])

        plt.figure(1)
        plt.plot(self.xs[:, 0], self.xs[:, 1], 'ob', label='Source samples')
        plt.plot(self.xt[:, 0], self.xt[:, 1], 'or', label='Target samples')
        plt.legend(loc=0)
        plt.title('Source and target distributions')
        plt.show()


if __name__ == '__main__':
    em = EarthMovers(50)
    em.plot_ot()