"""
Exemple taken from
https://pot.readthedocs.io/en/stable/auto_examples/plot_OT_2D_samples.html
"""

import numpy as np
import matplotlib.pyplot as plt
import ot
import ot.plot

n = 50  # nb samples

xs = np.random.random_sample((n, n))
xt = np.random.random_sample((n, n))

# loss matrix
M = ot.dist(xs, xt)
M /= M.max()

# uniform distribution on samples
a = np.ones((n,)) / n
b = np.ones((n,)) / n
G0 = ot.emd(a, b, M)
ot.plot.plot2D_samples_mat(xs, xt, G0, c=[.5, .5, 1])

plt.figure(1)
plt.plot(xs[:, 0], xs[:, 1], 'ob', label='Source samples')
plt.plot(xt[:, 0], xt[:, 1], 'or', label='Target samples')
plt.legend(loc=0)
plt.title('Source and target distributions')
plt.show()
