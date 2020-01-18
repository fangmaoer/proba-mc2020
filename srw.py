"""Create a simple 2D random walk animation"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sqrt, pi
from numba import jit

# Fixing random state for reproducibility
# np.random.seed(19680801)

# four possible directions
directions = ((0, 1),   # North
              (1, 0),   # East
              (-1, 0),  # West
              (0, -1))  # South


def generate_walk(nstep: int):
    """
    Create a nstep-random walk path on Cartesian grid
    """
    x = np.empty(nstep)
    y = np.empty(nstep)

    # initial position
    x[0] = 0
    y[0] = 0

    # time loop
    for step in range(1, nstep):
        direction = directions[np.random.randint(4)]
        x[step] = x[step - 1] + direction[0]
        y[step] = y[step - 1] + direction[1]
    return x, y


def create_1Dfigure(title):
    """Return a 1D figure and plot area"""
    # Create figure and plot area
    fig, ax = plt.subplots()
    ax.set_xlabel('$n$')
    ax.set_ylabel('$d$')
    ax.set_title(title)
    return fig, ax


def create_2Dfigure(nstep: int):
    """Return a figure and plot area for a nstep-walk"""
    # Create figure and plot area
    fig, ax = plt.subplots()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'{nstep}-step random walk')
    ax.set_aspect('equal', 'datalim')
    ax.grid(True)
    return fig, ax


def generate_animation(nstep: int = 50):
    """return a nstep-random walk animation"""

    def update_path(step: int):
        """
        Update line and spot data with current step (step index)
        """
        # plot all edges from start to current position
        path.set_data(x[:step + 1], y[:step + 1])
        # plot only current position
        spot.set_data(x[step], y[step])

    x, y = generate_walk(nstep)

    # create figure and plot area
    fig, ax = create_2Dfigure(nstep)

    # plot a continuous line for walk path
    path, = ax.plot(x, y)
    # plot a round-symbol spot for current position
    spot, = ax.plot(x[0], y[0], 'o')

    anim = animation.FuncAnimation(fig,
                                   func=update_path,
                                   frames=nstep,
                                   interval=50,
                                   blit=False,
                                   repeat=False)

    return anim


def plot_walk(nstep: int):
    """Plot a nstep-random walk in a figure"""
    x, y = generate_walk(nstep)
    fig, ax = create_2Dfigure(nstep)

    path, = ax.plot(x, y)  # a line for path

    # symbols for initial (squared) and final (round) position
    ax.plot(x[0], y[0], 's', color=path.get_color())
    ax.plot(x[-1], y[-1], 'o')


class Distance:
    """Abstract class that should not be instantiated"""

    title = ''

    def __init__(self, nwalk):
        self.nwalk = nwalk

    def compute_average(self, nstep: int):
        """Compute average of func over nwalk repetitions of nstep-walks"""
        vfunc = np.vectorize(self.compute_distance)
        return np.sum(vfunc(np.full(self.nwalk, nstep))) / self.nwalk

    @staticmethod
    def compute_distance(nstep: int) -> float:
        pass

    def plot(self):
        """
        Plot mean distance from starting point as function of number of steps
        """
    
        def compute_distances():
            """
            return a mean over nwalk samples of the distance for various nsteps
            """
            dist = np.empty_like(nsteps, dtype=float)
            for i, nstep in np.ndenumerate(nsteps):
                dist[i] = self.compute_average(nstep)
            return dist

        nsteps = np.arange(1, 1000, 100)
        distances = compute_distances()

        fig, ax = create_1Dfigure(self.title)

        ax.plot(nsteps, np.sqrt(2 * nsteps / pi), label=r'$\sqrt{\frac{2n}{\pi}}$')
        ax.plot(nsteps, distances, 'o', label=f'Average over {self.nwalk} samples')
        ax.legend()


class FinalDistance(Distance):

    title = 'Distance as a function of number of steps $n$'

    @staticmethod
    def compute_distance(nstep: int):
        """Compute distance from start to end position for a 2D random walk"""

        def get_direction(i: int):
            """Return direction tuple from index"""
            return directions[i]

        vget_direction = np.vectorize(get_direction)

        # random array of direction indices
        dir_arr = np.random.randint(4, size=nstep)
        xsteps, ysteps = vget_direction(dir_arr)
        # Accumulate steps in each direction
        x = np.sum(xsteps)
        y = np.sum(ysteps)
    
        return sqrt(x**2 + y**2)


@jit(nopython=True)
def compute_max_distance(nstep: int):
    """Compute max reached distance for a 2D random walk"""

    # initial position
    x = 0
    y = 0
    max_distance = 0.

    # time loop
    for step in range(1, nstep):
        direction = directions[np.random.randint(4)]
        x += direction[0]
        y += direction[1]
        max_distance = max(max_distance, sqrt(x**2 + y**2))

    return max_distance


class MaxDistance(Distance):

    title = 'Maximum distance as a function of number of steps $n$'

    @staticmethod
    def compute_distance(nstep: int):
        return compute_max_distance(nstep)


def sample_cointossing(n: int, p: float) -> np.ndarray:
    """sample n random bernoulli numbers P(1)=p"""

    x = np.zeros(n)
    for i in range(n):
        x[i] = 2 * np.random.binomial(1, p, 1) - 1
    return x


def simu_rw_z(n, p, x0):
    """
    simulate random walk with n step and probability of +1 is p,
    start at x0,
    return the sample, number of visit to x0, max distance
    """

    z = sample_cointossing(n, p)
    x = np.empty(n+1)
    x[0] = x0
    x[1:] = x0 + np.cumsum(z)
    # count number of visit of x0
    i = 0
    for l in range(n):
        if x[l] == x0:
            i += 1
    # record the max distance
    j = 0
    for l in range(n):
        m = np.abs(x0 - x[l])
        if m > j:
            j = m
    return x, i, j


def head_tail_game(a, b, p):
    """
    head and tail game, return true if A wins, A bet on head,
    with probability p he wins 1 penny,
    A start with a pennies and B start with b pennies
    """
    x = b
    while x > 0 and x < a+b:
        x += 2*np.random.binomial(1, p, None) - 1
    return x == 0


def empirical_winrate_A(a, b, p, n):
    """
    empirical proba that A wins,
    theoretical value is (1-(p/(1-p))^b)/ (1-(p/(1-p))^(a+b)) if p != 1/2
    and a/(a+b) if p=1/2
    """
    win = 0
    for i in range(n):
        if head_tail_game(a, b, p):
            win += 1
    return win/n

# does not seems to work very well..
#print(empirical_winrate_A(3, 5, 1/3, 10000))


if __name__ == '__main__':
    anim = generate_animation(100)

    FinalDistance(1000).plot()
    MaxDistance(1000).plot()
    plt.show()