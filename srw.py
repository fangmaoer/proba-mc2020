"""Create a simple 2D random walk animation"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from math import sqrt, pi
from numba import jit

# Fixing random state for reproducibility
# np.random.seed(19680801)

# four possible directions
directions = ((0, 1),   # North
              (1, 0),   # East
              (-1, 0),  # West
              (0, -1))  # South


class Walk2D:

    def __init__(self, nstep: int):
        self.nstep = nstep
        self.x, self.y = self._compute_walk(self.nstep)

    @staticmethod
    @jit(nopython=True)
    def _compute_walk(nstep: int):
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

    def _init_figure(self):
        """Return a figure and plot area"""
        # Create figure and plot area
        fig, ax = plt.subplots()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title(f'{self.nstep}-step random walk')
        ax.set_aspect('equal', 'datalim')
        ax.grid(True)
        return fig, ax

    def generate_animation(self):
        """return a nstep-random walk animation"""

        x = self.x
        y = self.y

        def update_path(step: int):
            """
            Update line and spot data with current step (step index)
            """
            # plot all edges from start to current position
            path.set_data(x[:step + 1], y[:step + 1])
            # plot only current position
            spot.set_data(x[step], y[step])

        # create figure and plot area
        fig, ax = self._init_figure()

        # plot a continuous line for walk path
        path, = ax.plot(x, y)
        # plot a square-symbol for initial position
        ax.plot(x[0], y[0], 's', color=path.get_color())
        # plot a round-symbol spot for current position
        spot, = ax.plot(x[0], y[0], 'o')

        anim = animation.FuncAnimation(fig,
                                       func=update_path,
                                       frames=self.nstep,
                                       interval=50,
                                       blit=False,
                                       repeat=False)

        return anim

    def plot(self, colorize=True):
        """Plot walk in a figure"""
        x = self.x
        y = self.y
        fig, ax = self._init_figure()

        if colorize:
            t = np.arange(1, x.shape[0] + 1)  # time variable

            # set up an array of (x,y) points
            points = np.array([x, y]).transpose().reshape(-1, 1, 2)

            # set up a list of segments
            segs = np.concatenate([points[:-1], points[1:]], axis=1)

            # make the collection of segments
            cmap = plt.get_cmap('jet')
            lc = LineCollection(segs, cmap=cmap, alpha=0.5)
            lc.set_array(t)  # color the segments by the time parameter

            # plot the collection
            ax.add_collection(lc)
            # line collections donnot auto-scale the plot
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())

            cb = fig.colorbar(lc)
            cb.ax.set_title(r"$n_{step}$")
            start_color = cmap(t[0])
            end_color = cmap(t[-2] * 256 / t.shape[0])
        else:
            path, = ax.plot(x, y)  # a line for path
            start_color = 'C1'  # blue
            end_color = 'C2'  # orange

        # symbols for initial (squared) and final (round) position
        ax.plot(x[0], y[0], 's', ms=10,
                markerfacecolor='white',
                markeredgecolor=start_color,
                markeredgewidth=3)
        ax.plot(x[-1], y[-1], 'o', ms=10,
                markerfacecolor='white',
                markeredgecolor=end_color,
                markeredgewidth=3)


class NWalk:
    """
    Abstract class for a n-walk computation
    (should not be instantiated)
    """

    suptitle = ''
    title = 'as a function of the number of steps $n$'

    def __init__(self, nwalk: int, nstepmax=1000, step_num=10):
        """
        nwalk: number of random walks for averaging
        nstepmax: maximum final step
        num_step: number of nsteps to compute and plot
        """
        self.nwalk = nwalk
        self.nsteps = np.linspace(1, nstepmax, num=step_num, endpoint=True,
                                  dtype=int)
        # abscissa values for analytical formula
        self.x_ana = np.linspace(1, nstepmax, num=100, endpoint=True,
                                 dtype=int)

    @staticmethod
    def compute_step(nstep: int):
        pass

    def compute_average(self, nstep: int):
        """Compute average of func over nwalk repetitions of nstep-walks"""
        vfunc = np.vectorize(self.compute_step)
        return np.sum(vfunc(np.full(self.nwalk, nstep))) / self.nwalk

    def compute_steps(self) -> np.ndarray:
        """
        return a mean over nwalk samples of the distance for various nsteps
        """
        result = np.empty_like(self.nsteps, dtype=float)
        for i, nstep in np.ndenumerate(self.nsteps):
            result[i] = self.compute_average(nstep)
        return result

    def _init_figure(self, ylabel='$d$'):
        """Return a 1D figure and plot area"""
        # Create figure and plot area
        fig, ax = plt.subplots()
        ax.set_xlabel('$n$')
        ax.set_ylabel(ylabel)
        if self.suptitle:
            fig.suptitle(self.suptitle)
            ax.set_title(self.title, fontsize=10)
        else:
            ax.set_title(self.title)
        return fig, ax

    def plot(self):
        pass


class Distance(NWalk):
    """
    Abstract class for computin distance as function of nstep
    (should not be instantiated)
    """

    analytical_function = r''

    def get_analytical(self):
        pass

    def plot(self):
        """
        Plot mean distance from starting point as function of number of steps
        """

        distances = self.compute_steps()

        fig, ax = self._init_figure()

        ax.plot(self.nsteps, distances, 'o',
                label=f'Average over {self.nwalk} samples')
        ax.plot(self.x_ana, self.get_analytical(),
                label=self.analytical_function)
        ax.legend()


class FinalDistance(Distance):
    """A class to plot final distance as function of nstep"""

    suptitle = 'Distance to starting point'
    analytical_function = r'$\sqrt{\frac{2n}{\pi}}$'

    @staticmethod
    def compute_step(nstep: int) -> np.ndarray:
        """Compute distance from start to end position for a 2D random walk"""

        def get_direction(i: int) -> tuple:
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

    def get_analytical(self):
        return np.sqrt(2 * self.x_ana / pi)


class MaxDistance(Distance):
    """A class to plot maximum distance as function of nstep"""

    suptitle = 'Maximum distance'
    analytical_function = r'$\sqrt{n}$'

    @staticmethod
    @jit(nopython=True)
    def compute_step(nstep: int):
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

    def get_analytical(self):
        return np.sqrt(self.x_ana)


class BackToStart(NWalk):
    """
    A class to plot the number times when the walk go back to starting point
    """

    suptitle = 'Number of return to starting point'

    def __init__(self, nwalk: int, nstepmax=1000, step_num=10):
        """
        nwalk: number of random walks for averaging
        nstepmax: maximum final step
        num_step: number of nsteps to compute and plot
        """
        self.nwalk = nwalk
        self.nsteps = np.logspace(2, np.log10(nstepmax), num=step_num,
                                  endpoint=True, dtype=int)

    @staticmethod
    @jit(nopython=True)
    def compute_step(nstep: int):
        """Compute the number of return to start for a 2D random walk"""

        x = 0
        y = 0
        ntimes = 0.

        # time loop
        for step in range(1, nstep):
            direction = directions[np.random.randint(4)]
            x += direction[0]
            y += direction[1]
            if x == 0 and y == 0:
                ntimes += 1

        return ntimes

    def plot(self):
        """
        Plot mean distance from starting point as function of number of steps
        """

        ntimes = self.compute_steps()

        fig, ax = self._init_figure(ylabel='Number of times')
        ax.plot(self.nsteps, ntimes, 'o',
                label=f'Average over {self.nwalk} samples')
        ax.set_xscale('log')
        ax.legend()


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
    walk = Walk2D(100)
    anim = walk.generate_animation()
    walk.plot()

    FinalDistance(nwalk=1000).plot()
    MaxDistance(nwalk=1000).plot()
    BackToStart(nwalk=10000, nstepmax=10000, step_num=8).plot()

    plt.show()
