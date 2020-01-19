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


class Walk2D:

    def __init__(self, nstep: int):
        self.nstep = nstep
        self.x, self.y = self._generate_walk()

    def _generate_walk(self):
        """
        Create a nstep-random walk path on Cartesian grid
        """
        x = np.empty(self.nstep)
        y = np.empty(self.nstep)

        # initial position
        x[0] = 0
        y[0] = 0

        # time loop
        for step in range(1, self.nstep):
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

    def plot(self):
        """Plot walk in a figure"""
        x = self.x
        y = self.y
        fig, ax = self._init_figure()

        path, = ax.plot(x, y)  # a line for path

        # symbols for initial (squared) and final (round) position
        ax.plot(x[0], y[0], 's', color=path.get_color())
        ax.plot(x[-1], y[-1], 'o')


def plot_walk(nstep: int):
    """plot a nstep-2D walk"""
    walk = Walk2D(nstep)
    walk.plot()


class NWalk:
    """
    Abstract class for a n-walk computation
    (should not be instantiated)
    """

    suptitle = ''
    title = ''

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
        self.x_ana = np.linspace(self.nsteps[0], self.nsteps[-1], num=100,
                                 endpoint=True)

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
        ax.set_title(self.title)
        return fig, ax

    def plot(self):
        pass


class Distance(NWalk):
    """
    Abstract class for computin distance as function of nstep
    (should not be instantiated)
    """

    def plot(self):
        """
        Plot mean distance from starting point as function of number of steps
        """

        distances = self.compute_steps()

        fig, ax = self._init_figure()

        ax.plot(self.x_ana, np.sqrt(2 * self.x_ana / pi),
                label=r'$\sqrt{\frac{2n}{\pi}}$')
        ax.plot(self.nsteps, distances, 'o',
                label=f'Average over {self.nwalk} samples')
        ax.legend()


class FinalDistance(Distance):
    """A class to plot final distance as function of nstep"""

    title = 'Distance as a function of number of steps $n$'

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


class MaxDistance(Distance):
    """A class to plot maximum distance as function of nstep"""

    title = 'Maximum distance as a function of number of steps $n$'

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


class BackToStart(NWalk):
    """
    A class to plot the number times when the walk go back to starting point
    """

    suptitle = 'Number of return to starting point'
    title = 'as a function of number of steps $n$'

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

        ax.plot(self.x_ana, np.log(self.x_ana)*3/10,
                label=r'$\frac{10}{3}\ln(n)$')
        ax.plot(self.nsteps, ntimes, 'o',
                label=f'Average over {self.nwalk} samples')
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
    BackToStart(nwalk=10000, nstepmax=10000).plot()

    plt.show()
