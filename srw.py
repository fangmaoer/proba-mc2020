"""Create a simple 2D random walk animation"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sqrt, pi

# Fixing random state for reproducibility
np.random.seed(19680801)

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


def compute_distance(nstep: int):
    x = y = 0
    dir = np.random.randint(4, size=nstep)
    for step in range(nstep):
        dx, dy = directions[np.random.randint(4)]
        x += dx
        y += dy
    return sqrt(x**2 + y**2)


def create_1Dfigure():
    """Return a figure and plot area for a nstep-walk"""
    # Create figure and plot area
    fig, ax = plt.subplots()
    ax.set_xlabel('$n$')
    ax.set_ylabel('$d$')
    ax.set_title('Distance as function of number of steps $n$')
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
    starting_point, = ax.plot(x[0], y[0], 's', color=path.get_color())
    ending_point, = ax.plot(x[-1], y[-1], 'o')


def plot_distance(nwalk: int = 1000):
    """
    Plot mean distance from starting point as function of number of steps
    """

    def compute_distances(nwalk: int):
        """
        return a mean over nwalk samples of the distance for various nsteps
        """
        dist = np.empty_like(nsteps, dtype=float)
        for istep, nstep in np.ndenumerate(nsteps):
            # Mean over nwalk realizations
            sum = 0.
            for iwalk in range(nwalk):
                sum += compute_distance(nstep)
            dist[istep] = sum / nwalk
        return dist

    nsteps = np.arange(0, 1000, 100)

    fig, ax = create_1Dfigure()

    ax.plot(nsteps, np.sqrt(2 * nsteps / pi), label=r'$\sqrt{\frac{2n}{\pi}}$')
    ax.plot(nsteps, compute_distances(nwalk), '+',
            label=f'Average over {nwalk} samples')
    ax.legend()


if __name__ == '__main__':
    # anim = generate_animation(100)
    plot_distance(500)
    plt.show()
