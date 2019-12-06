"""Plot a random walk animation"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Fixing random state for reproducibility
np.random.seed(19680801)


def generate_walk(nstep: int):
    """
    Create a random walk path on Cartesian grid
    nstep: number of steps
    """
    walk = np.empty((2, nstep))

    # initial position
    walk[:, 0] = np.array((0, 0))
    # four possible directions
    directions = ((0, 1), (1, 0), (-1, 0), (0, -1))

    # time loop
    for istep in range(1, nstep):
        step = directions[np.random.randint(4)]
        walk[:, istep] = walk[:, istep - 1] + step
    return walk


def update_lines(istep: int, data: np.ndarray, path, spot):
    """
    Update line date with current time step (istep)
    """
    # plot all edges from start to current position
    path.set_data(data[:, :istep+1])
    # plot only current position
    spot.set_data(data[:, istep])


def plot_walk(nstep: int = 50):
    """plot a random walk animation"""

    walk = generate_walk(nstep)

    # Create figure and plot area
    fig, ax = plt.subplots()
    # a continuous line to draw walk path
    walk_path = ax.plot(walk[0], walk[1])
    # a round spot to show current position
    walk_spot = ax.plot(walk[0][0], walk[1][0], 'o')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('2D random walk')

    ani = animation.FuncAnimation(fig, update_lines, range(1, nstep),
                                  fargs=(walk, walk_path[0], walk_spot[0]),
                                  interval=50,
                                  blit=False,
                                  repeat=False)

    plt.show()


if __name__ == '__main__':
    plot_walk(100)
