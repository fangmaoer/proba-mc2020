"""Plot a simple 2D random walk animation"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Fixing random state for reproducibility
np.random.seed(19680801)


def generate_walk(nstep: int):
    """
    Create a nstep-random walk path on Cartesian grid
    walk is a (2, nstep) array with:
        - first dimension: (x, y) coordinates
        - second dimension: step index
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


def generate_animation(nstep: int = 50):
    """return a nstep-random walk animation"""

    def update_path(istep: int):
        """
        Update line date with current time step (istep)
        """
        # plot all edges from start to current position
        path.set_data(walk[:, :istep+1])
        # plot only current position
        spot.set_data(walk[:, istep])

    walk = generate_walk(nstep)

    # Create figure and plot area
    fig, ax = plt.subplots()
    # a continuous line to draw walk path
    path, = ax.plot(walk[0], walk[1])
    # a round-symbol spot to show current position
    spot, = ax.plot(walk[0, 0], walk[1, 0], 'o')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'{nstep}-nstep random walk')
    ax.set_aspect('equal', 'datalim')
    ax.grid(True)

    anim = animation.FuncAnimation(fig,
                                   func=update_path,
                                   frames=nstep,
                                   interval=50,
                                   blit=False,
                                   repeat=False)

    return anim


if __name__ == '__main__':
    anim = generate_animation(100)
    plt.show()
