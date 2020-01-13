import numpy as np
import matplotlib.pyplot as plt
import math

# four possible directions
directions = ((0, 1), (1, 0), (-1, 0), (0, -1))


def distance(nstep: int):
    x = 0
    y = 0

    for istep in range(1, nstep):
        step = directions[np.random.randint(4)]
        x += step[0]
        y += step[1]
    return math.sqrt(x**2 + y**2)


nwalk = 500
#steps = np.arange(50, 1000, 50)
steps = np.logspace(2, 10, base=2, dtype=int, num=5)
dist = np.empty_like(steps)
for istep, nstep in np.ndenumerate(steps):
    print('nstep =', nstep)
    total_dist = 0.
    for iwalk in range(nwalk):
        total_dist += distance(nstep)
    dist[istep] = total_dist / nwalk

plt.plot(steps, dist, '+')
plt.show()
