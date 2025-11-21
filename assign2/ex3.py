import numpy as np
import matplotlib.pyplot as plt


def indicator(x: float) -> float:
    if int(x) == x:  # if it is an integer
        if x == 0.0:
            return 1.0
        else:
            return 0.0
    else:
        return indicator(2 * x) + indicator(2 * x - 1)


def tent(x: float) -> float:
    if int(x) == x:  # if it is an integer
        if x == 1.0:
            return 1.0
        else:
            return 0.0
    else:
        return 0.5 * tent(2 * x) + tent(2 * x - 1) + 0.5 * tent(2 * x - 2)


xs = np.linspace(-1, 3, 4 * 2**4 + 1, endpoint=True)

ys_ind = [indicator(x) for x in xs]
ys_tent = [tent(x) for x in xs]

plt.plot(xs, ys_tent)
plt.plot(xs, ys_ind)
plt.show()
