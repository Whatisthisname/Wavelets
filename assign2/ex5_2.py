import numpy as np
import matplotlib.pyplot as plt
from helper import get_as, LOWER_a3, UPPER_a3, DAUBECHIES_a3
from ex5_1 import cascade2

if __name__ == "__main__":
    j = 15
    as_ = np.array([-0.25, 0.5, 1.5, 0.5, -0.25])

    resolution = 2 + 1 + (len(as_) - 1) * (2**j - 1)  # is the same as len(result)
    support_width = (resolution - 1) / 2**j
    xs = np.linspace(0, support_width, resolution)

    npoints = 2**5

    result = np.concatenate(([0], cascade2(as_, j)[0], [0]))  # padded with zeros
    plt.plot(xs, result)
    plt.show()
