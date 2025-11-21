import numpy as np
import matplotlib.pyplot as plt
from helper import get_as, LOWER_a3, UPPER_a3, DAUBECHIES_a3
from ex5_1 import cascade2

as_ = np.array(
    [
        0.05349751482162,
        -0.03372823688575,
        -0.15644653305798,
        0.53372823688574,
        1.20589803647272,
        0.53372823688574,
        -0.15644653305798,
        -0.03372823688575,
        0.05349751482162,
    ]
)

print("sum of sqURE", np.sum(as_**2))

j = 15

resolution = 2 + 1 + (len(as_) - 1) * (2**j - 1)  # is the same as len(result)
support_width = (resolution - 1) / 2**j
xs = np.linspace(0, support_width, resolution)
phi, psi = cascade2(as_, j)
padded_phi = np.concatenate(([0], phi, [0]))  # padded with zeros
padded_psi = np.concatenate(([0], psi, [0]))  # padded with zeros

inner_product = np.sum(padded_phi * padded_psi)
print("inner_prod = ", inner_product)

plt.plot(xs, padded_phi, label="phi")
plt.plot(xs, padded_psi, label="psi")
plt.legend()
plt.show()
