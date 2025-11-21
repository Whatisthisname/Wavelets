import numpy as np
import matplotlib.pyplot as plt
from helper import get_as, LOWER_a3, UPPER_a3, DAUBECHIES_a3
from ex5_1 import cascade2

as_ = get_as(DAUBECHIES_a3)

j = 7

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
exit()

as_ = get_as(1.0)

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
