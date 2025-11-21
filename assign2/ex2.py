import matplotlib.pyplot as plt
import numpy as np
from helper import get_as


def m(as_, chi):
    a_0, a_1, a_2, a_3 = as_
    return 0.5 * (
        a_0 * np.exp(-1j * 0 * chi)
        + a_1 * np.exp(-1j * 1 * chi)
        + a_2 * np.exp(-1j * 2 * chi)
        + a_3 * np.exp(-1j * 3 * chi)
    )


def theta(as_, chi, M: int):
    return np.prod([m(as_, 2 ** (-j) * chi) for j in range(1, M + 1)])


as_ = get_as(0.1424)
as_ = [1.0, 1.0, 0.0, 0.0]

M = 8
n_points = 1 + 2**M * 100

chi_values = np.linspace(-(2**M) * np.pi, 2**M * np.pi, n_points, endpoint=True)

m_values = [theta(as_, chi, M) for chi in chi_values]

plt.figure(figsize=(10, 6))
# plt.plot(np.real(m_values), np.imag(m_values))
plt.plot(chi_values, np.real(m_values))x
plt.xlim(-10 * np.pi, 10 * np.pi)
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.grid()
plt.show()
