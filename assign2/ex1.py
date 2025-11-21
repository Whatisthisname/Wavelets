import matplotlib.pyplot as plt
import numpy as np
from helper import get_as, DAUBECHIES_a3


def m(as_, chi):
    a_0, a_1, a_2, a_3 = as_
    return 0.5 * (
        a_0 * np.exp(-1j * 0 * chi)
        + a_1 * np.exp(-1j * 1 * chi)
        + a_2 * np.exp(-1j * 2 * chi)
        + a_3 * np.exp(-1j * 3 * chi)
    )


as_ = get_as(DAUBECHIES_a3)

chi_values = np.linspace(-np.pi, np.pi, 501, endpoint=True)
m_values = [m(as_, chi) for chi in chi_values]

chi_values_half = np.linspace(-np.pi / 2, np.pi / 2, 501, endpoint=True)
m_values_half = [m(as_, chi) for chi in chi_values_half]

plt.figure(figsize=(10, 6))
plt.plot(np.real(m_values), np.imag(m_values), label="m(chi)")
plt.plot(
    np.real(m_values_half), np.imag(m_values_half), label="m(chi) from  -pi/2 to pi/2"
)
plt.title("Complex plane plot of m over chi")
plt.xlabel("Real part")
plt.ylabel("Imaginary part")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")


plt.legend()
plt.grid()
plt.show()


print(
    np.absolute(m(as_, chi_values)) ** 2 + np.absolute(m(as_, chi_values + np.pi)) ** 2
)
