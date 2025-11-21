import numpy as np
import matplotlib.pyplot as plt
from helper import get_as, LOWER_a3, UPPER_a3, DAUBECHIES_a3
from ex4 import A_matrix, integer_points


def super_cascade(as_: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray]:
    integer_points_ = integer_points(A_matrix(as_))

    coefficients = -integer_points_
    # coefficients = np.array([1.0])
    old_coeffs = coefficients

    i = 0
    for i in range(1, j + 1):
        result = np.zeros(len(coefficients) + (len(as_) - 1) * 2 ** (i - 1))
        # print(coefficients)
        # print(result)

        for shift, a in enumerate(as_):
            shift_ = shift * 2 ** (i - 1)

            start = shift_
            end = shift_ + len(coefficients)

            result[start:end] += a * coefficients

        old_coeffs = coefficients
        coefficients = result

    psi_coefficients = np.zeros(len(old_coeffs) + (len(as_) - 1) * 2 ** (i - 1))
    for shift, a in enumerate(as_):
        shift_ = (len(as_) - 1 - shift) * 2 ** (i - 1)

        sign = (-1) ** (shift)
        start = shift_
        end = shift_ + len(old_coeffs)

        psi_coefficients[start:end] += sign * a * old_coeffs

    return coefficients, psi_coefficients


as_97 = np.array(
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

as_ = get_as(DAUBECHIES_a3)
as_ = get_as(0.5)
as_ = get_as(1.0)
as_ = get_as(0.0)
as_ = as_97
as_ = np.array([-0.25, 0.5, 1.5, 0.5, -0.25])
j = 20

phi, psi = super_cascade(as_, j)
padded_phi = np.concatenate(([0], phi, [0]))  # padded with zeros
padded_psi = np.concatenate(([0], psi, [0]))  # padded with zeros

resolution = len(padded_phi)
support_width = (resolution - 1) / 2**j
xs = np.linspace(0, support_width, resolution)

l2_inner_product = np.sum(padded_phi * padded_psi)
print(l2_inner_product)

plt.plot(xs, padded_phi, label="phi")
plt.plot(xs, padded_psi, label="psi")
plt.legend()
plt.show()
