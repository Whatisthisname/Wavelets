import numpy as np
import matplotlib.pyplot as plt
from helper import get_as, LOWER_a3, UPPER_a3, DAUBECHIES_a3
from mpl_toolkits.mplot3d import Axes3D
import typing

as_ = get_as(-0.1830127)

as_ = [-0.25, 0.5, 1.5, 0.5, -0.25]


def cascade(as_: np.ndarray, j: int) -> np.ndarray:
    coefficients = np.array([1.0])

    for i in range(1, j + 1):
        result = np.zeros(len(coefficients) + (len(as_) - 1) * 2 ** (i - 1))
        # print(coefficients)
        # print(result)

        for shift, a in enumerate(as_):
            shift_ = shift * 2 ** (i - 1)

            start = shift_
            end = shift_ + len(coefficients)

            result[start:end] += a * coefficients

        coefficients = result

    return coefficients


def cascade2(as_: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.array([1.0])
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


if __name__ == "__main__":
    j = 7

    cmap = plt.get_cmap("viridis")

    resolution = 2 + 1 + (len(as_) - 1) * (2**j - 1)  # is the same as len(result)
    support_width = (resolution - 1) / 2**j
    xs = np.linspace(0, support_width, resolution)

    npoints = 2**5

    # Choose point on 1D manifold of scaling functions
    a3s = np.linspace(0.0, 0.5, npoints)
    a3s = np.linspace(0.0, UPPER_a3 - 0.0001, npoints)
    a3s = np.linspace(1.05, UPPER_a3 - 0.001, npoints)
    a3s = np.linspace(0 + 0.5, UPPER_a3 - 0.001, npoints)
    a3s = np.linspace(LOWER_a3 + 0.0001, 0, npoints)
    a3s = np.linspace(LOWER_a3 + 0.0001, UPPER_a3 - 0.001, npoints)
    a3s = np.linspace(0.95, 1.05, npoints)

    mode: typing.Literal["3D", "2D"] = "3D"
    _3D_value_grid = np.zeros((resolution, npoints))
    for i, a3 in enumerate(a3s):
        as_ = get_as(a3)
        result = np.concatenate(([0], cascade2(as_, j)[1], [0]))  # padded with zeros

        if mode != "3D":
            plt.plot(xs, result, c=cmap(i / (npoints - 1)), label=f"a3={a3:.4f}")
        else:
            _3D_value_grid[:, i] = result

    if mode == "3D":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        X, Y = np.meshgrid(xs, a3s, indexing="ij")
        ax.plot_surface(
            X,
            Y,
            _3D_value_grid,
            facecolors=cmap((Y - Y.min()) / (Y.max() - Y.min())),
            edgecolor="none",
            rcount=len(xs),
            ccount=len(a3s),
            linewidth=0,
        )

        ax.set_xlabel("xs")
        ax.set_ylabel("a3s")
        ax.set_zlabel("Values")
    else:
        plt.legend()
    plt.show()
