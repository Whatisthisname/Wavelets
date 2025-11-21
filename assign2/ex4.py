import numpy as np
from helper import get_as
import matplotlib.pyplot as plt


def A_matrix(as_):
    n = len(as_) - 1
    mat = np.zeros((n + 1, n + 1))

    for i in range(1, n + 2):  # {1, n+1}
        for j in range(1, n + 2):
            idx = 2 * (i - 1) - (j - 1)
            if 0 <= idx <= n:
                mat[i - 1, j - 1] = as_[2 * (i - 1) - (j - 1)]
    return mat


def integer_points(mat: np.ndarray) -> np.ndarray:
    eigval, eigvec = np.linalg.eig(mat)
    print(eigval)
    _1idx = np.nonzero(np.isclose(eigval, 1.0))[0][0]
    return eigvec[:, _1idx].real * np.sqrt(2)  # it is normalized so


if __name__ == "__main__":
    for number in np.linspace(-0.1830127, -0.1830127 + 0.2, 5)[:2]:
        number = 0

        as_ = get_as(number)
        print(A_matrix(as_))

        # multiply by sqrt(2) to get it to norm 1
        integer_points_ = integer_points(A_matrix(as_))
        print(as_)
        print(integer_points_)

        def magic(x: float) -> float:
            if int(x) == x:  # if it is an integer
                if 0 <= x <= 3:
                    return integer_points_[int(x)]
                else:
                    return 0.0
            else:
                return np.sum([as_[i] * magic(2 * x - i) for i in range(4)])

        xs = np.linspace(-1, 4, 5 * 2**4 + 1, endpoint=True)
        ys_ind = [magic(x) for x in xs]

        plt.plot(xs, ys_ind, label=f"a[3]={number}")
        plt.legend()
        break

    plt.show()
