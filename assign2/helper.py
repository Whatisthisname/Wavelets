import sympy as sp
import numpy as np

a0, a1, a2, a3 = sp.symbols("a0, a1, a2, a3")

squares_to_2 = a0**2 + a1**2 + a2**2 + a3**2 - 2
sum_to_2 = a0 + a1 + a2 + a3 - 2
alternate_to_0 = a0 - a1 + a2 - a3

a0_eq = sp.solve(sum_to_2 + alternate_to_0, a0)[0]

mat = sp.Matrix(
    [
        [a0 - 1, 0, 0, 0],  #
        [a2, a1 - 1, a0, 0],  #
        [0, a3, a2 - 1, a1],  #
        [0, 0, 0, a3 - 1],  #
    ]
)
mat2 = sp.Matrix(
    [
        [a0, 0, 0, 0],  #
        [a2, a1, a0, 0],  #
        [0, a3, a2, a1],  #
        [0, 0, 0, a3],  #
    ]
)
char = sp.det(mat)
solutions = sp.solve(char, (a0, a1, a2, a3))

a1_eq = sp.simplify(solutions[1][1].subs({a0: a0_eq}))

eq = squares_to_2.subs({a0: a0_eq, a1: a1_eq})

shall_be_2 = sp.simplify(eq)
a2_eq = sp.solve(shall_be_2, a2)[0]

LOWER_a3, UPPER_a3 = [float(n) for n in sp.solve(-4 * a3**2 + 4 * a3 + 1, a3)]
DAUBECHIES_a3 = -0.1830127


def get_as(a3_input: float) -> np.ndarray:
    a_3 = a3_input
    a_2 = float(a2_eq.subs({a3: a_3}))
    a_1 = float(a1_eq.subs({a3: a_3, a2: a_2}))
    a_0 = float(a0_eq.subs({a3: a_3, a2: a_2, a1: a1_eq}))
    return np.array([a_0, a_1, a_2, a_3])


if __name__ == "__main__":
    as_ = get_as(DAUBECHIES_a3)
    # as_ = np.array([-0.25, 0.5, 1.5, 0.5, -0.25])

    print("sums to 2:", np.sum(as_))
    print("squares to 2:", np.sum(as_**2))
    print("alterate to 0:", np.sum([(-1) ** j * a for j, a in enumerate(as_)]))
