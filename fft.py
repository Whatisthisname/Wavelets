import numpy as np


def assert_is_power_of_two(n: int) -> None:
    if n & (n - 1) != 0:
        raise ValueError("n must be a power of two")


def assert_is_ndarray(arr: np.ndarray) -> None:
    assert isinstance(arr, np.ndarray), "expected ndarray but got {}".format(type(arr))


# Basic way to evaluate polynomial
def eval_polynomial(coeffs: np.ndarray, x: float) -> float:
    return np.sum([c * x**i for i, c in enumerate(coeffs)])


# We can split the polynomial f(x) into
# p(x^2) + x * q(x^2)
# where p is a polynomial that has all the even coefficients
# and q is a polynomial with the odd coefficients
# This one does the splitting.
def split_polynomial_coeffs(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert_is_power_of_two(len(coeffs))
    even = coeffs[::2]
    odd = coeffs[1::2]
    return even, odd


# And here we can recursively evaluate a polynomial by splitting.
def eval_poly_with_split(coeffs: np.ndarray, x: float) -> float:
    if len(coeffs) == 1:
        return coeffs[0]
    assert_is_power_of_two(len(coeffs))
    even, odd = split_polynomial_coeffs(coeffs)
    square = x**2
    result = eval_poly_with_split(even, square) + x * eval_poly_with_split(odd, square)
    return result


# Verifying that it works (sanity check - we don't use the evaluation code again)
for i in range(10):
    coeff_rand = np.random.standard_normal(8)
    p = np.random.random() * 100 - 50
    a1 = eval_polynomial(coeff_rand, x=p)
    a2 = eval_poly_with_split(coeff_rand, x=p)
    assert np.allclose(a1, a2)
    del coeff_rand


# This is the same as FFT!!!! Wow. Need time to comprehend this.
def eval_polynomial_in_roots_of_unity_recursive(coeffs: np.ndarray) -> np.ndarray:
    if len(coeffs) == 1:
        return coeffs
    assert_is_power_of_two(len(coeffs))

    roots = np.exp(2 * np.pi * np.linspace(0, 1, len(coeffs), endpoint=False) * 1j)
    even, odd_ = split_polynomial_coeffs(coeffs)

    answer = np.zeros(len(coeffs), dtype=complex)
    even_evals = eval_polynomial_in_roots_of_unity_recursive(even)
    answer[0 : len(coeffs) // 2] = even_evals
    answer[len(coeffs) // 2 :] = even_evals

    odd__evals = eval_polynomial_in_roots_of_unity_recursive(odd_)
    answer[0 : len(coeffs) // 2] += roots[0 : len(coeffs) // 2] * odd__evals
    answer[len(coeffs) // 2 :] += roots[len(coeffs) // 2 :] * odd__evals
    return answer


# This way of implementing the inverse is mind-blowing as well.
# I have not understood why it works yet.
def inverse_fft(values: np.ndarray) -> np.ndarray:
    n = len(values)
    assert_is_power_of_two(n)
    conjugated = np.conjugate(values)
    y = eval_polynomial_in_roots_of_unity_recursive(conjugated)
    return np.conjugate(y) / n


# We can now efficiently multiply polynomials in coefficient form by
# first evaluating them both in the same 2n distinct points,
# multiplying their function output pointwise
# and then mapping it back to the unique polynomial that goes
# through all those points.
def polynomial_mult(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    f1 = eval_polynomial_in_roots_of_unity_recursive(c1)
    f2 = eval_polynomial_in_roots_of_unity_recursive(c2)
    pointwise = f1 * f2
    return inverse_fft(pointwise)


coefs1 = [1, 1, 1, 0]
coefs2 = [0, 1, 0, 0]

print(polynomial_mult(coefs1, coefs2).round(5).real)
