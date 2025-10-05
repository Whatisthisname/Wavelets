import matplotlib.pyplot as plt
import matplotlib.axes as maxes
import numpy as np


def assert_is_power_of_two(n: int):
    if n & (n - 1) != 0:
        raise ValueError("n must be a power of two")


def assert_is_ndarray(arr: np.ndarray):
    assert isinstance(arr, np.ndarray), "expected ndarray but got {}".format(type(arr))


# Mostly for plotting but might remove (slow)
def eval_in_standard_basis(st_coeffs, x: float) -> float:
    assert_is_power_of_two(len(st_coeffs))

    j = int(np.log2(len(st_coeffs)))
    running_sum = 0.0

    for k in range(len(st_coeffs)):
        (start, end) = (2 ** (-j) * k, 2 ** (-j) * (k + 1))
        running_sum = running_sum + st_coeffs[k] * int(start <= x < end) * 2 ** (j / 2)

    return running_sum


# Mostly for plotting but might remove (slow)
def eval_in_haar_basis(h_coeffs, x: float) -> float:
    assert_is_power_of_two(len(h_coeffs))
    running_sum = int(0 <= x < 1) * h_coeffs[0]

    num_layers = int(np.log2(len(h_coeffs)))

    for j in range(num_layers):
        num_elements_in_layer = int(2**j)
        for k in range(num_elements_in_layer):
            idx = 2**j + k
            (start, mid, end) = (
                2 ** (-j) * k,
                2 ** (-j) * (k + 0.5),
                2 ** (-j) * (k + 1),
            )
            value = (int(start <= x < mid) - int(mid <= x < end)) * 2 ** (j / 2)

            running_sum = running_sum + h_coeffs[idx] * value

    return running_sum


# For Exercise 4
def DHT(st_coeff_vec: np.ndarray) -> np.ndarray:
    j = int(np.log2(len(st_coeff_vec)))
    s = np.sqrt(2) / 2

    result = np.zeros(2**j)
    result[0] = np.sum(st_coeff_vec) * s**j

    for j_ in range(j):
        for k in range(int(2**j_)):
            start = (2 ** (j - j_)) * k
            mid = start + (2 ** (j - j_ - 1))
            end = (2 ** (j - j_)) * (k + 1)

            result[2**j_ + k] = np.sum(
                st_coeff_vec[start:mid] * s ** (j - j_)
                + st_coeff_vec[mid:end] * (-(s ** (j - j_)))
            )

    return result


# For Exercise 4
def iDHT(
    h_coeff_vec: np.ndarray,
) -> np.ndarray:
    j = int(np.log2(len(h_coeff_vec)))
    s = np.sqrt(2) / 2

    result = np.zeros(2**j)
    result += (h_coeff_vec[0]) * s**j

    for j_ in range(j):
        for k in range(int(2**j_)):
            start = (2 ** (j - j_)) * k
            mid = start + (2 ** (j - j_ - 1))
            end = (2 ** (j - j_)) * (k + 1)

            result[start:mid] += h_coeff_vec[2**j_ + k] * (s ** (j - j_))
            result[mid:end] += h_coeff_vec[2**j_ + k] * (-(s ** (j - j_)))

    return result


# For Exercise 4
def FHT(st_coeffs: np.ndarray) -> np.ndarray:
    """Convert from Standard basis to Haar Basis"""
    assert_is_power_of_two(len(st_coeffs))
    assert_is_ndarray(st_coeffs)

    s = np.sqrt(2) / 2

    st_coeffs = st_coeffs.copy()
    j = int(np.log2(len(st_coeffs)))

    for step in range(j):
        coeffs_to_make = int(len(st_coeffs) / 2**step)
        p = np.zeros(coeffs_to_make)
        for k in range(coeffs_to_make // 2):
            left_const_coeff = st_coeffs[2 * k]
            right_const_coeff = st_coeffs[2 * k + 1]

            # The first half of the coeffiecents are the constant coefficients
            p[k] = s * (left_const_coeff + right_const_coeff)
            # And the second half of the coefficients are the wiggle coefficients
            p[coeffs_to_make // 2 + k] = s * (left_const_coeff - right_const_coeff)

        st_coeffs[:coeffs_to_make] = p
    return st_coeffs


# For Exercise 4
def iFHT(h_coeffs: np.ndarray) -> np.ndarray:
    """Convert from Haar basis to Standard Basis"""
    assert_is_power_of_two(len(h_coeffs))

    s = np.sqrt(2) / 2
    h_coeffs = h_coeffs.copy()
    j = int(np.log2(len(h_coeffs)))

    for bstep in range(j - 1, -1, -1):
        step = j - 1 - bstep

        coeffs_to_make = int(len(h_coeffs) / 2**bstep)
        new_coeffs = np.zeros(coeffs_to_make)

        for k in range(
            coeffs_to_make // 2
        ):  # do make two coeffs per iteration so do half the coeffs
            # These correspond to the pairs of constant and difference functions on the same domain and offset
            # idx in the first step: (0 1)
            # idx in the second    : (0 2) (1 3)
            # idx in the third     : (0 4) (1 5) (2 6) (3 7)
            # etc.
            constant_coeff = h_coeffs[k]
            wiggle_coeff = h_coeffs[k + 2**step]

            new_coeffs[2 * k] = s * (constant_coeff + wiggle_coeff)
            new_coeffs[2 * k + 1] = s * (constant_coeff - wiggle_coeff)

        h_coeffs[:coeffs_to_make] = new_coeffs

    return h_coeffs


def discrete_haar_transform_matrix(j: int):
    dim = int(2**j)
    M = np.zeros((dim, dim))
    s = np.sqrt(2) / 2
    M[0, :] = s**j
    for j_ in range(j):
        for k in range(int(2**j_)):
            start = (2 ** (j - j_)) * k
            mid = start + (2 ** (j - j_ - 1))
            end = (2 ** (j - j_)) * (k + 1)
            M[2**j_ + k, start:mid] = s ** (j - j_)
            M[2**j_ + k, mid:end] = -(s ** (j - j_))

    return M


def count_and_plot_nonzeros():
    print(discrete_haar_transform_matrix(j=3).round(2))

    plt.plot(
        [sum(discrete_haar_transform_matrix(j=j).flatten() != 0) for j in range(1, 10)]
    )
    plt.plot([100 + (2**j) * (1 + j) for j in range(1, 10)], label="predicted")
    plt.legend()
    plt.show()


def sample_function_into_st_basis(func, j: int) -> np.ndarray:
    """Samples function 'func' with 2**j points in the midpoints of intervals and scales to convert to standard basis"""
    dim = int(2**j)
    gap = 1 / dim
    xs = np.linspace(0 + gap / 2, 1 - gap / 2, dim, endpoint=True)
    evals = np.array([func(x) for x in xs], dtype=float)
    return evals / (np.sqrt(2) ** j)


def sample_function_average(func, j: int) -> np.ndarray:
    """Samples function 'func' by taking the average of 10 evenly spaced points in each interval"""
    dim = int(2**j)
    gap = 1 / dim
    evals = np.zeros(dim)

    for k in range(dim):
        start = k * gap
        end = (k + 1) * gap

        # Take 10 evenly spaced points in the interval
        sample_points = np.linspace(start, end, 10, endpoint=False)
        samples = np.array([func(x) for x in sample_points])

        # Take the average
        evals[k] = np.mean(samples)

    return evals / (np.sqrt(2) ** j)


# For Exercise 8
def compute_ratios(h_coeffs: np.ndarray) -> np.ndarray:
    assert_is_power_of_two(len(h_coeffs))
    assert_is_ndarray(h_coeffs)
    j = int(np.log2(len(h_coeffs)))

    max_coeff = np.zeros(j)
    for layer in range(j):
        coeffs = h_coeffs[2**layer : 2 ** (layer + 1)]
        max_coeff[layer] = np.max(np.abs(coeffs))

    consecutive_quotients = max_coeff[:-1] / max_coeff[1:]
    return consecutive_quotients


# For Exercise 9
def draw_coefficients_at_layer(
    h_coeffs: np.ndarray, layer: int, ax: maxes.Axes | None = None, color: str = "blue"
) -> maxes.Axes:
    assert_is_power_of_two(len(h_coeffs))
    assert_is_ndarray(h_coeffs)
    if ax is None:
        fig, ax = plt.subplots()
    j = int(np.log2(len(h_coeffs)))
    assert 0 <= layer < j
    coeffs = h_coeffs[2**layer : 2 ** (layer + 1)]

    ax.plot(
        np.linspace(0, 1, 2**layer),
        coeffs,
        c=color,
        label=f"layer {layer}, {len(coeffs)} coefficients",
    )
    return ax


def f(x: float) -> float:
    return np.cos(2 * np.pi * x)


def g(x: float) -> float:
    return np.sqrt(np.abs(np.cos(2 * np.pi * x)))


def exercise_4():
    j = 5

    def func(x):
        return x**2 + 3 * x - 0.3 + 0.5 * np.sin(20 * np.sqrt(np.abs(1.1 - x)))

    xs = np.linspace(0, 1, 1000, endpoint=False)
    ys = [func(x) for x in xs]

    n = 4
    fig, axs = plt.subplots(2, 4, figsize=(4 * n, 8))

    for col, j in zip(axs.T, range(2, 6)):
        st_coeffs = sample_function_into_st_basis(func, j)
        h_coeffs = FHT(st_coeffs)

        ys1 = [eval_in_haar_basis(h_coeffs, x) for x in xs]
        col[0].plot(xs, ys, color="red")
        # col[0].plot(xs, ys1, label=f"Sampled + FHT, dim = {2**j}", color="blue")
        for c, i in zip(st_coeffs, range(2**j)):
            start = i * 2**-j
            end = i * 2**-j + 2**-j
            val = 2 ** (j / 2)
            col[0].plot([start, end], [c * val, c * val], color="blue")

        col[0].legend()
        col[1].scatter(
            range(len(h_coeffs)),
            h_coeffs,
            label="Haar coefficients",
            color="blue",
            s=10,
        )
        col[1].legend()

    empirical_equivalence_check()
    plt.show()


def empirical_equivalence_check():
    for _ in range(10):
        st_coeffs = np.random.randn(2**10)
        fht_h_coeffs = FHT(st_coeffs)
        dht_h_coeffs = DHT(st_coeffs)
        # check that the inverse transforms are correct
        assert np.allclose(st_coeffs, iFHT(fht_h_coeffs))
        assert np.allclose(st_coeffs, iDHT(dht_h_coeffs))

        # check that the FHT and DHT are equivalent
        assert np.allclose(fht_h_coeffs, dht_h_coeffs)
    print("Empirical equivalence check passed")


def exercise_8():
    j = 14
    # First plot for cos(2πx)
    ratios_f = compute_ratios(FHT(sample_function_into_st_basis(f, j)))
    bound_f = 2 * np.sqrt(2)

    # plt.figure(figsize=(5, 3))
    plt.plot(ratios_f, "b-", label="Coefficient ratios")
    plt.axhline(y=bound_f, color="r", linestyle="--", label=f"Trend ({bound_f:.2f})")
    plt.title("Consecutive max coefficient ratios of cos(2πx)")
    plt.xlabel("Ratio index")
    plt.ylabel("Ratio value")
    plt.legend()
    plt.grid(True)

    # 2x1 subplot for √|cos(2πx)| comparisons
    fig, (ax1) = plt.subplots(1, 1)

    # Midpoint sampling plot
    ratios_g = compute_ratios(FHT(sample_function_into_st_basis(g, j)))
    bound_g = 2
    ax1.plot(ratios_g, "b-", label="Coefficient ratios")
    ax1.axhline(y=bound_g, color="r", linestyle="--", label=f"Trend ({bound_g})")
    ax1.set_title("Consecutive max coefficient ratios of √|cos(2πx)|")
    ax1.set_xlabel("Ratio index")
    ax1.set_ylabel("Ratio value")
    ax1.legend()
    ax1.grid(True)

    # # Average sampling plot
    # ratios_g_avg = compute_ratios(FHT(np.sqrt(2**j) * sample_function_average(g, j)))
    # ax2.plot(ratios_g_avg, "b-", label="Coefficient ratios (average sampling)")
    # ax2.axhline(y=bound_g, color="r", linestyle="--", label=f"Trend ({bound_g})")
    # ax2.set_title("√|cos(2πx)| with better projection")
    # ax2.set_xlabel("Ratio index")
    # ax2.set_ylabel("Ratio value")
    # ax2.legend()
    # ax2.grid(True)

    # plt.tight_layout()
    plt.show()
    return

    # Plot comparing g(x) with its piecewise constant approximations
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    for idx, j in enumerate([5, 8]):
        # Original function
        x_fine = np.linspace(0, 1, 1000)
        y_fine = [g(x) for x in x_fine]
        axes[idx, 0].plot(x_fine, y_fine, "k-", linewidth=2, label="g(x) = √|cos(2πx)|")

        # Midpoint sampling
        st_coeffs_midpoint = sample_function_into_st_basis(g, j)
        for c, i in zip(st_coeffs_midpoint, range(2**j)):
            start = i * 2**-j
            end = i * 2**-j + 2**-j
            val = c * 2 ** (j / 2)  # Scale back to original function values
            axes[idx, 0].plot([start, end], [val, val], color="red", linewidth=2)

        # Average sampling
        st_coeffs_average = sample_function_average(g, j)
        for c, i in zip(st_coeffs_average, range(2**j)):
            start = i * 2**-j
            end = i * 2**-j + 2**-j
            val = c * 2 ** (j / 2)  # Scale back to original function values
            axes[idx, 0].plot([start, end], [val, val], color="blue", linewidth=2)

        axes[idx, 0].set_title(f"g(x) = √|cos(2πx)| (j={j})")
        axes[idx, 0].set_xlabel("x")
        axes[idx, 0].set_ylabel("Function value")
        axes[idx, 0].legend(["Original g(x)", "Midpoint sampling", "Average sampling"])
        axes[idx, 0].grid(True, alpha=0.3)

        # Plot difference between midpoint and average sampling
        x_range = np.arange(2**j)
        diff = (st_coeffs_midpoint - st_coeffs_average) * 2 ** (
            j / 2
        )  # Scale back to original values
        axes[idx, 1].plot(x_range, diff, "k-", linewidth=2)
        axes[idx, 1].set_title(f"Difference")
        axes[idx, 1].set_xlabel("Coefficient index")
        axes[idx, 1].set_ylabel("Midpoint - Average")
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def exercise_9():
    j = 10

    fig, axs = plt.subplots(2, 1)
    xs = np.linspace(0, 1, 100, endpoint=False)
    ys = [eval_in_standard_basis(sample_function_into_st_basis(g, j), x) for x in xs]
    axs[0].plot(xs, ys)

    ax = draw_coefficients_at_layer(
        FHT(sample_function_into_st_basis(g, j)), layer=j - 1, ax=axs[1], color="blue"
    )
    ax = draw_coefficients_at_layer(
        FHT(sample_function_into_st_basis(g, j)), layer=j - 2, ax=ax, color="red"
    )
    ax = draw_coefficients_at_layer(
        FHT(sample_function_into_st_basis(g, j)), layer=j - 3, ax=ax, color="green"
    )
    ax.legend()
    plt.show()
    exit()


def exercise_11():
    def func(x):
        return np.arctan(50 * (x - 1 / 3))

    j = 14

    # Sample the function
    interval_midpoints = np.linspace(
        0 + 2 ** (-j + 1), 1 - 2 ** (-j + 1), 2**j, endpoint=True
    )
    approx_func = sample_function_into_st_basis(func, j) * np.sqrt(
        2**j
    )  # have to scale up to get "raw" samples

    # Convert to standard basis by scaling (this is built into the sampling function)
    st_coeffs = sample_function_into_st_basis(func, j)

    # Convert to Haar coefficients
    h_coeffs = FHT(st_coeffs)

    # Investigate consecutive ratios of max haar coefficients:
    print(f"consecutive ratio of max coefficient:\n {compute_ratios(h_coeffs)}")

    abs_sort_idx = np.argsort(np.abs(h_coeffs))

    fig, axs = plt.subplots(1, 2)  # , figsize=(4, 2))
    axs[0].set_title("sorted absolute Haar coefficients")
    sorted_abs_c = np.sort(np.abs(h_coeffs))[::-1]
    axs[0].plot(sorted_abs_c)
    sqrd_c = sorted_abs_c**2
    errors = np.array([np.sqrt(np.sum(sqrd_c[i:])) for i in range(1, 2**j + 1)])
    axs[1].plot(errors, label="Best $N$ term approximation")
    axs[1].set_ylabel("l2 norm of removed coefficients$")
    axs[1].set_xlabel("N")
    axs[1].legend()
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    fig.tight_layout()

    ax = draw_coefficients_at_layer(h_coeffs, layer=j - 1, ax=None, color="blue")
    ax = draw_coefficients_at_layer(h_coeffs, layer=j - 2, ax=ax, color="red")
    ax = draw_coefficients_at_layer(h_coeffs, layer=j - 3, ax=ax, color="green")
    ax.legend()
    # ax.set_title()

    norm_of_approx = np.linalg.norm(h_coeffs)
    target_error = 0.01 * norm_of_approx
    best_N = np.argmin(np.abs(errors - target_error))
    print(
        "best N is",
        best_N,
        "which reconstructs",
        1 - errors[best_N] / norm_of_approx,
        "%",
    )

    h_compressed = h_coeffs.copy()
    h_compressed[abs_sort_idx[:best_N]] = 0

    st_compress = iFHT(h_compressed)

    fig, axs = plt.subplots(1, 2)
    axs[1].set_title(f"{best_N}-term reconstruction of function")
    for c, i in zip(st_compress, range(2**j)):
        start = i * 2**-j
        end = i * 2**-j + 2**-j
        val = 2 ** (j / 2)
        axs[1].plot([start, end], [c * val, c * val], color="blue")

    axs[0].set_title("Approximate projection of function")
    axs[0].plot(interval_midpoints, approx_func, color="blue", label="approx $P_jf$")

    plt.show()


if __name__ == "__main__":
    # exercise_4()  # Show DHT and FHT Sampling
    # exercise_9()  # Show coefficients at each layer
    # exercise_8()  # Show coefficient ratios and compare piecewise constant approximations
    exercise_11()  # Show best N-term approximation
