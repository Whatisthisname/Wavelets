import sympy as sp

# Define the symbolic variable
s = sp.Symbol("s")

# Define the first matrix
A = sp.Matrix([[s, s, 0, 0], [s, -s, 0, 0], [0, 0, s, s], [0, 0, s, -s]])

# Define the second matrix
B = sp.Matrix([[s, 0, s, 0], [0, 1, 0, 0], [s, 0, -s, 0], [0, 0, 0, 1]])

# Compute the matrix product A * B
result = B @ A

print("\nResult A * B:")
sp.pprint(result)

# Simplify the result
simplified_result = sp.simplify(result)
print("\nSimplified result:")
sp.pprint(simplified_result)
