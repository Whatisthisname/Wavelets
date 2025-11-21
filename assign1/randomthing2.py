import sympy as sp

# Define the symbolic variable
s = sp.Symbol("s")
# Create a 4x4 block matrix with 2x2 blocks on diagonal
block = sp.Matrix([[s, s], [s, -s]])
C = sp.BlockMatrix(
    [
        [block, sp.zeros(2), sp.zeros(2), sp.zeros(2)],
        [sp.zeros(2), block, sp.zeros(2), sp.zeros(2)],
        [sp.zeros(2), sp.zeros(2), block, sp.zeros(2)],
        [sp.zeros(2), sp.zeros(2), sp.zeros(2), block],
    ]
)
M1 = C.as_explicit()

# from indicator basis to first Haar basis we get
# [c_30, c_31, c_32, c_33, c_34, c_35, c_36, c_37] # this is indicator basis
# [c_20, h_20, c_21, h_21, c_22, h_22, c_23, h_23] # this is first Haar basis

# Now we want to convert to the next layer
# [c_20, h_20, c_21, h_21, c_22, h_22, c_23, h_23] # this is first Haar basis
# [c_10, h_20, h_10, h_21, c_11, h_22, h_11, h_23] # this is second Haar basis

# Need this matrix
M2 = sp.Matrix(
    [
        [s, 0, s, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [s, 0, -s, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, s, 0, s, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, s, 0, -s, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)

# Now we want to convert to the next layer
# [c_10, h_20, h_10, h_21, c_11, h_22, h_11, h_23] # this is second Haar basis
# [c_00, h_20, h_10, h_21, h_00, h_22, h_11, h_23] # this is third Haar basis

# Need this matrix
M3 = sp.Matrix(
    [
        [s, 0, 0, 0, s, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [s, 0, 0, 0, -s, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)

sp.pprint(M1)
sp.pprint(M2)
sp.pprint(M3)
result = M3 @ M2 @ M1

simplified_result = sp.simplify(result)
print("\nSimplified result:")
sp.pprint(simplified_result)
