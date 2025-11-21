import numpy as np
import ex5
import ex4
import matplotlib.pyplot as plt
from helper import get_as, LOWER_a3, UPPER_a3

as_ = get_as(UPPER_a3 - 0.0001)
integer_points = ex4.integer_points(ex4.A_matrix(as_))
print("integer points:", integer_points)
result = ex5.cascade(as_, 7)

plt.plot(result)
plt.show()
