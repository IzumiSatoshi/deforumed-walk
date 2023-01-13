import numpy as np

# fmt: off
A = np.array(
    [[1, 0, 0, 5], 
     [0, 1, 0, 5], 
     [0, 0, 1, 5]]
)

x = np.array([1, 2, 3, 1])

print(A @ x)
