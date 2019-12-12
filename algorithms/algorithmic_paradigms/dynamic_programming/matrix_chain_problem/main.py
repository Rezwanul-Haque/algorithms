import sys
import time


def MatrixChain(matrix_sizes, i, j):
    if i == j:
        return 0
    minimum_computations = sys.maxsize
    for k in range(i, j):
        count = (MatrixChain(matrix_sizes, i, k) + MatrixChain(matrix_sizes, k+1, j) + matrix_sizes[i - 1] * matrix_sizes[k] * matrix_sizes[j])
        
        if count < minimum_computations:
            minimum_computations = count
        return minimum_computations

matrix_sizes = [20, 30, 45, 50]
start_time = time.time()
print("Minimum Multiplications are", MatrixChain(matrix_sizes, 1, len(matrix_sizes) - 1))
total_time = time.time() - start_time
print(total_time)