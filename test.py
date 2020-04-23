import random
import numpy as np

A = [(1,2,3), (4,5,6), (7,8,9)]
print(A)

B = random.sample(A,2)
print(B)

print(list(map(np.array, list(zip(*B)))))

print(np.prod(A[0]))