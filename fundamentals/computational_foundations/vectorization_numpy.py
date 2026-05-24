# Vectorization in NumPy means performing operations on entire
#  arrays at once instead of using explicit Python loops.

import numpy as np

a = np.array([1,2,3,4])

result = []

for i in range(len(a)):
    result.append(a[i] * 2)

print(result)


a = np.array([1,2,3,4])

result = a * 2

print(result)

a = np.array([1,2,3])
b = np.array([4,5,6])

c = a + b

print(c)

# Vectorization Is Fast Because NumPy arrays are:

# contiguous memory blocks
# fixed data types
# processed in compiled C loops

# This avoids Python interpreter overhead