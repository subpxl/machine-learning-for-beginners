import numpy as np

a = np.arange(1000000)

print(a.nbytes)

# 3. Coordinate (COO) Format

# One common sparse format.
# Only nonzero entries stored. to save maemory 

# TF-IDF Matrices

# Very sparse.

# One-Hot Encoding
import numpy as np
from scipy.sparse import csr_matrix

dense = np.array([
    [0,0,3],
    [4,0,0],
    [0,5,0]
])

sparse = csr_matrix(dense)

print(sparse)

dense_again = sparse.toarray()

print(dense_again)