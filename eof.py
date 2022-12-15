import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Create some fake data
n = 100
p = 3
e = np.random.randn(n, p)

# Create the matrix X of the observations
X = np.empty((n, p))
X[:, 0] = 1  # first column is 1
X[:, 1] = np.arange(n)
X[:, 2] = np.arange(n)**2

# Create the matrix Y of the observations
Y = np.dot(X, e.T).T

# Compute the EOFs
U, s, V = linalg.svd(Y, full_matrices=False)

# Compute the EOFs and PCs using the covariance matrix
# (Note: the svd function returns the transpose of V, so we transpose it back)
W, V = linalg.eig(np.cov(Y))
V = V.T

# Sort the EOFs and PCs in decreasing order of the EOFs
ii = np.argsort(-W)
W = W[ii]
V = V[ii, :]

# Normalize the EOFs
V /= np.sqrt(np.sum(V**2, axis=1))[:, np.newaxis]

# Compute the PCs
P = np.dot(Y, V.T)

# Plot the first two EOFs
plt.figure()
plt.plot(V[0, :], label='EOF 1')
plt.plot(V[1, :], label='EOF 2')
plt.legend()
plt.show()

# Plot the first two PCs
plt.figure()
plt.plot(P[:, 0], label='PC 1')
plt.plot(P[:, 1], label='PC 2')
plt.legend()
plt.show()