"""Chapter 2: The Mathematics of AI

Section-aligned walkthrough: every code block from this chapter of the book,
in book order, each in its own cell under its section heading so you can run
and check them one by one. Generated from Book.md; keep the book as the source.
"""

# %% [markdown]
# ## 2.0.1 Functions, composition, and inverses

# %% [markdown]
# The computer can confirm the composition and the inverse at once.

# %%
# Composition: apply g first, then f. The order is part of the meaning.
g = lambda x: 2 * x + 1          # doubles and adds one
f = lambda u: u ** 2             # squares

print(f(g(3)))                   # 49   g first: 3 -> 7, then 7 squared
print(g(f(3)))                   # 19   f first: 3 -> 9, then 2 * 9 + 1

# The inverse of g undoes it: solve y = 2x + 1 for x
g_inv = lambda y: (y - 1) / 2
print(g_inv(g(3)))               # 3.0  back where we started

# %% [markdown]
# ## 2.0.2 Exponentials and logarithms

# %% [markdown]
# Without the log laws, the arithmetic would not survive at all.

# %%
import numpy as np

# e emerges from compounding ever more often: (1 + 1/n)^n
for n in [1, 10, 1000, 1_000_000]:
    print(n, (1 + 1/n) ** n)            # 2.0, then 2.5937..., 2.71692..., 2.71828...

# the product law: ln(2*3) should equal ln 2 + ln 3
print(np.log(2) + np.log(3))            # 1.791759469228055
print(np.log(6))                        # 1.791759469228055  the same number

# the power law: ln(2^10) should equal 10 * ln 2
print(np.log(2 ** 10), 10 * np.log(2))  # 6.931471805599453 6.931471805599453

# change of base: log2 of x is ln x divided by ln 2
print(np.log(10) / np.log(2))           # 3.3219280948873626
print(np.log2(10))                      # 3.321928094887362  one unit in the last place from the division
# their difference, beside np.spacing: the distance from a float to its nearest neighbor
print(np.log(10) / np.log(2) - np.log2(10), np.spacing(np.log2(10)))   # 4.440892098500626e-16 4.440892098500626e-16

# monotonicity: the biggest score is also the biggest log score
scores = np.array([0.008, 0.036, 0.012])              # three candidate models
print(np.argmax(scores), np.argmax(np.log(scores)))   # 1 1  the same winner either way

# why we bother: a direct product of 1000 small probabilities underflows...
print(0.01 ** 1000)                     # 0.0  the true value 1e-2000 is unrepresentable
# ...while the sum of their logs is an ordinary number
print(1000 * np.log(0.01))              # -4605.170185988091

# %% [markdown]
# ## 2.0.3 The sigmoid and the logit, an inverse pair

# %% [markdown]
# The computer can confirm the round trip, and one floating-point caution, at once.

# %%
import numpy as np

# An inverse pair: the sigmoid squashes the real line into (0, 1),
# and the logit stretches (0, 1) back over the real line.
sigmoid = lambda z: 1 / (1 + np.exp(-z))
logit   = lambda p: np.log(p / (1 - p))

p = sigmoid(2.0)
print(round(float(p), 3))                      # 0.881
print(round(float(logit(p)), 3))               # 2.0   the inverse recovers the input
print(round(float(logit(sigmoid(-1.5))), 3))   # -1.5  works anywhere on the domain

# Floating point is not exact arithmetic: far enough out, the sigmoid rounds to exactly 1.0
print(sigmoid(36.0) == 1.0, sigmoid(37.0) == 1.0)   # False True

# PyTorch works in 32-bit floats by default, which carry fewer digits and reach the edge sooner
import torch
print(torch.get_default_dtype())                    # torch.float32
sigmoid32 = lambda z: np.float32(1) / (np.float32(1) + np.exp(-np.float32(z)))
print(sigmoid32(16.0) == 1.0, sigmoid32(17.0) == 1.0)   # False True

# %% [markdown]
# ## 2.0.4 Angles, sine and cosine, and the hyperbolic tangent

# %% [markdown]
# It subtracts each year's average to remove the long upward trend, averages what is left by calendar month to show the seasonal cycle, and measures distances between months on the circle.

# %%
import numpy as np
import statsmodels.api as sm

angles = np.array([0, np.pi / 6, np.pi / 4, np.pi / 2, np.pi])   # 0, 30, 45, 90, and 180 degrees
print(np.round(np.cos(angles), 4))                 # [ 1.      0.866   0.7071  0.     -1.    ]
print(np.round(np.sin(angles), 4))                 # [0.     0.5    0.7071 1.     0.    ]

theta = np.linspace(-10, 10, 1001)                 # 1,001 angles, negative and positive
print(np.allclose(np.sin(theta)**2 + np.cos(theta)**2, 1.0))   # True
a, b = 0.7, 1.9
print(np.isclose(np.sin(a + b), np.sin(a) * np.cos(b) + np.cos(a) * np.sin(b)),
      np.isclose(np.cos(a + b), np.cos(a) * np.cos(b) - np.sin(a) * np.sin(b)))   # True True
print(round(float(np.sin(90)), 4), float(np.sin(np.radians(90))))   # 0.894 1.0  NumPy reads radians

sigmoid = lambda z: 1 / (1 + np.exp(-z))
x = np.linspace(-5, 5, 11)
print(np.allclose(np.tanh(x), 2 * sigmoid(2 * x) - 1), round(float(np.tanh(1.0)), 4))   # True 0.7616

co2 = sm.datasets.co2.load_pandas().data.dropna()  # weekly Mauna Loa CO2 readings, in ppm
yearly = co2.groupby(co2.index.year)["co2"].mean()
seasonal = co2["co2"].to_numpy() - yearly.loc[co2.index.year].to_numpy()   # each reading minus its year's mean
profile = [seasonal[co2.index.month == m].mean() for m in range(1, 13)]
print(len(co2), np.round(profile, 2))              # 2225 [-0.6 0.15 ... -0.27]  January to December

angle = 2 * np.pi * np.arange(1, 13) / 12          # month m sits at the angle 2 pi m / 12
pts = np.column_stack([np.sin(angle), np.cos(angle)])
dec_jan = pts[11] - pts[0]                         # December minus January on the circle
jun_jul = pts[6] - pts[5]                          # July minus June
print(12 - 1, round(float(np.sqrt(np.sum(dec_jan**2))), 4),
      round(float(np.sqrt(np.sum(jun_jul**2))), 4))   # 11 0.5176 0.5176

# %% [markdown]
# ## 2.1.1 Scalars, vectors, matrices, and tensors

# %% [markdown]
# When we say we feed the data into a model, we mean we hand the model one of these tensors.

# %%
import numpy as np

scalar = 21.5
vector = np.array([1800, 3, 12])              # shape (3,)   one house
matrix = np.array([[1800, 3, 12],
                   [1500, 2, 30],
                   [2400, 4,  5]])             # shape (3, 3) three houses
image  = np.zeros((28, 28, 3))                # shape (28,28,3) a 28x28 color image

print(vector.shape, matrix.shape, image.shape)   # (3,) (3, 3) (28, 28, 3)
print(matrix[1, 2])                               # 30  row 2 (index 1), column 3 (index 2)

# %% [markdown]
# ## 2.1.2 Adding, scaling, and transposing

# %% [markdown]
# The rule is simply .

# %%
u = np.array([1, 2, 3])
print(u + np.array([10, 20, 30]))   # [11 22 33]  entry by entry, the hand result
print(2 * u)                        # [2 4 6]     every entry scaled by 2

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A + 10)        # broadcasting (1.5.3): the single 10 is added to every entry
print(2 * A)         # scales every entry
print(A.T)           # transpose, shape (3, 2)

# %% [markdown]
# ## 2.1.3 Matrix multiplication, the operation AI is built on

# %% [markdown]
# Hardware and numerical libraries are built to do this one operation in bulk, which is why we work hard to express models as matrix multiplies.

# %%
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot([1, 2, 3], [4, 5, 6]))   # 32     matches the hand calculation
print(A @ B)                            # [[19 22]
                                        #  [43 50]]  matches exactly
print(B @ A)                            # [[23 34]   the other order: a different grid
                                        #  [31 46]]
print(A * B)                            # [[ 5 12]   elementwise, NOT matrix multiply
                                        #  [21 32]]

# %% [markdown]
# The block checks the laws on the 442-by-10 feature matrix of scikit-learn's diabetes dataset, which the lab in 2.10 describes in full.

# %%
import numpy as np
from sklearn.datasets import load_diabetes

X = load_diabetes().data                          # 442 patients by 10 features
rng = np.random.default_rng(0)
B = rng.normal(size=(10, 4))                      # three random matrices whose shapes
C = rng.normal(size=(4, 3))                       # fit the products below
D = rng.normal(size=(10, 4))

print(np.allclose((X @ B) @ C, X @ (B @ C)))      # True  associativity: the grouping is free
print(np.allclose(X @ (B + D), X @ B + X @ D))    # True  distributivity
print(np.allclose((X @ B).T, B.T @ X.T))          # True  a transposed product reverses its order

G = X.T @ X                                       # the 10 by 10 matrix of the normal equation
print(G.shape, np.allclose(G, G.T))               # (10, 10) True  X^T X is symmetric

v = rng.normal(size=442)
print(np.allclose(X @ (X.T @ v), (X @ X.T) @ v))  # True  the same vector from either grouping
n, d = X.shape
print(2 * n * d, n * n * d + n * n)               # 8840 2149004  multiplications for each grouping

Q = np.array([[0., -1.], [1., 0.]])               # a quarter turn, an orthogonal matrix
print(Q.T @ Q)                                    # [[1. 0.]
                                                  #  [0. 1.]]  Q^T Q = I

# %% [markdown]
# ## 2.1.4 The inverse, the determinant, and solving systems

# %% [markdown]
# We give this situation its proper name in 2.1.7 when we reach rank, and we meet its cure, regularization, throughout the book.

# %%
A = np.array([[4., 7.], [2., 6.]])
print(np.linalg.det(A))        # 10.000000000000002  the determinant, 10 up to rounding
print(np.linalg.inv(A))        # [[ 0.6 -0.7]
                               #  [-0.2  0.4]]
b = np.array([1., 2.])
print(np.linalg.solve(A, b))   # the x that satisfies A x = b, without forming the inverse

# %% [markdown]
# Example 2-1 implements elimination with partial pivoting and back substitution in one short function, reruns both hand examples, and then solves the normal equation of 2.1.11 for a linear regression on the diabetes data, with an intercept and all ten features.

# %%
import numpy as np
from scipy.linalg import lu
from sklearn.datasets import load_diabetes

def solve_by_elimination(A, b, pivot=True):
    """Solve A x = b by Gaussian elimination, then back substitution."""
    A = np.array(A, dtype=float)                  # work on copies, since elimination overwrites rows
    b = np.array(b, dtype=float)
    n = len(b)
    for k in range(n - 1):                        # clear every entry below the pivot A[k, k]
        if pivot:                                 # partial pivoting: swap up the row whose
            p = k + np.argmax(np.abs(A[k:, k]))   # entry in column k is largest in size
            A[[k, p]], b[[k, p]] = A[[p, k]], b[[p, k]]
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]                 # the multiplier, which becomes the entry L[i, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):                # back substitution, from the last row up
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x

print(solve_by_elimination([[4, 3, 2], [2, 1, 1], [6, 5, 4]], [1, 2, 3]))   # [ 1. -3.  3.]

tiny = [[1e-20, 1.0], [1.0, 1.0]]                 # a tiny first pivot; the exact answer is close to [1, 1]
print(solve_by_elimination(tiny, [1.0, 2.0], pivot=False))   # [0. 1.]  x1 is ruined
print(solve_by_elimination(tiny, [1.0, 2.0]))                # [1. 1.]  the row swap repairs it

data = load_diabetes()
X = np.column_stack([np.ones(442), data.data])    # a column of ones for the intercept, then 10 features
G, rhs = X.T @ X, X.T @ data.target               # the normal equation G w = rhs of 2.1.11
w = solve_by_elimination(G, rhs)
print(np.round(w[:3], 2))                         # [ 152.13  -10.01 -239.82]  intercept, age, sex
print(f"{np.abs(w - np.linalg.solve(G, rhs)).max():.1e}")   # 5.8e-12  agrees with NumPy's solver
P, L, U = lu(G)                                   # SciPy's pivoted elimination, kept as factors
print(f"{np.abs(P @ L @ U - G).max():.1e}")       # 2.2e-16  the factors rebuild G

# %% [markdown]
# That is why NumPy's `np.linalg.det` computes the determinant from an LU factorization (NumPy developers, 2026a, documentation of `numpy.linalg.det`).

# %%
from scipy.linalg import lu

A = np.array([[4., 3., 2.], [2., 1., 1.], [6., 5., 4.]])
print(4 * (-0.5) * 1, np.linalg.det(A))      # the product of the hand pivots, then NumPy's determinant
P, L, U = lu(A)                               # SciPy pivots, so it swaps rows before eliminating
print(np.round(np.diag(U), 4), round(np.linalg.det(P)))   # its pivots, and the sign of its swaps

# %% [markdown]
# ## 2.1.5 Three products you must not confuse: dot, outer, and Hadamard

# %% [markdown]
# It appears wherever gates and masks appear, for example the forget gate inside an LSTM in Chapter 14.

# %%
a = np.array([1, 2, 3]); b = np.array([4, 5, 6])
print(a @ b)                    # 32           dot, returns a scalar
print(np.outer(a, [4, 5]))      # 3x2 matrix   outer, returns a matrix
print(a * b)                    # [ 4 10 18]   Hadamard, keeps the shape

# %% [markdown]
# Both sides give , as the identity promised.

# %%
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# The Kronecker product: every entry of A times a full copy of B,
# so the top-left block is 1*B and the bottom-right block is 4*B
print(np.kron(A, B))                 # [[ 5  6 10 12]
                                     #  [ 7  8 14 16]
                                     #  [15 18 20 24]
                                     #  [21 24 28 32]]

# Two flat 1D vectors give the outer-product grid flattened row by row
print(np.kron([1, 2, 3], [4, 5]))    # [ 4  5  8 10 12 15]

# vec stacks COLUMNS, so ask for Fortran order, not NumPy's row-major default
vec = lambda M: M.reshape(-1, order="F")
X = np.array([[1., 1.], [1., 1.]])

lhs = vec(A @ X @ B)                 # flatten the matrix sandwich
rhs = np.kron(B.T, A) @ vec(X)       # one big matrix times one long vector
print(lhs)                           # [36. 84. 42. 98.]
print(np.allclose(lhs, rhs))         # True: vec(A X B) = (B^T kron A) vec(X)

# %% [markdown]
# ## 2.1.6 Norms, the size of a vector

# %% [markdown]
# the norm you choose changes what the model learns.

# %%
x = np.array([3., -4.])
print(np.linalg.norm(x, 2))       # 5.0   L2
print(np.linalg.norm(x, 1))       # 7.0   L1
print(np.linalg.norm(x, np.inf))  # 4.0   L-infinity, the largest absolute entry
print(round(np.linalg.norm(x, 4), 4), round(np.linalg.norm(x, 10), 4))   # 4.2846 4.022  p = 4 and p = 10 close in on 4

# %% [markdown]
# if every vector is first normalized to unit length by dividing it by its own L2 norm, the denominator becomes 1 and cosine similarity is again just a dot product, which is why the retrieval systems of Chapter 26 usually store embeddings alr

# %%
import numpy as np

def cosine(u, v):
    # the dot product, divided by the product of the two L2 norms
    return u @ v / (np.linalg.norm(u) * np.linalg.norm(v))

a = np.array([1., 2., 3.])
b = np.array([4., 5., 6.])
print(round(cosine(a, b), 4))          # 0.9746  matches 32 / 32.833
print(round(cosine(a, 10 * b), 4))     # 0.9746  ten times longer, same direction
print(cosine(np.array([1., 0.]), np.array([0., 1.])))   # 0.0  perpendicular
print(round(cosine(a, -a), 4))         # -1.0    exactly opposite

q  = np.array([1., 2., 3.])            # the query
d1 = np.array([2., 4., 6.])            # on-topic, short
d2 = np.array([30., 5., 0.])           # off-topic, long
print(q @ d1, q @ d2)                  # 28.0 40.0   raw dot ranks d2 first
print(round(cosine(q, d1), 4), round(cosine(q, d2), 4))  # 1.0 0.3515  cosine ranks d1 first

# %% [markdown]
# Every one of these high-dimensional claims can be checked in a few lines.

# %%
import numpy as np
from math import gamma, pi

# volume flees to the boundary: shave 5 percent off every face of a unit cube
for d in [2, 10, 100, 1000]:
    print(d, f"{0.9 ** d:.3g}")          # 2 0.81   10 0.349   100 2.66e-05   1000 1.75e-46

# the inscribed ball loses to the corners: ball volume over cube volume
for d in [2, 10, 20]:
    ball = pi ** (d / 2) / gamma(d / 2 + 1)    # radius-1 ball in d dimensions
    print(d, f"{ball / 2 ** d:.3g}")           # 2 0.785   10 0.00249   20 2.46e-08

# lengths concentrate on sqrt(d), and random directions become perpendicular
rng = np.random.default_rng(0)
for d in [3, 100, 768, 10000]:
    X = rng.standard_normal((2000, d))         # 2000 random Gaussian vectors
    length = np.linalg.norm(X, axis=1)
    U = X / length[:, None]                    # scale each to unit length
    cos = (U[:1000] * U[1000:]).sum(axis=1)    # cosines of 1000 independent pairs
    print(d, round(length.mean(), 2), round(length.std(), 2),
          round(cos.std(), 3), round(1 / np.sqrt(d), 3))

# %% [markdown]
# Spectral normalization, which divides each weight matrix by its own spectral norm so that every layer has spectral norm 1, was introduced by Miyato et al. (2018) to stabilize the training of a GAN's discriminator, and 24.5 lists it among the remedies for unstable GAN training.

# %%
import numpy as np

A = np.array([[3., 0.],
              [0., 2.]])                     # stretches the x axis by 3, the y axis by 2
print(np.linalg.norm(A, 'fro'))              # 3.605551275463989   sqrt(13), the total energy
print(np.linalg.norm(A, 2))                  # 2.9999999999999996  the spectral norm, 3 up to rounding
print(np.linalg.norm(A, 'nuc'))              # 5.0                 the nuclear norm, 3 + 2

# hunt for the worst stretch by brute force: feed A a hundred thousand random directions
rng = np.random.default_rng(0)
x = rng.standard_normal((2, 100_000))
x = x / np.linalg.norm(x, axis=0)            # scale every column to unit length
print(np.linalg.norm(A @ x, axis=0).max())   # 2.9999999999288236  the search rediscovers the spectral norm

# a rank-1 matrix has one stretch direction, so all three norms collapse to one number
R = np.outer([1., 2., 3.], [4., 5.])         # the 3 by 2 outer product from 2.1.5
for kind in ['fro', 2, 'nuc']:
    print(np.linalg.norm(R, kind))           # 23.958297101421877, matching to the last digit or two

# %% [markdown]
# ## 2.1.7 Linear independence, span, basis, and rank

# %% [markdown]
# The usual fixes are to drop the redundant feature or to regularize.

# %%
A = np.array([[1, 0, 2],
              [0, 1, 3],
              [0, 0, 0]])     # the third row adds no new direction
print(np.linalg.matrix_rank(A))   # 2, not 3, so the matrix is rank deficient

# %% [markdown]
# The rank printout of 2 said that one direction is redundant somewhere; the vector names the precise combination of columns that cancels, which is what you need in order to repair a dataset.

# %%
import numpy as np
from scipy.linalg import null_space

A = np.array([[1., 0., 2.],
              [0., 1., 3.],
              [0., 0., 0.]])
v = np.array([-2., -3., 1.])              # the nullspace direction we found by hand

print(A @ v)                              # [0. 0. 0.]   A squashes v to nothing
print(-2*A[:, 0] - 3*A[:, 1] + A[:, 2])   # [0. 0. 0.]   the column recipe v encodes
N = null_space(A)                         # a basis for the nullspace, columns are unit vectors
print(N.ravel().round(4))                 # [-0.5345 -0.8018  0.2673], which is v / sqrt(14)
print(np.linalg.matrix_rank(A) + N.shape[1])   # 3, rank plus nullity, the number of columns

# %% [markdown]
# The difference is a nullspace direction of the data matrix, because with the centimeter column, , and sliding any distance along that direction changes the coefficients without changing a single prediction.

# %%
import numpy as np

cm = np.array([170., 180., 160.])
X = np.column_stack([cm, cm / 2.54])   # height in centimeters and the same height in inches
w  = np.array([1., 0.])                # one perfectly good weight vector
w2 = w + np.array([1., -2.54])         # the same vector shifted along the nullspace direction

print(X @ w)                           # [170. 180. 160.]
print(X @ w2)                          # [170. 180. 160.]   identical predictions
print(np.abs(X @ w - X @ w2).max())    # 0.0, the data cannot tell the two models apart

# %% [markdown]
# ## 2.1.8 Eigenvalues and eigenvectors, what a matrix does to space

# %% [markdown]
# The same idea underlies spectral clustering and, with a twist, the original PageRank algorithm.

# %%
A = np.array([[2., 1.], [1., 2.]])
vals, vecs = np.linalg.eig(A)
print(vals)            # [3.+0.j 1.+0.j]   the eigenvalues, stored as complex numbers
print(vecs)            # columns are unit-length versions of [1, 1] and [1, -1]

# %% [markdown]
# The sentence "the first component explains 62 percent of the variance" means exactly divided by the trace, an accounting Chapter 8 performs on real data.

# %%
import numpy as np

A = np.array([[2., 1.], [1., 2.]])
vals = np.linalg.eigvalsh(A)            # A is symmetric: real eigenvalues [1. 3.], in ascending order

# the two bookkeeping identities: trace = eigenvalue sum, det = eigenvalue product
print(np.trace(A), vals.sum())          # 4.0 4.0
print(np.linalg.det(A), vals.prod())    # 2.9999999999999996 3.0

# tr(M^T M) is the squared Frobenius norm from 2.1.6
M = np.array([[1., 2.], [3., 4.]])
print(np.trace(M.T @ M), np.linalg.norm(M)**2)   # 30.0 30.0

# the cyclic property: rotating the factors preserves the trace...
B = np.array([[0., 1.], [2., 3.]])
C = np.array([[1., 4.], [5., 6.]])
print(np.trace(A @ B @ C), np.trace(B @ C @ A), np.trace(C @ A @ B))  # 85.0 85.0 85.0
# ...but genuinely reordering them does not
print(np.trace(A @ C @ B))              # 87.0

# %% [markdown]
# the loss curves upward, or at worst stays flat, along every direction you probe.

# %%
import numpy as np

A = np.array([[2., 1.], [1., 2.]])   # the section's worked matrix, eigenvalues 3 and 1
B = np.array([[1., 3.], [3., 1.]])   # every entry positive, but do not trust the entries
C = np.array([[1., 1.], [1., 1.]])   # the covariance shape of two perfectly redundant features

# eigvalsh is the eigenvalue routine specialized for symmetric matrices (ascending order)
for name, M in [("A", A), ("B", B), ("C", C)]:
    print(name, np.linalg.eigvalsh(M))        # A [1. 3.]   B [-2.  4.]   C [0. 2.]

x = np.array([1., -1.])                       # the probe that exposes B and flattens C
print("probe on B:", x @ B @ x)               # -4.0  the form goes negative, B is indefinite
print("probe on C:", x @ C @ x)               # 0.0   no response, C is semidefinite, not definite

# any matrix of the form G^T G is positive semidefinite: every response is a squared length
rng = np.random.default_rng(0)
G = rng.standard_normal((50, 4))              # think of G as 50 centered data points, 4 features
S = G.T @ G                                   # the shape every covariance and kernel matrix has
probes = rng.standard_normal((100000, 4))     # one random probe direction per row
responses = (probes @ S * probes).sum(axis=1)     # row i gives the quadratic form x_i^T S x_i
print("smallest of 100000 responses:", responses.min())   # about 0.248, never negative

# %% [markdown]
# ## 2.1.9 Eigendecomposition and the SVD, factoring a matrix into meaning

# %% [markdown]
# This is the Eckart-Young theorem, and it is rare for theory to hand you the optimal answer so directly.

# %%
from numpy.linalg import svd
rng = np.random.default_rng(0)
# real data has structure: a few patterns explain most of it, unlike pure noise
base = rng.standard_normal((100, 5)) @ rng.standard_normal((5, 80))   # a rank-5 signal
A = base + 0.05 * rng.standard_normal((100, 80))                      # plus a little noise
U, s, Vt = svd(A, full_matrices=False)
k = 10                                 # keep only the 10 largest singular values
A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
# stored numbers: 100*10 + 10 + 10*80 = 1810 versus 100*80 = 8000, about 4.4x smaller
print("relative error:", np.linalg.norm(A - A_k) / np.linalg.norm(A))   # about 0.02
print(np.round(s[:7], 2))              # the seven largest singular values: five large, then a cliff

# %% [markdown]
# Pure random noise has no such structure and does not compress at all, and the claim is quick to check:

# %%
R = np.random.default_rng(0).standard_normal((100, 80))   # pure noise, no patterns to find
U, s, Vt = svd(R, full_matrices=False)
R_10 = U[:, :10] @ np.diag(s[:10]) @ Vt[:10, :]
print(round(np.linalg.norm(R - R_10) / np.linalg.norm(R), 3))   # 0.811
print(round(s[0], 2), round(s[-1], 2), round(np.sum(s[:10]**2) / np.sum(s**2), 3))   # 19.52 1.21 0.343

# %% [markdown]
# The next block runs the same reconstruction on the china.jpg photograph that ships with scikit-learn (a Creative Commons image from Flickr, credited in the library's image README), averaged over its three color channels into one 427-by-640 grayscale matrix.

# %%
from numpy.linalg import svd
from sklearn.datasets import load_sample_image

img = load_sample_image("china.jpg").mean(axis=2)   # 427 x 640 grayscale photograph
U, s, Vt = svd(img, full_matrices=False)
m, n = img.shape
for k in [5, 20, 50]:
    img_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    err = np.linalg.norm(img - img_k) / np.linalg.norm(img)
    stored = k * (m + n + 1) / (m * n)              # fraction of the original numbers kept
    print(k, round(float(err), 3), round(float(stored), 3))

# %% [markdown]
# A rotation can do neither. The switch to a rotation also has a history, with a different stated motive. Mikolov, Le, and Sutskever (2013) translated words with a general linear map between two languages' word vectors, learned from a small amount of bilingual data, and Xing et al. (2015) obtained better English-to-Spanish word translations by normalizing every word vector to unit length and constraining the map to be orthogonal. Their argument was consistency: the vectors were trained with inner products and compared with cosine similarity, while the map was fit by mean squared error, so they made every vector unit length, which turns the inner product into the cosine, kept the map orthogonal so that mapped vectors stay unit length, and fit it by cosine similarity. Example 2-2 confirms the two-point calculation and then runs a harder test: six noisy matched pairs in five dimensions, fitted once by the rotation and once by an unconstrained least-squares map, then scored on 500 points that neither fit has seen.

# %%
import numpy as np

# the two-point example from the text: B is A rotated 90 degrees counterclockwise
A = np.array([[1., 1.],
              [1., -1.]])
B = np.array([[-1., 1.],
              [1., 1.]])
M = A.T @ B                          # the alignment matrix
print(M)                             # [[ 0.  2.]
                                     #  [-2.  0.]]  matches the hand calculation
U, s, Vt = np.linalg.svd(M)
print(s)                             # [2. 2.]  both singular values are 2, the stretch we discard
R = U @ Vt                           # the Procrustes rotation
print(R)                             # [[ 0.  1.]
                                     #  [-1.  0.]]  the 90 degree rotation we secretly applied
print(np.linalg.norm(A @ R - B))     # 0.0  the alignment is exact

# a harder test: a tiny dictionary of 6 matched pairs in 5 dimensions, plus noise
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))          # the true map we hope to recover
A_dict = rng.standard_normal((6, 5))                      # the 6 anchor points we get to see
B_dict = A_dict @ Q + 0.1 * rng.standard_normal((6, 5))   # their noisy matched partners

U, s, Vt = np.linalg.svd(A_dict.T @ B_dict)
R = U @ Vt                           # the rotation, fitted on the 6 pairs
X = np.linalg.pinv(A_dict) @ B_dict  # the unconstrained least-squares map, same 6 pairs

A_test = rng.standard_normal((500, 5))                    # 500 points neither map has seen
B_test = A_test @ Q                                       # where those points truly belong
print(np.linalg.norm(A_test @ R - B_test) / np.linalg.norm(B_test))  # about 0.123, generalizes
print(np.linalg.norm(A_test @ X - B_test) / np.linalg.norm(B_test))  # about 1.341, fit the noise
print(np.linalg.norm(R.T @ R - np.eye(5)))   # about 1.6e-15, R is orthogonal to machine precision
print(np.linalg.norm(X.T @ X - np.eye(5)))   # about 11, X shears and stretches freely

# %% [markdown]
# The next block moves the partners of the six-pair dictionary 3 units along every axis and fits the rotation both ways.

# %%
# shifted clouds: move the partners 3 units along every axis, then fit the rotation two ways
B_moved = B_dict + 3.0
U, s, Vt = np.linalg.svd(A_dict.T @ B_moved)
R_raw = U @ Vt                                            # fitted without centering
a_bar, b_bar = A_dict.mean(axis=0), B_moved.mean(axis=0)  # the two centroids
U, s, Vt = np.linalg.svd((A_dict - a_bar).T @ (B_moved - b_bar))
R_cen = U @ Vt                                            # fitted to the centered clouds
target = B_test + 3.0                                     # where the 500 unseen points now belong
scale = np.linalg.norm(B_test)                            # the same yardstick as the unshifted six-pair errors
print(np.linalg.norm(A_test @ R_raw - target) / scale)                    # about 3.154, far off
print(np.linalg.norm((A_test - a_bar) @ R_cen + b_bar - target) / scale)  # about 0.108, aligned again

# %% [markdown]
# ## 2.1.10 The pseudoinverse, the condition number, and why training blows up

# %% [markdown]
# The math you just learned is both the diagnosis and the cure.

# %%
A = np.array([[1., 1.], [1., 1.0001]])    # nearly singular, the rows are almost identical
U, s, Vt = np.linalg.svd(A)
print(np.round(s, 5))                     # [2.00005e+00 5.00000e-05]  the two stretch factors
print(np.round(1 / s, 1))                 # [5.00000e-01 2.00005e+04]  the diagonal of Sigma+
A_plus = Vt.T @ np.diag(1 / s) @ U.T      # A+ = V Sigma+ U^T, assembled by hand
print(np.round(A_plus, 1))                # [[ 10001. -10000.]
                                          #  [-10000.  10000.]]  np.linalg.pinv(A) agrees
print(round(np.linalg.cond(A)))           # 40002 = s[0] / s[1] before rounding, ill conditioned

x  = A_plus @ np.array([2., 2.0001])      # solve A x = b for b = [2, 2.0001]
x2 = A_plus @ np.array([2.0001, 2.])      # ...and again after a tiny nudge to b
print(np.round(x, 4), np.round(x2, 4))    # [1. 1.] [ 3.0001 -1.    ]  the answer jumped

X = np.array([[1., 0.], [0., 1.], [1., 1.]])   # three equations, two unknowns
y = np.array([1., 1., 0.])
w = np.linalg.pinv(X) @ y                      # the least-squares solution
print(np.round(w, 4), np.round(X.T @ (y - X @ w), 12))   # [0.3333 0.3333] [0. 0.]

# %% [markdown]
# ## 2.1.11 Projections and orthogonality, the geometry of best fit

# %% [markdown]
# For now, hold the image that the best fit is the nearest point, which is a projection.

# %%
b = np.array([2., 3.]); a = np.array([1., 0.])
proj = (a @ b) / (a @ a) * a
print(proj)                 # [2. 0.]
print((b - proj) @ a)       # 0.0, the residual is orthogonal to a, as promised

# %% [markdown]
# ## 2.1.12 How the factorizations are actually computed

# %% [markdown]
# a permutation matrix (the row swaps that keep the arithmetic stable), a lower-triangular , and an upper-triangular . Example 2-3 factors the matrix that Example 2-1 eliminated in 2.1.4 and solves the same system with the factors.

# %%
from scipy.linalg import lu, solve_triangular
A = np.array([[4., 3., 2.], [2., 1., 1.], [6., 5., 4.]])
P, L, U = lu(A)                                  # A = P L U
b = np.array([1., 2., 3.])
y = solve_triangular(L, P.T @ b, lower=True)     # forward-substitute  L y = P^T b
x = solve_triangular(U, y, lower=False)          # back-substitute     U x = y
print(np.abs(P @ L @ U - A).max())               # 0.0  the factorization reconstructs A exactly
print(x)                                         # [ 1. -3.  3.]
print(np.abs(A @ x - b).max())                   # 1.7763568394002505e-15  zero at machine precision

# %% [markdown]
# Example 2-4 factors the normal-equation matrix of the three-equation system from 2.1.10, solves that system with two triangular solves, and checks two promises made earlier: the semidefinite failure of 2.1.8 and the `Ridge` solver choice.

# %%
from sklearn.linear_model import Ridge

X = np.array([[1., 0.], [0., 1.], [1., 1.]])     # the tall system of 2.1.10
y = np.array([1., 1., 0.])
G = X.T @ X                                      # the normal-equation matrix [[2, 1], [1, 2]]
L = np.linalg.cholesky(G)                        # G = L L^T with L lower triangular
print(np.round(L, 4))                            # [[1.4142 0.    ]
                                                 #  [0.7071 1.2247]]
print(np.allclose(L @ L.T, G))                   # True: the factors rebuild G
z = solve_triangular(L, X.T @ y, lower=True)     # forward-substitute  L z = X^T y
w = solve_triangular(L.T, z, lower=False)        # back-substitute     L^T w = z
print(np.round(w, 4))                            # [0.3333 0.3333]  the least-squares answer of 2.1.10

try:
    np.linalg.cholesky(np.array([[1., 1.], [1., 1.]]))   # semidefinite, not definite
except np.linalg.LinAlgError as err:
    print(err)                                   # Matrix is not positive definite
print(Ridge().fit(X, y).solver_)                 # cholesky

# %% [markdown]
# Orthonormal bases are built by the Gram-Schmidt process, which takes independent vectors and makes them mutually perpendicular and unit length, one at a time. Example 2-5 builds such a basis from three vectors and checks it against NumPy's QR routine.

# %%
def gram_schmidt(V):
    Q = []
    for v in V:
        w = v.astype(float).copy()
        for q in Q:
            w = w - (q @ v) * q                   # subtract v's projection onto each earlier basis vector
        w = w / np.linalg.norm(w)                 # normalize what remains to unit length
        Q.append(w)
    return np.array(Q)

V = np.array([[1., 1., 0.], [1., 0., 1.], [0., 1., 1.]])
Q = gram_schmidt(V)
print(np.round(Q @ Q.T, 6))          # the 3 by 3 identity: the basis is orthonormal
Qv, Rv = np.linalg.qr(V.T)           # the library's QR of the same three vectors, stored as columns
signs = np.sign(np.diag(Rv))         # Householder may flip a vector; Gram-Schmidt keeps R's diagonal positive
print(np.allclose(Qv * signs, Q.T))  # True: undo the flips and the library's basis is ours
Qm, R = np.linalg.qr(A)              # the same A that the LU decomposition factored
print(np.allclose(Qm @ R, A))        # True: A = QR reconstructs A

# %% [markdown]
# The simplest method, power iteration, just multiplies a vector by the matrix over and over. Example 2-6 runs it on a symmetric 2 by 2 matrix and checks the answer against NumPy.

# %%
S = np.array([[2., 1.], [1., 3.]])
def power_iteration(M, iters=200):
    v = np.ones(M.shape[0])
    for _ in range(iters):
        v = M @ v                                 # stretch along the dominant direction...
        v = v / np.linalg.norm(v)                 # ...and renormalize so it does not blow up
    lam = float(v @ M @ v)                         # Rayleigh quotient = the matched eigenvalue
    return lam, v

lam, v = power_iteration(S)
print(round(lam, 3))                              # 3.618  the dominant eigenvalue
vals, vecs = np.linalg.eigh(S)                    # eigh: the routine specialized for symmetric matrices
print(np.round(vals, 3))                          # [1.382 3.618]  numpy agrees, both eigenvalues real
print(np.round(vecs.T @ vecs, 6))                 # the 2 by 2 identity: the eigenvectors are orthonormal

# %% [markdown]
# Example 2-7 runs both on the same matrix S: deflation, then the unshifted QR algorithm.

# %%
lam2, v2 = power_iteration(S - lam * np.outer(v, v))   # deflate: remove the found direction, repeat
print(round(lam2, 3))                                   # 1.382  the second eigenvalue

Ak = S.copy()
for _ in range(50):                                     # the unshifted QR algorithm
    Qk, Rk = np.linalg.qr(Ak)                           # factor the current matrix...
    Ak = Rk @ Qk                                        # ...and multiply the factors in reverse order
print(np.round(Ak, 3))                                  # [[ 3.618 -0.   ]
                                                        #  [ 0.     1.382]]  both eigenvalues on the diagonal

# %% [markdown]
# ## 2.1.13 Coordinates and change of basis

# %% [markdown]
# The block repeats the example in NumPy, finding the coordinates with np.linalg.solve, which solves Q c = x without forming the inverse.

# %%
A = np.array([[2., 1.],
              [0., 1.]])
Q = np.array([[1., 1.],
              [0., -1.]])                  # columns: the eigenvectors [1, 0] and [1, -1] of A
x = np.array([3., -2.])

c = np.linalg.solve(Q, x)                  # coordinates of x: solve Q c = x
print(c, Q @ c)                            # the coordinates, then the vector they rebuild
print(Q.T @ x)                             # plain dot products with the basis vectors
print(np.linalg.inv(Q) @ A @ Q)            # A rewritten in eigenvector coordinates
print(round(float(np.linalg.norm(x)), 4), round(float(np.linalg.norm(c)), 4))   # lengths before and after

# %% [markdown]
# PCA is a change of basis to an orthonormal basis that the data chooses: the scores are the coordinates of each standardized wine along the principal directions.

# %%
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

X = load_wine().data                                   # 178 wines, 13 chemical measurements
Z = (X - X.mean(axis=0)) / X.std(axis=0)               # standardize each column
U, s, Vt = np.linalg.svd(Z, full_matrices=False)       # rows of Vt: orthonormal principal directions
scores = Z @ Vt.T                                      # row i holds the coordinates V^T z_i

sk = PCA().fit(Z).transform(Z)                         # fit finds the directions, transform the scores
signs = np.sign((scores * sk).sum(axis=0))             # each direction is defined only up to sign
print(scores.shape, np.allclose(scores, sk * signs))
print(np.allclose(np.linalg.norm(scores, axis=1), np.linalg.norm(Z, axis=1)))   # lengths survive
print(np.round(scores[0, :3], 4))                      # the first wine's first three coordinates

# %% [markdown]
# ## 2.2.1 Derivatives and the idea of a slope

# %% [markdown]
# A loss that jumps, like the raw count of classification mistakes, which leaps by a whole unit the instant a prediction flips, gives gradient descent a slope of zero everywhere else and no warning at the cliff, which is why 2.5 replaces such

# %%
import numpy as np

# 1) A limit is where the outputs head. The slope fraction of f(x) = x^2
#    at x = 3 is g(h) = ((3+h)^2 - 9)/h, undefined at h = 0 (it is 0/0),
#    so watch it from both sides instead:
def g(h):
    return ((3 + h)**2 - 9) / h

print([round(g(h), 6) for h in [0.1, 0.01, 0.001]])     # [6.1, 6.01, 6.001]
print([round(g(h), 6) for h in [-0.1, -0.01, -0.001]])  # [5.9, 5.99, 5.999]

# 2) One-sided limits can disagree. For ReLU(x) = max(0, x), rise over run
#    at x = 0 is ReLU(h)/h:
def relu(x):
    return max(0.0, x)

print([relu(h) / h for h in [0.1, 0.001]])    # [1.0, 1.0]   right-hand limit 1
print([relu(h) / h for h in [-0.1, -0.001]])  # [-0.0, -0.0] left-hand limit 0

# 3) Limits at infinity: the sigmoid flattens toward its two ceilings.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print(sigmoid(np.array([2.0, 5.0, 10.0])))   # [0.88079708 0.99330715 0.9999546 ]
print(sigmoid(np.array([-5.0, -10.0])))      # [6.69285092e-03 4.53978687e-05]

# %% [markdown]
# You can confirm the slope at without any calculus, by measuring rise over run with a tiny step, which is exactly what the definition does before the limit.

# %%
import numpy as np

def f(x):
    return x**2

h = 1e-6
x = 3.0
numeric_slope = (f(x + h) - f(x)) / h    # the definition, with a small but nonzero h
print(numeric_slope)                      # 6.000001000927568  ~ 6
print(2 * x)                              # 6.0  the exact derivative 2x

# %% [markdown]
# Any other base drags a constant factor through every derivative, which is why the formulas of calculus are cleanest in base e.

# %%
import numpy as np

# the slope of b^x at x = 0, measured with a small step h, beside ln b
h = 0.001
for b in [2.0, np.e, 10.0]:
    print(round(float(b), 4), round(float((b ** h - 1) / h), 4), round(float(np.log(b)), 4))

# %% [markdown]
# ## 2.2.2 Partial derivatives and the gradient

# %% [markdown]
# The block checks the two limits and then tries all 360 whole-degree directions.

# %%
import numpy as np

def bowl(p):
    return p[0]**2 + p[1]**2                       # f(x, y) = x^2 + y^2 at the point p = (x, y)

p = np.array([3.0, 4.0])
g = np.array([6.0, 8.0])                           # the gradient at (3, 4)
h = 1e-6
for u in [np.array([0.6, 0.8]), np.array([1.0, 0.0])]:
    print(round(float((bowl(p + h * u) - bowl(p)) / h), 4), float(u @ g))   # 10.0 10.0, then 6.0 6.0

angles = np.radians(np.arange(360))                # one unit vector per degree around the circle
dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
D = dirs @ g                                       # the directional derivative along every one
print(round(float(D.max()), 3), np.round(dirs[np.argmax(D)], 3))   # 10.0 [0.602 0.799]  along the gradient
print(round(float(D.min()), 3))                                    # -10.0  along minus the gradient

# %% [markdown]
# The learning rate is the size of each step, and choosing it well is its own art, taken up in 2.6 and Chapter 11.

# %%
def grad(w):
    x, y = w
    return np.array([2 * x, 2 * y])      # gradient of x^2 + y^2

w = np.array([3.0, 4.0])
eta = 0.1
w_next = w - eta * grad(w)               # one step of gradient descent
print(grad(w))                            # [6. 8.]   the gradient at (3, 4)
print(w_next)                             # [2.4 3.2] one step downhill
for _ in range(100):                      # now take 100 steps in a row
    w = w - eta * grad(w)
print(w)                                  # [6.11110793e-10 8.14814391e-10]  nearly the origin

# %% [markdown]
# ## 2.2.3 The chain rule, the engine of backpropagation

# %% [markdown]
# Multiplying on the way back is the whole idea of backpropagation, just with vectors and many layers.

# %%
def y_of_x(x):
    u = 3 * x + 1
    return u**2

h = 1e-6
x = 1.0
numeric = (y_of_x(x + h) - y_of_x(x)) / h    # measured slope
analytic = 6 * (3 * x + 1)                     # chain-rule result 6(3x+1)
print(numeric)        # 24.000008995272992, about 24 + 9h
print(analytic)       # 24.0

# %% [markdown]
# The by-hand work is the version we already did; SymPy automates it.

# %%
import sympy as sp

x = sp.symbols('x')                    # x is now an algebraic symbol, not a number
y = (3 * x + 1)**2                     # the same nested function y = (3x + 1)^2
dy = sp.diff(y, x)                     # differentiate the formula itself
print(dy)                              # 18*x + 6
print(sp.factor(dy))                   # 6*(3*x + 1), the chain-rule form
print(dy.subs(x, 1))                   # 24

# %% [markdown]
# the mathematical error falls as shrinks while the rounding error grows, and for centered differences in float64 the sweet spot sits near .

# %%
def f(x):
    return x**3                                   # true derivative at x = 1 is 3x^2 = 3

x = 1.0
for h in [1e-1, 1e-2, 1e-3]:
    forward  = (f(x + h) - f(x)) / h              # one-sided estimate, error about 3h
    centered = (f(x + h) - f(x - h)) / (2 * h)    # two-sided estimate, error exactly h^2 here
    print(f"h={h:g}  forward error={forward - 3:.2e}  centered error={centered - 3:.2e}")
# h=0.1  forward error=3.10e-01  centered error=1.00e-02
# h=0.01  forward error=3.01e-02  centered error=1.00e-04
# h=0.001  forward error=3.00e-03  centered error=1.00e-06

for h in [1e-5, 1e-8, 1e-13]:                     # now push h toward the rounding floor
    centered = (f(x + h) - f(x - h)) / (2 * h)
    print(f"h={h:g}  centered error={centered - 3:.2e}")
# h=1e-05  centered error=9.74e-11    near the float64 sweet spot
# h=1e-08  centered error=-7.13e-09   rounding noise now dominates
# h=1e-13  centered error=-7.32e-04   cancellation has destroyed the estimate

# %% [markdown]
# The next block runs the check on the tiny network from this subsection, where the chain rule said , and then deliberately tests a buggy formula, , the kind of dropped factor a tired human produces.

# %%
def y_of_w(w, x=2.0):
    return (w * x)**2                     # the tiny network of this subsection: u = wx, then y = u^2

w = 1.5
analytic = 2 * w * 2.0**2                 # chain rule: dy/dw = 2wx^2 = 2(1.5)(4) = 12
h = 1e-5
numeric = (y_of_w(w + h) - y_of_w(w - h)) / (2 * h)     # centered difference in w
rel_err = abs(numeric - analytic) / (abs(numeric) + abs(analytic))
print(analytic, numeric)                  # 12.0 12.000000000078613
print(f"relative error = {rel_err:.1e}")  # relative error = 3.3e-12   the algebra is right

wrong = 2 * w * 2.0                       # the buggy "derivative" 2wx that dropped a factor of x
rel_err_wrong = abs(numeric - wrong) / (abs(numeric) + abs(wrong))
print(f"wrong-formula relative error = {rel_err_wrong:.1e}")   # 3.3e-01   the bug is exposed

# %% [markdown]
# The block checks the formula with both tools of this subsection, a centered difference at the two points and SymPy for every input.

# %%
import numpy as np
import sympy as sp

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

h = 1e-5
for x0 in [0.0, 2.0]:
    numeric = (sigmoid(x0 + h) - sigmoid(x0 - h)) / (2 * h)   # centered difference
    formula = sigmoid(x0) * (1 - sigmoid(x0))                  # the derived rule
    print(x0, round(float(numeric), 6), round(float(formula), 6))   # 0.0 0.25 0.25, then 2.0 0.104994 0.104994

z = sp.symbols('z')
s = 1 / (1 + sp.exp(-z))
print(sp.simplify(sp.diff(s, z) - s * (1 - s)))    # 0  the identity holds for every z

# %% [markdown]
# ## 2.2.4 The Jacobian and the Hessian

# %% [markdown]
# Their product is , the same answer, and at both give 6.

# %%
x = 1.0
Jg = np.array([[2 * x], [2.0]])            # Jacobian of g(x) = (x^2, 2x), shape 2 by 1
u, v = x**2, 2 * x
Jf = np.array([[v, u]])                    # Jacobian of f(u, v) = u v, shape 1 by 2
print(Jf @ Jg)                             # [[6.]]  matches d/dx of 2x^3 = 6x^2 at x = 1

# %% [markdown]
# SymPy builds the same matrix from the formula, keeping the promise made in 2.2.3, and the eigvalsh routine of 2.1.8 reads off its eigenvalues.

# %%
import sympy as sp

x, y = sp.symbols('x y')
f = x**2 + x*y + y**2
H = sp.hessian(f, (x, y))                              # every second partial derivative of f
print(H)                                               # Matrix([[2, 1], [1, 2]])
print(np.linalg.eigvalsh(np.array(H, dtype=float)))    # [1. 3.]  both positive: a bowl

g = x**2 + 3*x*y + y**2                                # the same diagonal, a stronger cross term
Hg = sp.hessian(g, (x, y))
print(Hg)                                              # Matrix([[2, 3], [3, 2]])
print(np.linalg.eigvalsh(np.array(Hg, dtype=float)))   # [-1.  5.]  mixed signs: not a bowl

# %% [markdown]
# ## 2.2.5 Maxima, minima, and saddle points

# %% [markdown]
# they carry speed through these flat regions instead of stalling.

# %%
H = np.array([[2., 0.], [0., -2.]])      # Hessian of x^2 - y^2 at the origin
eigvals = np.linalg.eigvalsh(H)          # eigvalsh of 2.1.8: H is symmetric, so real eigenvalues, ascending
print(eigvals)        # [-2.  2.]  mixed signs
print("saddle" if eigvals.min() < 0 < eigvals.max() else "not a saddle")

# %% [markdown]
# ## 2.2.6 Taylor series, why optimizers approximate

# %% [markdown]
# Gradient descent on the bowl multiplies its iterate by the constant factor each step, where is the learning rate, so the iterates form a geometric sequence, and they die out to the minimum exactly when .

# %%
import numpy as np

# add the geometric terms directly and compare against the formula a / (1 - r)
r = 0.9
S = np.cumsum(r ** np.arange(51))        # partial sums S_0 through S_50
print(S[10], S[50])                       # 6.861894039100001 9.953616023134117
print(1 / (1 - r))                        # 10.000000000000002   the limit
print(1 / (1 - 0.99))                     # 99.99999999999991    the r = 0.99 limit

# terms -> 0 is not enough: harmonic partial sums pass any bound
H = np.cumsum(1.0 / np.arange(1, 1_000_001))
print(H[9], H[999], H[999_999])           # 2.9289682539682538 7.485470860550343 14.392726722864989

# %% [markdown]
# near the current point, a low-order polynomial is good enough to choose a step.

# %%
import numpy as np

h = 0.1
first_order  = 1 + h                      # value + slope
second_order = 1 + h + 0.5 * h**2         # value + slope + curvature
true_value   = np.exp(h)
print(first_order, second_order, true_value)   # 1.1 1.105 1.10517091807...

# %% [markdown]
# The block adds the terms of the series for e to the power 0.1 one at a time.

# %%
from math import factorial
import numpy as np

h = 0.1
partial = 0.0
for n in range(5):
    partial += h**n / factorial(n)        # add the term h^n / n!; every derivative of e^x at 0 is 1
    print(n, f"{partial:.7f}", f"{np.exp(h) - partial:.2e}")   # the partial sum, then the error left
print(f"{h**4 / 24:.2e} {np.exp(h) * h**4 / 24:.2e}")          # 4.17e-06 4.60e-06  the bounds after h^3

# %% [markdown]
# ## 2.2.7 Integrals, the other half of calculus

# %% [markdown]
# halving the slice width roughly halves a one-sided sum's error but cuts the trapezoid rule's error by a factor of four.

# %%
import numpy as np

def f(x):
    return x**2                          # the curve whose area we want on [0, 1]

a, b = 0.0, 1.0
for n in [4, 100, 10_000]:               # slice the interval ever more finely
    dx = (b - a) / n                     # width of each slice
    edges = a + dx * np.arange(n + 1)    # the n+1 slice edges
    left  = np.sum(f(edges[:-1]) * dx)   # rectangles read their height at the left edge
    right = np.sum(f(edges[1:]) * dx)    # rectangles read their height at the right edge
    trap  = 0.5 * (left + right)         # trapezoid rule: average the two
    print(n, round(left, 5), round(right, 5), round(trap, 5))
# 4 0.21875 0.46875 0.34375
# 100 0.32835 0.33835 0.33335
# 10000 0.33328 0.33338 0.33333

print(b**3 / 3 - a**3 / 3)               # 0.3333333333333333  the FTC answer F(1) - F(0)

# %% [markdown]
# at it is off by about while the left sum is still off by about , roughly three hundred times more.

# %%
import numpy as np
from scipy.stats import norm

x = np.linspace(-6.0, 6.0, 100_001)                # fine grid across the bell curve
pdf = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)       # standard Gaussian density
print(round(float(np.trapezoid(pdf, x)), 6))       # 1.0     total area: a density must give 1
inside = (x >= -2) & (x <= 2)
print(round(float(np.trapezoid(pdf[inside], x[inside])), 4))  # 0.9545  area within two sigma
print(round(float(norm.cdf(2) - norm.cdf(-2)), 4)) # 0.9545  scipy's special-function answer

# %% [markdown]
# ## 2.2.8 Matrix calculus: gradients with respect to a whole vector

# %% [markdown]
# For vectors, the relative error of 2.2.3 uses lengths in place of absolute values.

# %%
import numpy as np
from sklearn.datasets import load_diabetes, load_wine
from sklearn.linear_model import LinearRegression

def numeric_grad(f, w, h=1e-5):
    """Centered differences of 2.2.3, one coordinate at a time."""
    g = np.zeros_like(w)
    for j in range(len(w)):
        e = np.zeros_like(w)
        e[j] = h
        g[j] = (f(w + e) - f(w - e)) / (2 * h)
    return g

def rel_err(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))

A = np.array([[1., 2.], [0., 3.]])                 # a matrix that is not symmetric
w = np.array([1., 1.])
print(numeric_grad(lambda v: v @ A @ v, w), (A + A.T) @ w, 2 * A @ w)   # [4. 8.] [4. 8.] [6. 6.]

X, y = load_diabetes(return_X_y=True)
X = np.column_stack([np.ones(len(y)), X])          # intercept column first, 11 weights in all
n = len(y)
mse = lambda v: np.mean((X @ v - y) ** 2)
w = np.random.default_rng(0).normal(size=11)       # any point will do for a gradient check
analytic = 2 / n * X.T @ (X @ w - y)               # the matrix calculus answer, all 11 at once
print(f"{rel_err(analytic, numeric_grad(mse, w)):.1e}")      # 7.6e-10

w_hat = np.linalg.solve(X.T @ X, X.T @ y)          # the zero-gradient point: X^T X w = X^T y
fit = LinearRegression().fit(X[:, 1:], y)
print(np.allclose(w_hat, np.concatenate([[fit.intercept_], fit.coef_])),
      f"{np.abs(2 / n * X.T @ (X @ w_hat - y)).max():.1e}")  # True 7.6e-14

C = np.cov(load_wine().data, rowvar=False)          # 13-by-13 covariance of the wine features
v = np.random.default_rng(1).normal(size=13)
print(np.allclose(C, C.T), f"{rel_err(2 * C @ v, numeric_grad(lambda u: u @ C @ u, v)):.1e}")   # True 4.6e-12

# %% [markdown]
# ## 2.2.9 Integration by substitution and by parts

# %%
import numpy as np
import sympy as sp

# positive symbols let SymPy see that e^(-lambda x) dies out as x grows
x, lam, sigma = sp.symbols('x lambda sigma', positive=True)
t, mu = sp.symbols('t mu', real=True)

density = lam * sp.exp(-lam * x)                          # the Exponential density of Table 2-4
mean = sp.integrate(x * density, (x, 0, sp.oo))           # by parts once on paper
second = sp.integrate(x**2 * density, (x, 0, sp.oo))      # by parts twice
print(mean, second, sp.simplify(second - mean**2))       # mean, E[X^2], variance

bell = sp.exp(-(t - mu)**2 / (2 * sigma**2))              # the Gaussian curve without its constant
print(sp.integrate(bell, (t, -sp.oo, sp.oo)))             # its area, by the substitution z = (t - mu)/sigma

# the same numbers from the trapezoid rule of 2.2.7, at lambda = 2 and sigma = 5
grid = np.linspace(0.0, 40.0, 400_001)                    # the mass beyond 40 is e^-80, nothing
p = 2.0 * np.exp(-2.0 * grid)
m1 = np.trapezoid(grid * p, grid)
m2 = np.trapezoid(grid**2 * p, grid)
print(round(float(m1), 6), round(float(m2 - m1**2), 6))  # mean and variance at lambda = 2
heights = np.linspace(102.0, 222.0, 120_001)              # 12 standard deviations either side of 162
print(round(float(np.trapezoid(np.exp(-(heights - 162)**2 / 50), heights)), 4),
      round(float(5 * np.sqrt(2 * np.pi)), 4))           # numerical area beside sigma * sqrt(2 pi)

# a product cannot be integrated factor by factor
u = np.linspace(0.0, 1.0, 100_001)
right = np.trapezoid(u * 2 * np.exp(-2 * u), u)
wrong = np.trapezoid(u, u) * np.trapezoid(2 * np.exp(-2 * u), u)
print(round(float(right), 4), round(float(wrong), 4))    # the true integral, then the shortcut

# %% [markdown]
# ## 2.3.1 Random variables, sample spaces, and events

# %% [markdown]
# This pattern, one minus the probability of none, returns in 2.4.10, where the chance that at least one of hypothesis tests fires falsely is , so it is worth owning now.

# %%
omega = {1, 2, 3, 4, 5, 6}            # the sample space of one die
A = {2, 4, 6}                          # event "even"
B = {4, 5, 6}                          # event "greater than 3"

def P(event):                          # fair die: probability is just counting
    return len(event) / len(omega)

print(A | B, P(A | B))                 # {2, 4, 5, 6} 0.6666666666666666
print(A & B, P(A & B))                 # {4, 6} 0.3333333333333333
print(omega - A, P(omega - A))         # {1, 3, 5} 0.5   (the complement is omega minus A)
print(P(A) + P(B) - P(A & B))          # 0.6666666666666667  the addition rule agrees
print(omega - (A | B) == (omega - A) & (omega - B))   # True  De Morgan checks out
print(1 - (5/6) ** 4)                  # 0.5177469135802468  at least one six in four rolls

# %% [markdown]
# For a continuous variable the probability of any exact value is zero, and probabilities are areas under the density curve over a range, a fact we will use the moment we meet the Gaussian.

# %%
import numpy as np

rolls = np.random.default_rng(0).integers(1, 7, size=100_000)   # 100k fair dice rolls
print((rolls % 2 == 0).mean())     # 0.49758  the empirical P(even), near 0.5
print((rolls == 6).mean())         # 0.16536  the empirical P(roll a 6), near 1/6, about 0.1667

# %% [markdown]
# The block builds the die's CDF as a running sum and checks both continuous values.

# %%
import numpy as np
from scipy.stats import norm

pmf = np.full(6, 1 / 6)                     # the fair die: faces 1 to 6, each with probability 1/6
F = np.cumsum(pmf)                          # its CDF at x = 1, 2, ..., 6, the running sum of the PMF
print(np.round(F, 4))                       # climbs by 1/6 per face, from 0.1667 to 1.0
print(round(float(F[3] - F[1]), 4))         # 0.3333  P(2 < X <= 4) = F(4) - F(2), the rolls 3 and 4
print(round(float(1 - np.exp(-2 * 0.5)), 4),          # 0.6321  the Exponential CDF with rate 2, at 0.5
      round(float(norm.cdf(2) - norm.cdf(-2)), 4))    # 0.9545  the standard Gaussian's F(2) - F(-2)

# %% [markdown]
# ## 2.3.2 Conditional probability and independence

# %% [markdown]
# represent each event as a boolean mask over the outcomes, and every probability is a mean of a mask.

# %%
rolls = np.arange(1, 7)          # the six equally likely outcomes of one die
A = (rolls == 6)                 # event A: the roll is a six
B = (rolls % 2 == 0)             # event B: the roll is even
print(round(A.mean(), 3), round(B.mean(), 3))       # 0.167 0.5    P(A) and P(B)
print(round((A & B).mean(), 3))                     # 0.167        P(A and B)
print(round((A & B).mean() / B.mean(), 3))          # 0.333        P(A | B), the formula verbatim

# %% [markdown]
# ## 2.3.3 Bayes' theorem

# %% [markdown]
# This base-rate effect is exactly why a classifier that scores 99 percent accuracy on a rare event can still be nearly useless, a theme we return to in Chapter 4.

# %%
p_d = 0.01          # prior: prevalence
p_pos_given_d = 0.99    # sensitivity
p_pos_given_not = 0.05  # false-positive rate

p_pos = p_pos_given_d * p_d + p_pos_given_not * (1 - p_d)
posterior = p_pos_given_d * p_d / p_pos
print(round(posterior, 3))     # 0.167

# %% [markdown]
# ## 2.3.4 The distributions you will meet everywhere

# %% [markdown]
# if events arrive as a Poisson process, the waiting time between them is Exponential with mean , which is why it models time until the next click, the next failure, or the next arrival, and underlies the survival analysis of Chapter 9.

# %%
from scipy.stats import binom, norm, poisson

print(round(binom.pmf(7, 10, 0.5), 3))     # 0.117  P(7 heads in 10 fair flips)
print(round(norm.cdf(190, 170, 10) - norm.cdf(150, 170, 10), 3))  # 0.954 within 2 sigma
print(round(poisson.pmf(5, 3), 3))         # 0.101  P(5 sign-ups when the rate is 3)

# %% [markdown]
# The block prints the categorical losses, then evaluates the density formula term by term at the three points, beside SciPy's `multivariate_normal` and the independent-bells value.

# %%
import numpy as np
from scipy.stats import multivariate_normal, norm

p = np.array([0.7, 0.2, 0.1])          # a categorical distribution over K = 3 classes
print(np.round(-np.log2(p), 3))        # the loss -log2(p_k) when class k occurs, in bits

mu = np.array([0.0, 0.0])              # mean vector
Sigma = np.array([[1.0, 0.8],
                  [0.8, 1.0]])         # unit variances, covariance 0.8

def mvn_pdf(x):                        # the density formula, term by term
    dev = x - mu
    q = dev @ np.linalg.inv(Sigma) @ dev                                  # the quadratic form of 2.1.8
    const = (2 * np.pi) ** (len(x) / 2) * np.sqrt(np.linalg.det(Sigma))  # (2 pi)^(d/2) det(Sigma)^(1/2)
    return np.exp(-0.5 * q) / const

for x in [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([1.0, -1.0])]:
    print(x, round(float(mvn_pdf(x)), 4),                          # by the formula
          round(float(multivariate_normal(mu, Sigma).pdf(x)), 4),  # scipy's version
          round(float(np.prod(norm.pdf(x))), 4))                   # two independent bells

# %% [markdown]
# That is clustering, and this paragraph is its mathematical seed.

# %%
import numpy as np
from scipy.stats import norm

w  = np.array([0.5, 0.5])      # mixing weights, one per component, summing to 1
mu = np.array([162.0, 176.0])  # component means in cm
sd = np.array([5.0, 5.0])      # component standard deviations

def mix_pdf(x):                # p(x) = sum_k w_k N(x; mu_k, sd_k^2)
    return float(np.sum(w * norm.pdf(x, mu, sd)))

print(round(mix_pdf(162), 4))  # 0.0407  the hump over the first group
print(round(mix_pdf(169), 4))  # 0.0299  the valley at the mixture mean
print(round(mix_pdf(176), 4))  # 0.0407  the hump over the second group

# Bayes' theorem: which component did a 168 cm person come from?
post = w[0] * norm.pdf(168, mu[0], sd[0]) / mix_pdf(168)
print(round(post, 3))          # 0.636  P(component 1 | x = 168)

# sample the mixture the way the definition says: pick, then draw
rng = np.random.default_rng(0)
comp = rng.choice(2, size=200_000, p=w)        # step 1: pick a component by weight
x = rng.normal(mu[comp], sd[comp])             # step 2: draw from that component
print(round(float(x.mean()), 1), round(float(x.var()), 1))  # 169.0 74.1

# %% [markdown]
# Seaborn's `geyser` dataset, which the seaborn-data README traces to R's `faithful` data, records the duration in minutes of 272 eruptions of Old Faithful, the same data that 9.2.2 returns to for Bayesian regression.

# %%
import numpy as np
import seaborn as sns

d = sns.load_dataset("geyser")["duration"].to_numpy()          # 272 eruption durations in minutes
counts, edges = np.histogram(d, bins=np.arange(1.5, 6.0, 0.5))  # half-minute bins from 1.5 to 5.5
print(counts)                                                  # [51 41  5  7 30 73 61  4]
print(round(float(d.mean()), 2),                               # 3.49  the overall mean
      round(float(d[d < 3].mean()), 2),                        # 2.04  the short eruptions
      round(float(d[d >= 3].mean()), 2))                       # 4.29  the long eruptions

# %% [markdown]
# ## 2.3.5 Joint, marginal, and conditional distributions

# %% [markdown]
# All three views fall out of one 2-by-2 array, and the code is worth typing because the same three moves, sum an axis for a marginal, slice and rescale for a conditional, take an outer product for the independence baseline, recur throughout

# %%
P = np.array([[0.30, 0.05],
              [0.10, 0.55]])           # Table 2-5: rows rain yes/no, columns umbrella yes/no
p_rain = P.sum(axis=1)                 # marginal of rain: sum across each row
p_umbr = P.sum(axis=0)                 # marginal of umbrella: sum down each column
print(p_rain, p_umbr)                  # [0.35 0.65] [0.4 0.6]
print(round(P[0, 0] / p_rain[0], 3))   # 0.857  P(umbrella | rain): slice the rain row, rescale
print(np.round(np.outer(p_rain, p_umbr), 3))   # the joint IF independent:
                                               # [[0.14 0.21]
                                               #  [0.26 0.39]]

# %% [markdown]
# ## 2.3.6 Expectation, variance, covariance, and correlation

# %% [markdown]
# A variable against itself has , so its correlation is exactly , and against its own negation exactly ; nothing can escape , because the correlation is the cosine similarity of 2.1.6 applied to the two deviation vectors, and Cauchy-Schwarz c

# %%
x = np.array([1., 2., 3., 4., 5.])                 # hours studied
y = np.array([2., 4., 5., 4., 5.])                 # quiz scores
print(np.mean((x - x.mean()) * (y - y.mean())))    # 1.2     the covariance, matching the hand sum
print(round(float(np.corrcoef(x, y)[0, 1]), 4))    # 0.7746  the correlation
print(round(float(np.corrcoef(x, x)[0, 1]), 4),
      round(float(np.corrcoef(x, -x)[0, 1]), 4))   # 1.0 -1.0  the two ends of the range

# %% [markdown]
# The block checks all four dice numbers exactly, by listing every equally likely outcome instead of sampling.

# %%
faces = np.arange(1, 7)                         # one fair die, every face equally likely
pair_sums = faces[:, None] + faces[None, :]     # all 36 equally likely totals of two independent rolls
triple_means = (faces[:, None, None] + faces[None, :, None]
                + faces[None, None, :]) / 3     # all 216 equally likely averages of three rolls
print(round(float(faces.var()), 4))             # 2.9167   Var(X) = 35/12 for one roll
print(round(float(pair_sums.var()), 4))         # 5.8333   independent variances add
print(round(float((2 * faces).var()), 4))       # 11.6667  Var(2X) = 4 Var(X)
print(round(float(triple_means.var()), 4))      # 0.9722   (35/12) / 3, the variance of a mean of 3

# %% [markdown]
# The block checks each mean and variance of Table 2-4 numerically, from the definitions alone, at one parameter value per row.

# %%
import numpy as np
from scipy.stats import binom, poisson

def moments(values, probs):                    # E[X] and E[X^2] - (E[X])^2 for a discrete variable
    m = np.sum(values * probs)
    return round(float(m), 4), round(float(np.sum(values**2 * probs) - m**2), 4)

print(moments(np.array([0, 1]), np.array([0.5, 0.5])))   # Bernoulli(0.5): p and p(1-p)
k = np.arange(11)
print(moments(k, binom.pmf(k, 10, 0.5)))                  # Binomial(10, 0.5): np and np(1-p)
k = np.arange(101)                                        # terms past k = 100 are far too small to matter
print(moments(k, poisson.pmf(k, 3)))                      # Poisson(3): lambda and lambda

x = np.linspace(0.0, 1.0, 100_001)                        # Uniform(0, 1): density 1 on [0, 1]
m1, m2 = np.trapezoid(x, x), np.trapezoid(x**2, x)
print(round(float(m1), 4), round(float(m2 - m1**2), 4), round(1 / 12, 4))   # (a+b)/2, then (b-a)^2/12 twice

h = np.linspace(110.0, 230.0, 120_001)                    # Gaussian(170, 10^2), 6 sigma either side
pdf = np.exp(-(h - 170)**2 / 200) / (10 * np.sqrt(2 * np.pi))
m1 = np.trapezoid(h * pdf, h)
print(round(float(m1), 4), round(float(np.trapezoid((h - m1)**2 * pdf, h)), 4))   # mu, then sigma^2

# %% [markdown]
# The block computes each species' mean and variance of petal length, checks the tower rule against the overall mean and the variance split against the overall variance, and then draws 100,000 i.i.d. samples of 10 flowers each, with replacement, to compare the variance of the sample mean with sigma squared over 10.

# %%
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
y = iris.data[:, 2]                                   # petal length in cm, 150 flowers
s = iris.target                                       # species 0, 1, 2 (setosa, versicolor, virginica)
p = np.array([np.mean(s == k) for k in range(3)])     # P(species = k)
m = np.array([y[s == k].mean() for k in range(3)])    # E[Y | species = k], one slice mean each
v = np.array([y[s == k].var() for k in range(3)])     # Var(Y | species = k)
print(np.round(p, 4), np.round(m, 3))                 # [0.3333 0.3333 0.3333] [1.462 4.26  5.552]
print(round(float(p @ m), 4), round(float(y.mean()), 4))      # 3.758 3.758  the tower rule
within, between = p @ v, p @ (m - p @ m) ** 2
print(round(float(within), 4), round(float(between), 4),
      round(float(within + between), 4), round(float(y.var()), 4))   # 0.1815 2.914 3.0955 3.0955

rng = np.random.default_rng(0)
means = rng.choice(y, size=(100_000, 10)).mean(axis=1)   # 100,000 i.i.d. samples of 10 flowers each
print(round(float(means.var()), 4), round(float(y.var() / 10), 4))   # 0.3108 0.3096  against sigma^2 / n

# %% [markdown]
# ## 2.3.7 Maximum likelihood and maximum a posteriori

# %% [markdown]
# Assume Gaussian noise and least squares stops being a convenient recipe and becomes the maximum likelihood estimate, one formal reason squared error is the default regression loss. It is not the only reason: 5.1.3 shows that under the Gauss-Markov conditions least squares is the best linear unbiased estimator with no Gaussian assumption at all, and that Gaussian noise is what makes the p-values of 5.1.4 exact. The lab in 2.10 trains with this same loss.

# %%
# the coin: the log likelihood 7 ln p + 3 ln(1-p) peaks at the calculus answer
p = np.linspace(0.01, 0.99, 99)
print(round(float(p[np.argmax(7 * np.log(p) + 3 * np.log(1 - p))]), 2))   # 0.7

# MAP with the prior 6p(1-p): add its log, ln p + ln(1-p); the constant ln 6 moves no peak
q = np.linspace(0.001, 0.999, 999)                 # a finer grid, step 0.001
print(round(float(q[np.argmax(8 * np.log(q) + 4 * np.log(1 - q))]), 3))     # 0.667 = 8/12
print(round(float(q[np.argmax(71 * np.log(q) + 31 * np.log(1 - q))]), 3))   # 0.696 = 71/102, 70 heads in 100

# least squares as Gaussian MLE: the same slope wins both contests
rng = np.random.default_rng(0)
x = np.linspace(-1, 1, 20)
y = 2 * x + 0.1 * rng.normal(size=20)              # data from a true slope of 2, sigma = 0.1
w = np.linspace(0, 4, 401)                         # candidate slopes, step 0.01
sse = ((y - w[:, None] * x) ** 2).sum(axis=1)      # sum of squared errors per candidate
loglik = -len(x) * np.log(0.1 * np.sqrt(2 * np.pi)) - sse / (2 * 0.1 ** 2)
print(round(float(w[np.argmin(sse)]), 2),
      round(float(w[np.argmax(loglik)]), 2))       # 1.97 1.97  the same winner

# %% [markdown]
# ## 2.3.8 Markov chains

# %% [markdown]
# PageRank is nothing more than the stationary distribution of a Markov chain over web pages.

# %%
import numpy as np

P = np.array([[0.8, 0.2], [0.4, 0.6]])
pi = np.array([1.0, 0.0])               # start sunny
for _ in range(50):
    pi = pi @ P                          # step the chain forward
print(np.round(pi, 3))                   # [0.667 0.333]  the stationary distribution

vals, vecs = np.linalg.eig(P.T)          # eigenvectors of P transposed = left eigenvectors of P
v = vecs[:, np.argmax(vals.real)].real   # the one for eigenvalue 1, the largest; .real drops zero imaginary parts
print(np.round(v / v.sum(), 3))          # [0.667 0.333]  the same answer, scaled to sum to 1

# %% [markdown]
# The block runs the switching chain for four steps.

# %%
P_flip = np.array([[0., 1.], [1., 0.]])   # always switch states: irreducible, but with period 2
pi = np.array([1.0, 0.0])                 # start in state 0 for certain
for t in range(1, 5):
    pi = pi @ P_flip
    print(t, pi)                          # 1 [0. 1.], 2 [1. 0.], 3 [0. 1.], 4 [1. 0.]: it never settles
print(np.array([0.5, 0.5]) @ P_flip)      # [0.5 0.5]  the stationary distribution exists all the same

# %% [markdown]
# a joint distribution stored as a graph of local tables.

# %%
import numpy as np

P = np.array([[0.8, 0.2], [0.4, 0.6]])    # rows today, columns tomorrow; 0 = sunny, 1 = rainy
path = [0, 0, 1, 1]                        # the four-day sequence sunny, sunny, rainy, rainy
prob = 1.0                                 # P(day 0 is sunny) = 1: we start sunny for certain
for today, tomorrow in zip(path[:-1], path[1:]):
    prob *= P[today, tomorrow]             # multiply one matrix entry per hop
print(round(prob, 3))                      # 0.096  matches 0.8 * 0.2 * 0.6 by hand

total = sum(P[0, i] * P[i, j] * P[j, k]    # every four-day path that starts sunny
            for i in (0, 1) for j in (0, 1) for k in (0, 1))
print(round(total, 3))                     # 1.0  the eight path probabilities sum to 1

# %% [markdown]
# The same loop as the weather chain computes it.

# %%
import numpy as np

# the three-page web: rows are the source page, columns are the target page
A = np.array([[0, 1, 1],     # page A links to B and to C
              [0, 0, 1],     # page B links to C
              [1, 0, 0]])    # page C links back to A
print(A.sum(axis=1))                     # [2 1 1]  the out-degrees, one row sum per page
print(np.linalg.matrix_power(A, 2)[0])   # [1 0 1]  two-click routes leaving page A

P = A / A.sum(axis=1, keepdims=True)     # divide each row by its degree: click probabilities
pi = np.array([1.0, 0.0, 0.0])           # drop the surfer on page A
for _ in range(50):
    pi = pi @ P                          # one click, the same loop as the weather chain
print(np.round(pi, 3))                   # [0.4 0.2 0.4]  the ranks: A and C tie, B trails

# %% [markdown]
# ## 2.3.9 Stochastic processes: random walks, Brownian motion, and Itô's lemma

# %% [markdown]
# A simulation can build the process the way the definition does, laying exponential gaps end to end, and then count what lands in each hour.

# %%
import numpy as np
rng = np.random.default_rng(0)
gaps = rng.exponential(scale=1/3, size=300_000)   # waiting times in hours; scale is the mean 1/lambda
times = np.cumsum(gaps)                           # arrival times: the gaps laid end to end
hours = int(times[-1])                            # whole hours covered by the arrivals
counts, _ = np.histogram(times, bins=np.arange(hours + 1))   # arrivals in each one-hour window
print(round(float(gaps.mean()), 4))               # the mean gap, near 1/3 hour
print(round(float(counts.mean()), 3), round(float(counts.var()), 3))   # mean and variance of the counts
print(round(float(np.mean(counts == 0)), 4))      # share of hours with no arrival at all

# %% [markdown]
# Simulating many paths confirms it.

# %%
import numpy as np
rng = np.random.default_rng(0)
n_paths, n_steps, dt = 20_000, 500, 1/500          # simulate up to time t=1 in 500 small steps
incr = rng.normal(0, np.sqrt(dt), size=(n_paths, n_steps))   # each increment ~ N(0, dt)
W = np.cumsum(incr, axis=1)                          # the path is the running sum of increments
print(round(float(W[:, 249].var()), 3))              # 0.492  -> Var(W_t) = t at t = 0.5
print(round(float(W[:, -1].var()), 3))               # 0.991  -> Var(W_t) = t at t = 1.0
print(round(float(np.sum(incr[0] ** 2)), 3))         # 1.028  -> sum of squared increments ~ t

# %% [markdown]
# The block runs 100,000 walks of 20 steps for each value of p and, for every moment a walk stands at -2, at 0, or at 2, averages the step that comes next.

# %%
import numpy as np
rng = np.random.default_rng(1)
for p in (0.5, 0.55):                                        # a fair walk, then one tilted upward
    steps = np.where(rng.random((100_000, 20)) < p, 1, -1)   # 100,000 walks of 20 steps: +1 with probability p
    X = np.cumsum(steps, axis=1)                             # positions after steps 1 to 20
    now, move = X[:, :-1], steps[:, 1:]                      # every position, and the step that follows it
    # average next move over all the moments a walk stood at -2, at 0, and at 2
    print(p, [round(float(move[now == x].mean()), 3) for x in (-2, 0, 2)])

# %% [markdown]
# ## 2.3.10 Transforming a random variable: the change-of-variables formula

# %% [markdown]
# The block measures the skewness before and after the log, fits a Gaussian to log(area), carries that density back to the area scale with and without the Jacobian factor 1/x, integrates both curves with the trapezoid rule, and compares fitted and observed probabilities for five ranges of area.

# %%
import numpy as np
from scipy.stats import norm, skew
from sklearn.datasets import load_breast_cancer

print(round(float(norm.pdf(169, loc=162, scale=5)), 4),
      round(float(norm.pdf((169 - 162) / 5) / 5), 4))      # 0.0299 0.0299  Y = 162 + 5Z

data = load_breast_cancer()
area = data.data[:, list(data.feature_names).index("mean area")]   # one tumor per row
print(len(area), round(float(skew(area)), 2), round(float(skew(np.log(area))), 2))   # 569 1.64 0.28

mu, sigma = np.log(area).mean(), np.log(area).std()    # a Gaussian fitted to log(area)
x = np.linspace(1, 20_000, 400_001)                    # a fine grid on the area scale
with_jac = norm.pdf(np.log(x), mu, sigma) / x          # carried back with the factor 1/x
without = norm.pdf(np.log(x), mu, sigma)               # the same curve with the factor dropped
print(round(float(np.trapezoid(with_jac, x)), 4), round(float(np.trapezoid(without, x)), 1))   # 1.0 651.8

edges = [0, 400, 600, 800, 1200, np.inf]
observed = [np.mean((area > a) & (area <= b)) for a, b in zip(edges[:-1], edges[1:])]
cdf = norm.cdf(np.log(np.array(edges[1:-1])), mu, sigma)   # fitted CDF at 400, 600, 800, 1200
model = np.diff(cdf, prepend=0, append=1)              # probabilities of the five ranges
print(np.round(observed, 3))                           # [0.207 0.381 0.17  0.146 0.095]
print(np.round(model, 3))                              # [0.221 0.307 0.219 0.187 0.066]

# %% [markdown]
# ## 2.4.1 Descriptive statistics

# %% [markdown]
# The reason the divisor is and not has a name, degrees of freedom, and 2.4.4 explains the counting behind it.

# %%
import statistics
import numpy as np
import pandas as pd

salaries = np.array([40, 45, 50, 55, 60])
print(np.mean(salaries))                 # 50.0
print(np.median(salaries))               # 50.0
print(np.std(salaries))                  # 7.0710678118654755  (population, divides by n; about 7.07)
print(np.percentile(salaries, 25))       # 45.0   the 25th percentile

outlier = np.array([40, 45, 50, 55, 1000])
print(np.mean(outlier), np.median(outlier))   # 238.0 50.0

# one standard deviation, three libraries, two defaults
print(round(np.std(salaries, ddof=1), 3))       # 7.906  NumPy told to divide by n - 1
print(round(pd.Series(salaries).std(), 3))      # 7.906  pandas divides by n - 1 unless told otherwise
print(round(statistics.pstdev(salaries.tolist()), 3),
      round(statistics.stdev(salaries.tolist()), 3))   # 7.071 7.906  population, then sample

# %% [markdown]
# Between the extremes sits the interquartile range, the spread between the 75th and 25th percentiles, whose breakdown point is ; Chapter 3 builds its standard outlier fences from it, a good practical compromise, with the MAD in reserve as th

# %%
import numpy as np

x = np.array([40, 45, 50, 55, 1000])           # four employees and a founder

# classic summaries: the founder owns both of them
mean, std = x.mean(), x.std()                   # np.std divides by n (population convention)
print(mean, round(std, 2))                      # 238.0 381.03

# classic z-scores: with n = 5, no score can exceed sqrt(4) = 2
z = (x - mean) / std
print(np.round(z, 2))                           # [-0.52 -0.51 -0.49 -0.48  2.  ]

# robust summaries: the median and the MAD never look at the 1000
med = np.median(x)
mad = np.median(np.abs(x - med))
print(med, mad)                                 # 50.0 5.0

# robust z-scores: the founder sits 128 robust sigmas out
rz = (x - med) / (1.4826 * mad)
print(np.round(rz, 2))                          # [ -1.35  -0.67   0.     0.67 128.15]

# the constant 1.4826: on a large clean Gaussian, 1.4826 * MAD recovers sigma
rng = np.random.default_rng(0)
g = rng.normal(0.0, 1.0, size=1_000_000)
print(round(1.4826 * np.median(np.abs(g - np.median(g))), 3))   # 1.002

# %% [markdown]
# pairing the data with Exponential or Uniform quantiles checks those families the same way.

# %%
import numpy as np
from scipy import stats

# by hand: the five salaries, sorted, are the empirical quantiles
salaries = np.array([40, 45, 50, 55, 60])
n = len(salaries)
p = (np.arange(1, n + 1) - 0.5) / n            # plotting positions: 0.1 0.3 0.5 0.7 0.9
q = stats.norm.ppf(p)                          # where a perfect Gaussian sample would sit
print(np.round(q, 4))                          # [-1.2816 -0.5244  0.      0.5244  1.2816]

# least-squares line through (q_i, x_(i)): slope estimates sigma, intercept estimates mu
slope = np.sum(q * salaries) / np.sum(q ** 2)  # q averages to zero, so this is the slope
print(round(float(slope), 2), round(float(np.mean(salaries)), 1))    # 8.05 50.0

# the library version builds the same pairs and fits the same line
(qt, xs), (m, b, r) = stats.probplot(salaries, dist="norm")
print(round(float(m), 2), round(float(b), 1), round(float(r), 4))    # 9.08 50.0 0.9984
print(np.round(qt, 3))                         # [-1.129 -0.486  0.     0.486  1.129]  quantiles at Filliben's levels

# straight for Gaussian data, bent for heavy tails: compare how straight the points are
rng = np.random.default_rng(0)
gauss = rng.normal(loc=50, scale=8, size=200)
heavy = 50 + 8 * rng.standard_t(df=2, size=200)   # centered at 50, scaled by 8, fat tails
print(round(float(stats.probplot(gauss, dist="norm")[1][2]), 4))     # 0.9959
print(round(float(stats.probplot(heavy, dist="norm")[1][2]), 4))     # 0.8755
print(round(float(gauss.max()), 1), round(float(heavy.max()), 1))    # 66.0 88.2

# %% [markdown]
# ## 2.4.2 Sampling and the central limit theorem

# %% [markdown]
# To cut that to 0.5 you would need a sample of 400.

# %%
import numpy as np

rng = np.random.default_rng(0)
population = rng.exponential(scale=1.0, size=1_000_000)   # a very skewed population
means = [rng.choice(population, size=50).mean() for _ in range(10_000)]
print(round(np.mean(means), 3), round(np.std(means), 3))  # 0.998 0.143

# %% [markdown]
# Roll a fair six-sided die, whose true mean is , and watch the running average after each roll.

# %%
import numpy as np

rng = np.random.default_rng(0)
rolls = rng.integers(1, 7, size=100_000)                      # a fair die, true mean 3.5
running = np.cumsum(rolls) / np.arange(1, len(rolls) + 1)     # the average after 1, 2, 3, ... rolls
print(round(running[9], 3), round(running[999], 3), round(running[99_999], 3))

# %% [markdown]
# Markov and Chebyshev on real data: the observed tail shares of the breast cancer mean area column against the Gaussian rule and the two bounds.

# %%
import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer

area = load_breast_cancer(as_frame=True).frame["mean area"].to_numpy()   # 569 tumors
mu, sd = area.mean(), area.std()       # the 569 values treated as the whole population
print(len(area), round(mu, 2), round(sd, 2))
for k in [2, 3]:
    share = np.mean(np.abs(area - mu) >= k * sd)       # observed share at least k sd from the mean
    print(k, round(share, 4), round(2 * norm.cdf(-k), 4), round(1 / k**2, 4))   # data, Gaussian, Chebyshev
for c in [2, 3]:
    print(c, round(np.mean(area >= c * mu), 4), round(1 / c, 4))   # Markov: P(X >= c mu) <= 1/c

# %% [markdown]
# Each additional decimal digit means a tenfold smaller error, and the says that costs one hundred times the darts.

# %%
import numpy as np

rng = np.random.default_rng(0)
for n in [100, 10_000, 1_000_000]:
    x = rng.random(n)                      # dart x-coordinates, uniform on [0, 1]
    y = rng.random(n)                      # dart y-coordinates
    inside = x**2 + y**2 <= 1.0            # True when a dart lands in the quarter circle
    pi_hat = 4 * inside.mean()             # the inside fraction estimates pi/4
    print(n, round(pi_hat, 4), "error", round(abs(pi_hat - np.pi), 4))
# 100 2.72 error 0.4216
# 10000 3.1448 error 0.0032
# 1000000 3.1414 error 0.0002

# %% [markdown]
# Try f(x) = cos(x) with X drawn from the standard Gaussian, whose exact expectation is exp(-1/2), about 0.6065.

# %%
rng = np.random.default_rng(0)                         # a fresh generator, so earlier draws do not matter
estimate = np.cos(rng.normal(size=1_000_000)).mean()   # f(x) = cos(x), X standard Gaussian
print(round(estimate, 4), round(np.exp(-0.5), 4))      # 0.6062 0.6065

# %% [markdown]
# The Cauchy distribution is the textbook case, its running average wandering forever.

# %%
rng = np.random.default_rng(0)
c = rng.standard_cauchy(size=100_000)
cr = np.cumsum(c) / np.arange(1, len(c) + 1)
print(round(cr[999], 2), round(cr[9_999], 2), round(cr[99_999], 2))   # -0.93  -0.14  0.3: never settles

# %% [markdown]
# When a dataset holds many rows per patient, user, or session, a plain random split scatters each group across train and test, so the model is partly graded on people it has already memorized; the honest split keeps every group wholly on one

# %%
import numpy as np

rng = np.random.default_rng(0)

# A population in two strata: 60% with mean 10, sd 2; 40% with mean 20, sd 3.
stratum_a = rng.normal(10, 2, size=600_000)
stratum_b = rng.normal(20, 3, size=400_000)
population = np.concatenate([stratum_a, stratum_b])

# Draw 10,000 samples of n = 100 under each design and watch the estimate wobble.
srs = [rng.choice(population, size=100).mean() for _ in range(10_000)]
strat = [0.6 * rng.choice(stratum_a, size=60).mean()      # 60 draws from A, weight 0.6
         + 0.4 * rng.choice(stratum_b, size=40).mean()    # 40 draws from B, weight 0.4
         for _ in range(10_000)]
print(round(np.std(srs), 3))     # 0.548  simple random: between-strata noise included
print(round(np.std(strat), 3))   # 0.244  stratified: only within-stratum noise survives

# %% [markdown]
# Move stratum B's mean from 20 to 30: the rerun keeps the seed and changes only that one number.

# %%
rng = np.random.default_rng(0)                    # the same seed as before
stratum_a = rng.normal(10, 2, size=600_000)
stratum_b = rng.normal(30, 3, size=400_000)       # stratum B's mean moved from 20 to 30
population = np.concatenate([stratum_a, stratum_b])
srs = [rng.choice(population, size=100).mean() for _ in range(10_000)]
strat = [0.6 * rng.choice(stratum_a, size=60).mean()
         + 0.4 * rng.choice(stratum_b, size=40).mean()
         for _ in range(10_000)]
print(round(np.std(srs), 3), round(np.std(strat), 3))   # 1.008 0.244

# %% [markdown]
# The error bar tightens around the wrong number, which is more dangerous than a wide error bar around the right one.

# %%
import numpy as np
rng = np.random.default_rng(0)

print(round(np.sqrt(2 / np.pi), 3))            # 0.798, the predicted biased mean
for n in [1_000, 1_000_000]:
    pop = rng.normal(0.0, 1.0, size=n)         # the real population: true mean 0
    seen = pop[pop > 0]                        # selection: only positive values get recorded
    print(n, round(seen.mean(), 3))            # 1000 0.782, then 1000000 0.799

# %% [markdown]
# ## 2.4.3 Hypothesis testing, p-values, and confidence intervals

# %% [markdown]
# The interval excludes 0.5, which is the same conclusion as the test, but it also shows the effect could plausibly be anywhere from a hair above fair to strongly biased, which a bare p-value hides.

# %%
from scipy import stats

# two-sided test that 60 heads in 100 flips is consistent with a fair coin
result = stats.binomtest(60, 100, 0.5)
print(round(result.pvalue, 3))      # 0.057 (exact test; the normal approximation gave ~0.046)

# the Gaussian shortcut with the continuity correction: "at least 60 heads" starts at 59.5
z_cc = (59.5 - 50) / 5
print(round(2 * (1 - stats.norm.cdf(z_cc)), 4))   # two-sided p-value, both tails

ci = result.proportion_ci()         # the exact 95 percent interval for the heads rate
print(round(ci.low, 3), round(ci.high, 3))

# %% [markdown]
# ## 2.4.4 Degrees of freedom and the t, chi-square, and F distributions

# %% [markdown]
# The block confirms the 0.8 by simulation.

# %%
import numpy as np
rng = np.random.default_rng(0)
samples = rng.normal(0.0, 1.0, size=(200_000, 5))   # 200,000 samples of n = 5; true variance 1
print(round(float(samples.var(axis=1, ddof=0).mean()), 3),   # ddof=0 divides by n
      round(float(samples.var(axis=1, ddof=1).mean()), 3))   # ddof=1 divides by n - 1

# %% [markdown]
# Degrees of freedom is the statistician's unit of model complexity, the budget of independent facts the data still holds after the model has been fed.

# %%
import numpy as np
from scipy import stats

# the five salaries from 2.4.1, in thousands of dollars
x = np.array([40, 45, 50, 55, 60])
n = len(x)

s = x.std(ddof=1)                      # Bessel's correction: divide the 250 by n - 1 = 4
se = s / np.sqrt(n)                    # estimated standard error of the mean
t_crit = stats.t.ppf(0.975, df=n - 1)  # 97.5th percentile of t with 4 degrees of freedom
print(round(s, 3), round(se, 3), round(t_crit, 3))   # 7.906 3.536 2.776

lo, hi = x.mean() - t_crit * se, x.mean() + t_crit * se
print(round(lo, 1), round(hi, 1))                    # 40.2 59.8

# why 1.96 is not enough at n = 5: draw 100,000 samples of five values from a
# Normal(50, 8) population and count how often each interval traps the true 50
rng = np.random.default_rng(0)
samples = rng.normal(50, 8, size=(100_000, n))
m = samples.mean(axis=1)
se_hat = samples.std(axis=1, ddof=1) / np.sqrt(n)
print(round(np.mean(np.abs(m - 50) <= 1.96 * se_hat), 3))     # 0.879  claims 95, delivers 88
print(round(np.mean(np.abs(m - 50) <= t_crit * se_hat), 3))   # 0.951  the honest interval

# %% [markdown]
# ## 2.4.5 The standard named tests: t-test, chi-square, and ANOVA

# %% [markdown]
# it is a two-sample test on proportions.

# %%
from scipy import stats

# one-sample t-test: the new model's accuracies on five random test splits against the claimed 0.80
acc_a = [0.82, 0.84, 0.81, 0.85, 0.83]
t1 = stats.ttest_1samp(acc_a, 0.80)
print(round(t1.statistic, 3), round(t1.pvalue, 4))    # 4.243 0.0132

# Welch two-sample t-test: model A's five splits against model B's five
acc_b = [0.80, 0.79, 0.82, 0.81, 0.78]
t2 = stats.ttest_ind(acc_a, acc_b, equal_var=False)
print(round(t2.statistic, 3), round(t2.pvalue, 4))    # 3.0 0.0171

# chi-square goodness of fit: 60 die rolls against the fair-die expectation of 10 each
chi = stats.chisquare([5, 8, 9, 8, 10, 20])
print(round(chi.statistic, 1), round(chi.pvalue, 4))  # 13.4 0.0199

# one-way ANOVA: three configurations, three latency measurements each
f = stats.f_oneway([4, 5, 6], [7, 8, 9], [10, 11, 12])
print(round(f.statistic, 1), round(f.pvalue, 5))      # 27.0 0.001

# the small-sample bar: two-sided 0.05 cutoffs for the normal and for t with 4 df
print(round(stats.norm.ppf(0.975), 2), round(stats.t.ppf(0.975, 4), 2))   # 1.96 2.78

# %% [markdown]
# The block runs all three tests on real data: logistic regression against the Gaussian Naive Bayes classifier of 6.2, both scored on the same five random 80/20 splits of the 569 tumors, which scikit-learn's `ShuffleSplit` draws.

# %%
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.naive_bayes import GaussianNB

X, y = load_breast_cancer(return_X_y=True)                     # 569 tumors, 30 features
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)   # five random 80/20 splits
acc_lr = cross_val_score(LogisticRegression(max_iter=5000), X, y, cv=cv)   # both models are
acc_nb = cross_val_score(GaussianNB(), X, y, cv=cv)                        # scored on the same splits
print(np.round(acc_lr, 3), np.round(acc_nb, 3))
print(round(float(np.corrcoef(acc_lr, acc_nb)[0, 1]), 2))   # do the two models' scores move together?

w = stats.ttest_ind(acc_lr, acc_nb, equal_var=False)   # unpaired (Welch): ignores the pairing
r = stats.ttest_rel(acc_lr, acc_nb)                     # paired: a one-sample test on the differences
print(round(w.statistic, 3), round(w.pvalue, 4))
print(round(r.statistic, 3), round(r.pvalue, 4))

d = acc_lr - acc_nb                     # one difference per shared split
J, n2 = len(d), 114                     # 5 splits, each with 114 test rows
n1 = len(y) - n2                        # and 455 training rows
t_c = d.mean() / np.sqrt((1 / J + n2 / n1) * d.var(ddof=1))   # the corrected resampled t
print(round(t_c, 3), round(2 * (1 - stats.t.cdf(abs(t_c), df=J - 1)), 4))

# %% [markdown]
# The experiment the text proposes: change `n_splits` to 20 and rerun the paired and corrected tests.

# %%
cv20 = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)   # twenty random 80/20 splits
acc_lr20 = cross_val_score(LogisticRegression(max_iter=5000), X, y, cv=cv20)
acc_nb20 = cross_val_score(GaussianNB(), X, y, cv=cv20)
d20 = acc_lr20 - acc_nb20                 # one difference per shared split, now twenty of them
r20 = stats.ttest_rel(acc_lr20, acc_nb20) # the plain paired test
t_c20 = d20.mean() / np.sqrt((1 / len(d20) + n2 / n1) * d20.var(ddof=1))   # the corrected resampled t
print(round(r20.statistic, 3), round(r20.pvalue, 4))                       # 4.974 0.0001
print(round(t_c20, 3), round(2 * (1 - stats.t.cdf(abs(t_c20), df=len(d20) - 1)), 4))   # 2.029 0.0567

# %% [markdown]
# ## 2.4.6 Nonparametric tests: Mann-Whitney and Kolmogorov-Smirnov

# %% [markdown]
# Ranked by absolute size, the lone negative difference comes first, so the negative ranks sum to 1 and the positive ranks to 20.

# %%
import numpy as np
from scipy import stats

a = np.array([1, 2, 3, 4, 5])          # sample A
b = np.array([3, 4, 5, 6, 7])          # sample B, shifted right by 2

# build both empirical CDFs at every observed value: the fraction of each
# sample at or below x, exactly the staircase we walked by hand
xs = np.sort(np.concatenate([a, b]))
F_a = np.searchsorted(a, xs, side="right") / len(a)
F_b = np.searchsorted(b, xs, side="right") / len(b)
print(np.max(np.abs(F_a - F_b)))       # 0.4, the by-hand D

# scipy's two-sample KS test: the same D plus an exact p-value
res = stats.ks_2samp(a, b)
print(round(res.statistic, 3), round(res.pvalue, 3))   # 0.4 0.873
print(round(stats.kstwobign.isf(0.05), 3))   # 1.358: the large-sample 5 percent point behind the rule of thumb

# the rank-based Mann-Whitney U test on the same two samples
mw = stats.mannwhitneyu(a, b)
print(mw.statistic, round(mw.pvalue, 3))   # 4.5 0.114: U for sample A, then the p-value

# the same p-value by hand: null mean 12.5, tie-corrected variance 22.5,
# and U_A moved half a unit toward the mean (the continuity correction)
z = (4.5 + 0.5 - 12.5) / np.sqrt(22.5)
print(round(z, 3), round(2 * stats.norm.cdf(z), 3))   # -1.581 0.114

# check the null mean and variance exactly: deal the ten pooled values into two
# groups of five in all 252 equally likely ways and compute U_A for every deal
from itertools import combinations
pooled = np.concatenate([a, b])
us = []
for idx in combinations(range(10), 5):
    A, B = pooled[list(idx)], np.delete(pooled, list(idx))
    us.append(np.sum(A[:, None] > B) + 0.5 * np.sum(A[:, None] == B))   # wins plus half the ties
print(len(us), np.mean(us), np.var(us))   # 252 12.5 22.5

# the signed-rank test on paired data: six made-up per-split accuracy differences
diff = np.array([0.012, 0.025, -0.004, 0.018, 0.031, 0.009])
sr = stats.wilcoxon(diff)
print(sr.statistic, round(sr.pvalue, 4))

# %% [markdown]
# Here are two groups where every value of the second sits about 0.2 above the first, except that one wild reading of 50 landed in the second group.

# %%
import numpy as np
from scipy import stats

# group y sits consistently about 0.2 above group x, but one wild
# outlier landed in y; watch what each test does with it
x = np.array([5.1, 4.9, 5.0, 5.2, 4.8, 5.0])
y = np.array([5.3, 5.1, 5.2, 5.4, 5.0, 50.0])

print(round(stats.ttest_ind(x, y).pvalue, 3))      # 0.329
print(round(stats.mannwhitneyu(x, y).pvalue, 3))   # 0.043

# %% [markdown]
# ## 2.4.7 A/B testing

# %% [markdown]
# A three-line check confirms the arithmetic.

# %%
import numpy as np
from scipy import stats
n_a = n_b = 2000
p_pool = (200 + 240) / (n_a + n_b)                      # 0.11, the shared rate under the null
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))   # the pooled standard error
z = (240/2000 - 200/2000) / se
print(round(float(se), 4), round(float(z), 2),
      round(float(2 * (1 - stats.norm.cdf(z))), 3))     # 0.0099 2.02 0.043

# %% [markdown]
# ## 2.4.8 Statistical power and sample size: how much data do you need

# %% [markdown]
# Halving the detectable lift roughly quadruples the sample, which is why serious experimentation platforms fight for every scrap of variance reduction.

# %%
import numpy as np
from scipy import stats

# the three design choices: baseline rate, smallest lift worth detecting, error budgets
p_a, p_b = 0.10, 0.12                    # control converts at 10%; a 2-point lift matters to us
alpha, power = 0.05, 0.80                # 5% false-alarm rate, 80% chance of catching the lift

z_alpha = stats.norm.ppf(1 - alpha / 2)  # 1.96, the two-sided critical value
z_beta = stats.norm.ppf(power)           # 0.8416, the quantile that leaves beta = 0.20

var_sum = p_a * (1 - p_a) + p_b * (1 - p_b)   # 0.09 + 0.1056 = 0.1956
delta = p_b - p_a                             # 0.02, the minimum detectable effect

n = (z_alpha + z_beta) ** 2 * var_sum / delta ** 2
print(int(np.ceil(n)))                   # 3839 users per group

# how much power did the 2000-per-group test actually have against this lift?
se = np.sqrt(var_sum / 2000)
print(round(stats.norm.cdf(delta / se - z_alpha), 3))    # 0.525, a coin flip

# the standard tool: statsmodels solves the same design question
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
h = proportion_effectsize(p_b, p_a)      # the effect size on a variance-stabilized scale
print(round(NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power)))  # 3835

# %% [markdown]
# ## 2.4.9 The bootstrap

# %% [markdown]
# Repeat a few thousand times, and read a 95 percent confidence interval straight off the resampled values as the range from their 2.5th to their 97.5th percentile.

# %%
rng = np.random.default_rng(0)
data = np.array([12, 15, 14, 10, 13, 99, 11, 14, 12, 13])   # note the outlier 99
boot_medians = [np.median(rng.choice(data, size=len(data), replace=True))
                for _ in range(10_000)]
lo, hi = np.percentile(boot_medians, [2.5, 97.5])
print(round(lo, 1), round(hi, 1))     # 11.5 14.5

# %% [markdown]
# The block lists all 27 resamples to confirm the count.

# %%
import numpy as np
from collections import Counter
from itertools import product

data3 = [1, 2, 6]
meds = [float(np.median(r)) for r in product(data3, repeat=3)]   # all 3**3 = 27 ordered resamples
print(sorted(Counter(meds).items()))           # each resample median with its count
print(round(float(np.std(meds)), 3))           # the exact bootstrap standard error
print(np.percentile(meds, [2.5, 97.5]))        # the 95 percent percentile interval

# %% [markdown]
# The block repeats everything for samples of 10 and of 100.

# %%
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
for n in (10, 100):
    pct = bca = 0
    for _ in range(1_000):                          # 1,000 fresh samples of size n
        x = rng.exponential(1.0, size=n)            # skewed data whose true mean is 1
        boot = rng.choice(x, size=(2_000, n), replace=True).mean(axis=1)   # 2,000 resample means
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pct += lo <= 1.0 <= hi                      # did the percentile interval trap the true 1?
        ci = stats.bootstrap((x,), np.mean, n_resamples=2_000, method="BCa", rng=rng).confidence_interval
        bca += ci.low <= 1.0 <= ci.high             # and did the BCa interval?
    print(n, pct / 1_000, bca / 1_000)

# %% [markdown]
# Example 2-8 runs both versions, the exact enumeration on the four timings and the ten thousand shuffles on the A/B test.

# %%
import numpy as np
from itertools import combinations

# --- the tiny experiment, exactly: try every relabeling of the 4 values ---
values = np.array([10.0, 12.0, 14.0, 16.0])   # A measured 10 and 12 ms, B measured 14 and 16
observed = values[2:].mean() - values[:2].mean()           # 15 - 11 = 4.0
diffs = []
for a_idx in combinations(range(4), 2):       # each way to tag 2 of the 4 values as group A
    mask = np.zeros(4, dtype=bool); mask[list(a_idx)] = True
    diffs.append(values[~mask].mean() - values[mask].mean())
diffs = np.sort(np.array(diffs))
print(diffs)                                  # [-4. -2.  0.  0.  2.  4.]
print(round(np.mean(np.abs(diffs) >= observed), 3))        # 0.333, the exact p-value

# --- the A/B test of 2.4.7, rerun by shuffling instead of the z formula ---
rng = np.random.default_rng(0)
outcomes = np.array([1]*200 + [0]*1800 + [1]*240 + [0]*1760)   # A's 2000 visitors, then B's
obs_lift = outcomes[2000:].mean() - outcomes[:2000].mean()     # 0.12 - 0.10 = 0.02
count = 0
for _ in range(10_000):
    shuffled = rng.permutation(outcomes)      # a world where the labels mean nothing
    lift = shuffled[2000:].mean() - shuffled[:2000].mean()
    count += abs(lift) >= abs(obs_lift)       # did chance match or beat the real lift?
print(round((1 + count) / (1 + 10_000), 3))   # 0.053 (the z formula said 0.043)

# %% [markdown]
# An estimate of a probability near 0.05 from ten thousand shuffles wobbles by a few thousandths, so ten times as many shuffles pin it down more tightly.

# %%
rng = np.random.default_rng(0)
count = 0
for _ in range(100_000):                      # ten times as many shuffled worlds
    shuffled = rng.permutation(outcomes)
    lift = shuffled[2000:].mean() - shuffled[:2000].mean()
    count += abs(lift) >= abs(obs_lift)
print(round((1 + count) / (1 + 100_000), 4))  # 0.0495

# %% [markdown]
# ## 2.4.10 Multiple testing

# %% [markdown]
# Both numbers can be watched in a simulation, since a p-value from a test where nothing is real is uniformly distributed on .

# %%
rng = np.random.default_rng(0)
p_null = rng.random((100_000, 20))               # 20 null tests per experiment, nothing real anywhere
print(round(float((p_null.min(axis=1) < 0.05).mean()), 3))     # 0.642  experiments with a false alarm
print(round(float((p_null.min(axis=1) < 0.0025).mean()), 3))   # 0.048  Bonferroni restores 5 percent

# %% [markdown]
# Bonferroni guards against even one false alarm, Benjamini-Hochberg tolerates a small, controlled fraction of false discoveries in exchange for far more power.

# %%
p = np.array([0.001, 0.012, 0.014, 0.04, 0.20])  # the five p-values, already sorted
thresholds = 0.05 * np.arange(1, 6) / 5          # i * alpha / m for i = 1..5
print(thresholds)                                # [0.01 0.02 0.03 0.04 0.05]
print(np.where(p <= thresholds)[0].max() + 1)    # 4  the largest passing rank: reject the first four

# %% [markdown]
# ## 2.4.11 Correlation versus causation

# %% [markdown]
# A confounder can be built in five lines, which makes the illusion concrete.

# %%
rng = np.random.default_rng(0)
heat = rng.normal(size=10_000)                    # the confounder: summer temperature
ice_cream = heat + 0.5 * rng.normal(size=10_000)  # heat drives ice cream sales
drownings = heat + 0.5 * rng.normal(size=10_000)  # heat also drives swimming, hence drownings
print(round(float(np.corrcoef(ice_cream, drownings)[0, 1]), 2))              # 0.8   strongly linked
print(round(float(np.corrcoef(ice_cream - heat, drownings - heat)[0, 1]), 2))  # -0.0  nothing left

# %% [markdown]
# ## 2.4.12 Paradoxes that trip up analysts

# %% [markdown]
# Berkson's paradox by simulation: the block confirms that selection squeezes the spread of the total S while the spread of the difference D stays where it was.

# %%
import numpy as np
rng = np.random.default_rng(0)

x, y = rng.standard_normal((2, 100_000))      # two unrelated scores for 100,000 people
kept = x + y > 1                              # the selection: only totals above 1 are kept
print(round(np.corrcoef(x, y)[0, 1], 3))      # everyone
print(round(np.corrcoef(x[kept], y[kept])[0, 1], 3), kept.sum())   # the kept, and how many
s, d = x[kept] + y[kept], x[kept] - y[kept]   # total and difference among the kept
print(round(s.var(), 3), round(d.var(), 3))   # selection squeezed the total, not the difference
print(round((s.var() - d.var()) / 4, 4),
      round(np.cov(x[kept], y[kept], ddof=0)[0, 1], 4))   # Cov(X, Y) from the identity, then directly

# %% [markdown]
# The lesson, which recurs throughout statistics, is that new information must be conditioned on correctly, and gut feeling is a poor guide.

# %%
rng = np.random.default_rng(0)
def play(switch):
    car = rng.integers(0, 3); pick = rng.integers(0, 3)
    goats = [d for d in range(3) if d != pick and d != car]    # doors the host may open
    opened = rng.choice(goats)                                 # the host opens one, at random if two qualify
    if switch:
        pick = [d for d in range(3) if d != pick and d != opened][0]   # the one other closed door
    return pick == car
print(round(np.mean([play(True)  for _ in range(100_000)]), 3))   # 0.667
print(round(np.mean([play(False) for _ in range(100_000)]), 3))   # 0.333

# %% [markdown]
# And the harder you selected, the more candidates you screened to find that 130, the luckier the winner and the harsher the fall.

# %%
import numpy as np
rng = np.random.default_rng(0)

n = 1_000_000
skill = rng.normal(100, 10, size=n)            # true ability, sigma_T = 10
test1 = skill + rng.normal(0, 10, size=n)      # first measurement adds luck, sigma_eps = 10
test2 = skill + rng.normal(0, 10, size=n)      # retest: same skill, fresh luck
top = test1 > 125                              # select the apparent stars
print(round(test1[top].mean(), 1), round(test2[top].mean(), 1))   # 130.7 115.3
print(round(100 + 0.5 * (test1[top].mean() - 100), 1))            # 115.4, the formula's prediction

# %% [markdown]
# ## 2.4.13 Judging an estimator: bias, variance, mean squared error, and consistency

# %% [markdown]
# The variance estimators on a real population: the 569 tumor mean areas, sampled with replacement 100,000 times at each sample size.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer

area = load_breast_cancer(as_frame=True).frame["mean area"].to_numpy()
sigma2 = area.var()                                     # the population variance, divisor 569
rng = np.random.default_rng(0)
for n in [5, 50]:
    samples = rng.choice(area, size=(100_000, n))       # 100,000 samples of size n, with replacement
    for ddof in [1, 0]:                                 # divide by n - 1, then by n
        est = samples.var(axis=1, ddof=ddof)
        bias, var = est.mean() - sigma2, est.var()
        print(n, ddof, round(bias / sigma2, 3), round(var / sigma2**2, 3),
              round((bias**2 + var) / sigma2**2, 3))    # bias, variance, MSE in units of sigma^2

# %% [markdown]
# The German tank problem on an invented population of 100 tanks, four seen at random, repeated 100,000 times.

# %%
rng = np.random.default_rng(0)
N, n = 100, 4                                           # 100 tanks exist; 4 serial numbers are seen
tanks = np.tile(np.arange(1, N + 1), (100_000, 1))      # 100,000 copies of the serials 1 to 100
seen = rng.permuted(tanks, axis=1)[:, :n]               # shuffle each copy and keep the first 4
m = seen.max(axis=1)                                    # the largest serial seen: the MLE
corrected = m + (m - n) / n                             # add the average gap below the maximum
for name, est in [("maximum", m), ("corrected", corrected)]:
    print(name, round(est.mean(), 2), round(est.std(), 2), round(np.mean((est - N) ** 2), 1))

# %% [markdown]
# ## 2.4.14 Kurtosis and the trimmed mean

# %%
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer

# the five salaries of 2.4.1, then the same five with the founder
for x in [np.array([40, 45, 50, 55, 60]), np.array([40, 45, 50, 55, 1000])]:
    z = (x - x.mean()) / x.std()                          # standardized with the divide-by-n sd
    print(round(float(np.mean(z**4)), 4),                 # kurtosis from the formula
          round(float(stats.kurtosis(x, fisher=False)), 4),   # SciPy's kurtosis, 3 not subtracted
          stats.trim_mean(x, 0.2))                        # drop one value at each end, average the rest

area = load_breast_cancer(as_frame=True).frame["mean area"].to_numpy()   # 569 tumors
print(round(float(stats.skew(area)), 2), round(float(stats.kurtosis(area)), 2))   # skewness, excess kurtosis
print(round(float(area.mean()), 1), round(float(stats.trim_mean(area, 0.1)), 1),
      round(float(np.median(area)), 1))                   # mean, 10 percent trimmed mean, median

# %% [markdown]
# Reference values of the excess kurtosis, then the wobble of the mean, the 10 percent trimmed mean, and the median on Gaussian and heavy-tailed samples.

# %%
import numpy as np
from scipy import stats

# excess kurtosis of four model distributions, as SciPy computes it
for name, dist in [("uniform", stats.uniform), ("gaussian", stats.norm),
                   ("laplace", stats.laplace), ("t, 3 df", stats.t(3))]:
    print(name, float(dist.stats(moments="k")))

# 20,000 samples of 50 values each; every estimate below targets the true center 0
rng = np.random.default_rng(0)
for name, samples in [("gaussian", rng.normal(size=(20_000, 50))),
                      ("t, 3 df", rng.standard_t(3, size=(20_000, 50)))]:
    centers = [samples.mean(axis=1),                      # the mean
               stats.trim_mean(samples, 0.1, axis=1),     # the 10 percent trimmed mean
               np.median(samples, axis=1)]                # the median
    print(name, [round(float(c.var()), 4) for c in centers])   # variance of each estimate = its MSE

# %% [markdown]
# ## 2.4.15 Kernel density estimation: a smooth histogram

# %%
import numpy as np
from scipy import stats
from sklearn.datasets import load_iris

def kde(x, data, h):
    # one Gaussian bump of width h on every observation, averaged; the 1/h keeps the area at 1
    u = (x[:, None] - data[None, :]) / h                  # every point of x against every observation
    return np.exp(-u**2 / 2).sum(axis=1) / (len(data) * h * np.sqrt(2 * np.pi))

print(np.round(kde(np.array([2.0, 3.0]), np.array([1.0, 2.0, 4.0]), 1.0), 4))   # the hand example

petal = load_iris().data[:, 2]                            # petal length in cm, 150 flowers
for start in [1.0, 0.75]:                                 # bins 0.5 cm wide, edges shifted by 0.25 cm
    print(np.histogram(petal, bins=np.arange(start, 7.5, 0.5))[0])

h = petal.std(ddof=1) * len(petal) ** (-1 / 5)            # Scott's rule, SciPy's default bandwidth
grid = np.linspace(-3.0, 11.0, 1401)                      # wide enough to hold every bump
ours = kde(grid, petal, h)
print(round(float(h), 4), np.allclose(ours, stats.gaussian_kde(petal)(grid)),
      round(float(np.trapezoid(ours, grid)), 4))          # bandwidth, agreement with SciPy, area
print(round(float(stats.norm.cdf(-petal / h).mean()), 4)) # share of the area below 0 cm

for bw in [0.05, 0.2, h, 1.5]:                            # from undersmoothed to oversmoothed
    dens = kde(grid, petal, bw)
    peak = (dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:])   # higher than both neighbors
    print(round(float(bw), 3), int(peak.sum()), np.round(grid[1:-1][peak][:3], 2))

# %% [markdown]
# ## 2.5.1 Entropy

# %% [markdown]
# Figure 2-10 plots this for every bias, a curve that peaks at one bit for the fair coin and falls to zero at either certain extreme.

# %%
import numpy as np

def entropy(p):
    p = np.array(p)
    p = p[p > 0]                       # 0 log 0 is treated as 0
    return -np.sum(p * np.log2(p))

print(round(entropy([0.5, 0.5]), 3))   # 1.0    a fair coin
print(round(entropy([0.9, 0.1]), 3))   # 0.469  a predictable coin
print(round(entropy([1.0]), 3))        # -0.0   a certain outcome

# %% [markdown]
# among all distributions with a given variance , the Gaussian has the largest differential entropy, and that maximum is h(\text{Gaussian}) = \tfrac{1}{2}\log_2\big(2\pi e\,\sigma^2\big).

# %%
sigma = 1.0
h_gauss = 0.5 * np.log2(2 * np.pi * np.e * sigma ** 2)
print(round(h_gauss, 3))                       # 2.047 bits for a standard Gaussian
print(round(np.log2(0.5), 3), round(np.log2(2.0), 3))   # -1.0 and 1.0: uniform [0,a] can be negative

# %% [markdown]
# ## 2.5.2 Cross-entropy, the loss of classification

# %% [markdown]
# For the common two-class case this is written out as binary cross-entropy, H(p, q) = -\big[y\log_2 q + (1 - y)\log_2(1 - q)\big], where is 0 or 1 for the true label and is the model's predicted probability of class 1.

# %%
def cross_entropy_true_class(q_true):
    return -np.log2(q_true)            # loss when the model gives q_true to the correct class

print(round(cross_entropy_true_class(0.7), 3))   # 0.515  confident and right
print(round(cross_entropy_true_class(0.1), 3))   # 3.322  confident and wrong
print(round(cross_entropy_true_class(1.0), 3))   # -0.0   perfect

# %% [markdown]
# This subtract-the-max trick is the numerically stable softmax, and it is what every framework implements.

# %%
def softmax(z):
    z = np.array(z, dtype=float)
    e = np.exp(z - z.max())            # subtract the max: same answer, no overflow
    return e / e.sum()

print(np.round(softmax([2.0, 1.0, 0.0]), 4))         # [0.6652 0.2447 0.09  ]
print(round(float(softmax([2.0, 1.0, 0.0]).sum()), 4))   # 1.0  a legitimate distribution
print(np.round(softmax([1000.0, 999.0, 998.0]), 4))  # [0.6652 0.2447 0.09  ]  stable at huge scores
with np.errstate(over="ignore", invalid="ignore"):   # silence the warnings; the nans tell the story
    naive = np.exp(np.array([1000., 999., 998.]))
    print(naive / naive.sum())                       # [nan nan nan]  the naive route self-destructs

# %% [markdown]
# Log-sum-exp computed stably and checked against SciPy, then the log softmax of scores where the naive routes overflow or underflow.

# %%
from scipy.special import logsumexp

def log_sum_exp(z):
    z = np.array(z, dtype=float)
    m = z.max()
    return m + np.log(np.exp(z - m).sum())     # factor e^m out of the sum, then take the log

z = np.array([1000.0, 999.0, 998.0])
with np.errstate(over="ignore"):
    print(np.log(np.exp(z).sum()))             # the naive route: e^1000 overflows first
print(round(log_sum_exp(z), 4), round(float(logsumexp(z)), 4))   # ours, then SciPy's
print(np.round(z - log_sum_exp(z), 4))         # log softmax: each score minus logsumexp

s = np.array([0.0, -800.0])
with np.errstate(divide="ignore"):
    print(np.log(softmax(s)))                  # softmax first, then the log
print(s - log_sum_exp(s))                      # the log probabilities computed directly

# %% [markdown]
# The cross-entropy of a trained classifier computed from its raw scores as logsumexp(z) - z_y and checked against scikit-learn's log_loss on the wine data.

# %%
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

X_wine, y_wine = load_wine(return_X_y=True)             # 178 wines, 13 measurements, 3 cultivars
X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.3,
                                                    random_state=0, stratify=y_wine)
mean, std = X_train.mean(axis=0), X_train.std(axis=0)   # standardize with training statistics
model = LogisticRegression(max_iter=1000).fit((X_train - mean) / std, y_train)
logits = model.decision_function((X_test - mean) / std) # raw scores, one column per class
rows = np.arange(len(y_test))
by_hand = np.mean(logsumexp(logits, axis=1) - logits[rows, y_test])   # mean of logsumexp(z) - z_y
proba = model.predict_proba((X_test - mean) / std)
print(logits.shape, round(by_hand, 6), round(log_loss(y_test, proba), 6))
print(np.allclose(proba, np.exp(logits - logsumexp(logits, axis=1, keepdims=True))))

# %% [markdown]
# ## 2.5.3 KL divergence

# %% [markdown]
# The model's wrong confidence cost it about three quarters of a bit of waste on every coin flip.

# %%
def kl(p, q):
    p, q = np.array(p), np.array(q)
    return np.sum(p * np.log2(p / q))

print(round(kl([0.5, 0.5], [0.9, 0.1]), 3))   # 0.737
print(round(kl([0.9, 0.1], [0.5, 0.5]), 3))   # 0.531  not the same, KL is asymmetric

# %% [markdown]
# For each of three distributions with variance 1 the block computes the entropy and the cross-entropy against the standard Gaussian, the latter by numerical integration.

# %%
from scipy import stats
from scipy.integrate import quad

gauss = stats.norm(0, 1)                                   # the standard Gaussian, variance 1
rivals = {"Gaussian": (gauss, -30, 30),
          "Laplace": (stats.laplace(0, 1 / np.sqrt(2)), -30, 30),        # scale 1/sqrt(2): variance 1
          "uniform": (stats.uniform(-np.sqrt(3), 2 * np.sqrt(3)), -np.sqrt(3), np.sqrt(3))}
for name, (dist, lo, hi) in rivals.items():
    def term(x):
        return -dist.pdf(x) * np.log2(gauss.pdf(x))       # integrand of the cross-entropy H(p, Gaussian)
    cross = quad(term, lo, hi, points=[0])[0]              # quad integrates numerically over [lo, hi]
    h_bits = dist.entropy() / np.log(2)                    # SciPy reports nats; convert to bits
    print(f"{name:8s} variance {dist.var():.3f}  h {h_bits:.4f}  H(p, Gaussian) {cross:.4f}  KL {cross - h_bits:.4f}")

# %% [markdown]
# ## 2.5.4 Mutual information

# %% [markdown]
# So knowing the weather removes about a third of a bit of uncertainty about the umbrella; if the joint had been the product of the marginals cell for cell, every ratio would be 1 and the total would be exactly 0.

# %%
P = np.array([[0.30, 0.05],
              [0.10, 0.55]])          # Table 2-5: rows rain yes/no, columns umbrella yes/no
px = P.sum(axis=1)                    # marginal of rain: [0.35 0.65]
py = P.sum(axis=0)                    # marginal of umbrella: [0.40 0.60]
I = sum(P[i, j] * np.log2(P[i, j] / (px[i] * py[j]))
        for i in range(2) for j in range(2))
print(round(float(I), 4))             # 0.3613 bits, the four hand terms summed

indep = np.outer(px, py)              # the joint a world of independent variables would produce
qx, qy = indep.sum(axis=1), indep.sum(axis=0)   # recompute ITS marginals the same way
I0 = sum(indep[i, j] * np.log2(indep[i, j] / (qx[i] * qy[j]))
         for i in range(2) for j in range(2))
print(round(float(I0), 4))            # -0.0  every cell sits on its independence baseline

# %% [markdown]
# The block repeats the conditional entropy computation with the entropy function of 2.5.1 and checks the chain rule.

# %%
H_rain = entropy(px)                                    # 2.5.1's entropy: uncertainty about rain alone
H_rain_given_umb = sum(py[j] * entropy(P[:, j] / py[j])   # each column of P, rescaled, is p(rain | umbrella)
                       for j in range(2))
print(round(float(H_rain), 4), round(float(H_rain_given_umb), 4),
      round(float(H_rain - H_rain_given_umb), 4))       # H(X), H(X | Y), and their difference
print(round(float(entropy(P)), 4),
      round(float(entropy(py) + H_rain_given_umb), 4))  # chain rule: H(X, Y) = H(Y) + H(X | Y)

# %% [markdown]
# Information gain on the Titanic passengers: the entropy of survival, the conditional entropy given sex and given class, and the check with mutual_info_score.

# %%
import seaborn as sns
from sklearn.metrics import mutual_info_score

titanic = sns.load_dataset("titanic")             # the 891 passengers of the 1.8 lab
survived = titanic["survived"]

def conditional_entropy(target, feature):
    """H(target | feature) in bits: the entropy left in each group, weighted by group size."""
    shares = feature.value_counts(normalize=True)
    return sum(shares[g] * entropy(target[feature == g].value_counts(normalize=True))
               for g in shares.index)

H_surv = entropy(survived.value_counts(normalize=True))
print("H(survived)", round(H_surv, 4))
for col in ["sex", "class"]:
    H_left = conditional_entropy(survived, titanic[col])
    gain = H_surv - H_left                        # information gain, which is I(survived; col)
    check = mutual_info_score(survived, titanic[col]) / np.log(2)   # scikit-learn reports nats
    print(col, round(H_left, 4), round(gain, 4), round(check, 4))
for cls, group in titanic.groupby("class", observed=True)["survived"]:
    print(cls, round(entropy(group.value_counts(normalize=True)), 4))   # entropy inside each class

# %% [markdown]
# ## 2.6.1 Convex versus nonconvex functions

# %% [markdown]
# The eigenvalue test is two lines of code, reusing eigvalsh from 2.1.8 on the Hessians of the two-dimensional bowl and the saddle.

# %%
import numpy as np

for name, H in [("bowl", np.diag([2., 2.])), ("saddle", np.diag([2., -2.]))]:
    ev = np.linalg.eigvalsh(H)
    print(name, ev, "convex" if ev.min() >= 0 else "not convex")

# %% [markdown]
# ## 2.6.2 Gradient descent from scratch

# %% [markdown]
# The algorithm found the bottom of the bowl with nothing but the slope to guide it.

# %%
def f_prime(x):
    return 2 * x                      # derivative of f(x) = x^2

x, eta = 10.0, 0.1
for step in range(40):
    x = x - eta * f_prime(x)          # one gradient-descent step
print(round(x, 6))                    # 0.001329  effectively the minimum at 0

# %% [markdown]
# ## 2.6.3 The learning rate: too big, too small, just right

# %% [markdown]
# a ratio smaller than one in magnitude converges, a ratio of magnitude exactly one circles forever (the flip between 10 and is the ratio ), and a ratio larger than one in magnitude explodes.

# %%
def run(eta, steps=8):
    x = 10.0
    for _ in range(steps):
        x = x - eta * 2 * x
    return round(x, 3)

print(run(0.1))   # 1.678   too small: still far from 0 after 8 steps
print(run(0.5))   # 0.0     just right: reaches the minimum immediately
print(run(1.1))   # 42.998  too big: diverging away from 0

# %% [markdown]
# Escaping requires changing the geometry itself, which is what preconditioning and the adaptive optimizers of Chapter 11 do; in effect they shrink the the optimizer feels.

# %%
import numpy as np

# The ill-conditioned bowl f(x, y) = 0.5 * (100 x^2 + y^2).
# Its Hessian has eigenvalues 100 and 1: L = 100, mu = 1, kappa = 100.
H = np.diag([100.0, 1.0])
L, mu, kappa = 100.0, 1.0, 100.0

def descend(eta, steps):
    v = np.array([1.0, 1.0])            # start at (1, 1)
    for _ in range(steps):
        v = v - eta * H @ v             # the gradient of (1/2) v^T H v is H v
    return v

# The theorem says a fixed step is safe only below 2/L = 0.02.
print(descend(0.019, 200))   # [7.05507911e-10 2.15675829e-02]  just under: converging
print(descend(0.021, 200))   # [1.89905276e+08 1.43396437e-02]  just over: the stiff axis explodes

# Strongly convex rate: at eta = 2/(L + mu) the error contracts by (kappa-1)/(kappa+1).
eta_best = 2 / (L + mu)
r = np.linalg.norm(descend(eta_best, 101)) / np.linalg.norm(descend(eta_best, 100))
print(round(r, 6), round((kappa - 1) / (kappa + 1), 6))   # 0.980198 0.980198  measured = predicted

# Convex O(1/k) guarantee at eta = 1/L: the gap after k steps is at most L*R^2/(2k).
k = 100
v = descend(1 / L, k)
gap = 0.5 * v @ H @ v                   # f(x_k) - f*, since the minimum value is 0
bound = L * 2.0 / (2 * k)               # R^2 = ||x_0 - x*||^2 = 1^2 + 1^2 = 2
print(round(gap, 4), "<=", round(bound, 4))               # 0.067 <= 1.0  the guarantee holds

# %% [markdown]
# ## 2.6.4 Batch, stochastic, and minibatch gradient descent

# %% [markdown]
# Take the linear-fit setup that the lab in 2.10 will use, and compare the exact full-batch gradient of the slope at the starting point against minibatch estimates of three sizes.

# %%
rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=10_000)
y = 2 * X + 1 + rng.normal(0, 1, size=10_000)                # the lab's noisy line, writ large
grad_w = lambda idx: float((2 / len(idx)) * np.sum((0 * X[idx] - y[idx]) * X[idx]))  # dL/dw at w = b = 0
print(round(grad_w(np.arange(10_000)), 2))                   # -12.02  the exact full-batch gradient
for m in [1, 32, 1024]:
    est = [grad_w(rng.integers(0, 10_000, m)) for _ in range(500)]
    print(m, round(float(np.mean(est)), 1), round(float(np.std(est)), 1))

# %% [markdown]
# ## 2.6.5 Second-order methods: Newton's method

# %% [markdown]
# Gradient descent needed dozens of steps for the same bowl, and three lines make the contrast concrete.

# %%
x = 10.0                                # the same start 2.6.2 gave gradient descent
x = x - (2 * x) / 2.0                   # one Newton step: slope over curvature
print(x)                                # 0.0  the exact minimum, first try

# %% [markdown]
# ## 2.6.6 Constrained optimization and Lagrange multipliers

# %% [markdown]
# This algorithm belongs to the class that Beck and Teboulle (2009) call ISTA, short for iterative shrinkage-thresholding algorithms.

# %%
import numpy as np

def soft_threshold(v, lam):
    # S_lambda(v): move v toward zero by lam, and stop exactly at zero if it would cross
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)

v = np.array([3.0, -2.5, 0.4])
print(soft_threshold(v, 1.0))      # [ 2.  -1.5  0. ]   the three by-hand values
print(v / (1 + 2 * 1.0))           # [ 1.         -0.83333333  0.13333333]   ridge: shrunk, never zero

# proximal gradient descent (ISTA) on a tiny Lasso problem:
# minimize 0.5 * ||X w - y||^2 + lam * ||w||_1, true weights (2, 0, -0.5)
rng = np.random.default_rng(0)
X = rng.normal(size=(50, 3))
y = X @ np.array([2.0, 0.0, -0.5]) + 0.05 * rng.normal(size=50)
lam, eta = 5.0, 0.01
w = np.zeros(3)
for _ in range(500):
    grad = X.T @ (X @ w - y)                       # gradient of the smooth data loss only
    w = soft_threshold(w - eta * grad, eta * lam)  # then shrink or kill every coordinate
print(np.round(w, 3), w[1] == 0.0)                 # [ 1.917  0.    -0.363] True

# %% [markdown]
# Experiment: at lam = 20 the weak true weight dies as well and only the strongest survives.

# %%
lam = 20.0                                         # a stronger penalty, same data and step size
w = np.zeros(3)
for _ in range(500):
    grad = X.T @ (X @ w - y)
    w = soft_threshold(w - eta * grad, eta * lam)
print(np.round(w, 3), w[1] == 0.0)                 # [ 1.648  0.    -0.   ] True

# %% [markdown]
# A short script confirms every number.

# %%
import numpy as np
from scipy.optimize import minimize

# Primal: minimize x^2 + y^2 subject to x + y >= 1.
# scipy states inequality constraints as fun(v) >= 0, so we hand it x + y - 1.
f = lambda v: v[0]**2 + v[1]**2
con = {"type": "ineq", "fun": lambda v: v[0] + v[1] - 1}
best = minimize(f, x0=[3.0, -2.0], constraints=[con])
print(np.round(best.x, 3), round(best.fun, 3))          # [0.5 0.5] 0.5, the KKT solution

# Shadow-price check: relax the constraint to x + y >= 0.9.
# The multiplier mu = 1 prices that relaxation at about 1 * 0.1.
con9 = {"type": "ineq", "fun": lambda v: v[0] + v[1] - 0.9}
relaxed = minimize(f, x0=[3.0, -2.0], constraints=[con9])
print(round(best.fun - relaxed.fun, 4))                 # 0.095, the drop from 0.5 to 0.405

# Dual: q(mu) = mu - mu^2 / 2. Scan mu >= 0 for the tightest lower bound.
mus = np.linspace(0, 2, 2001)
q = mus - mus**2 / 2
print(round(mus[np.argmax(q)], 3), round(q.max(), 3))   # 1.0 0.5, meeting the primal: gap zero

# %% [markdown]
# Experiment: with the objective (x - 2)^2 + y^2 the unconstrained minimum (2, 0) already satisfies x + y >= 1, so the constraint goes slack and relaxing it buys nothing.

# %%
f2 = lambda v: (v[0] - 2)**2 + v[1]**2
best2 = minimize(f2, x0=[3.0, -2.0], constraints=[con])
print(np.round(best2.x, 3), round(best2.fun, 3))        # [2. 0.] 0.0
relaxed2 = minimize(f2, x0=[3.0, -2.0], constraints=[con9])
print(round(best2.fun - relaxed2.fun, 4))               # 0.0: a slack constraint has a zero shadow price

# %% [markdown]
# Projected gradient descent: one projected step on a box by hand, then nonnegative least squares on the diabetes data, checked against SciPy and the KKT conditions.

# %%
import numpy as np
from scipy.optimize import nnls
from sklearn.datasets import load_diabetes

# by hand: one projected step for f(w) = (w1 - 2)^2 + (w2 + 1)^2 on the box [0, 1] x [0, 1]
w = np.array([0.5, 0.5])
step = w - 0.25 * 2 * (w - np.array([2.0, -1.0]))       # ordinary gradient step with eta = 0.25
print(step, np.clip(step, 0.0, 1.0))                    # the step, then its projection

# nonnegative least squares: minimize 0.5 * ||X w - y||^2 subject to every w_i >= 0
X, y = load_diabetes(return_X_y=True)
y = y - y.mean()                                        # the features are centered, so no intercept
L = np.linalg.eigvalsh(X.T @ X).max()                   # smoothness constant; eta = 1/L is safe (2.6.3)
w = np.zeros(X.shape[1])
for _ in range(500):
    w = np.maximum(w - X.T @ (X @ w - y) / L, 0.0)      # gradient step, then project onto w >= 0
print(np.round(w, 1))
print(np.abs(w - nnls(X, y)[0]).max() < 1e-9)           # SciPy's active-set solver agrees

g = X.T @ (X @ w - y)                                   # KKT check at the answer
print(bool(np.abs(g[w > 0]).max() < 1e-6), round(float(g[w == 0].min()), 1))

ols = np.linalg.lstsq(X, y, rcond=None)[0]              # least squares with no constraint
r2 = lambda v: 1 - np.sum((X @ v - y) ** 2) / np.sum(y ** 2)
print(round(float(r2(ols)), 4), round(float(r2(w)), 4),
      round(float(r2(np.maximum(ols, 0.0))), 4))        # OLS, NNLS, and OLS with negatives clipped
print(round(float(L), 4), np.round(ols[4:6], 1),        # L, the OLS weights on s1 and s2,
      f"{np.corrcoef(X[:, 4], X[:, 5])[0, 1]:.2f}",     # the correlation of those two columns,
      round(float(np.linalg.norm(ols)), 1))             # and the length of the OLS weight vector

# %% [markdown]
# Experiment: swap nonnegativity for a ball of radius 500, the constraint form of ridge regression.

# %%
w = np.zeros(X.shape[1])
for _ in range(500):
    v = w - X.T @ (X @ w - y) / L                       # the same gradient step
    w = v * min(1.0, 500 / np.linalg.norm(v))           # then project onto the ball ||w|| <= 500
print(round(float(np.linalg.norm(w)), 4), round(float(r2(w)), 4))   # 500.0 0.4466
print(np.round(w, 1))                                   # no weight is zero
print(np.round(ols, 1))                                 # some weights grew or changed sign

# %% [markdown]
# ## 2.6.7 Optimization beyond calculus: linear programming and classic OR problems

# %% [markdown]
# The shortest path finds the cheapest route between two nodes, and the minimum spanning tree finds the cheapest set of edges that connects every node.

# %%
import networkx as nx
G = nx.Graph()
G.add_weighted_edges_from([("A","B",2), ("A","C",5), ("B","C",1),
                           ("B","D",7), ("C","D",3), ("D","E",1), ("C","E",6)])
print(nx.dijkstra_path(G, "A", "E"), nx.dijkstra_path_length(G, "A", "E"))   # ['A','B','C','D','E'] 7
mst = nx.minimum_spanning_tree(G)
print(sum(w for _, _, w in mst.edges(data="weight")))                        # 7

# %% [markdown]
# How much water can you push from a source to a sink through pipes of limited capacity? Exactly as much as the capacity of the tightest "cut" that separates source from sink, a theorem with no slack in it.

# %%
D = nx.DiGraph()
D.add_edge("s","a", capacity=3); D.add_edge("s","b", capacity=2)
D.add_edge("a","b", capacity=1); D.add_edge("a","t", capacity=2); D.add_edge("b","t", capacity=3)
flow_value, _ = nx.maximum_flow(D, "s", "t")
cut_value, (reachable, non_reachable) = nx.minimum_cut(D, "s", "t")
print(flow_value, cut_value)                              # 5 5: max flow equals min cut
print(sorted(reachable), sorted(non_reachable))           # ['a', 'b', 's'] ['t']: the two sides of the cut

# %% [markdown]
# Maximizing profit is a linear program, and scipy solves both it and its dual at once.

# %%
from scipy.optimize import linprog
c = [-30, -50]                                            # minimize negative profit = maximize profit
A_ub = [[1, 2], [3, 1]]; b_ub = [40, 60]                  # labor and material constraints
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)], method="highs")
print(res.x, -res.fun)                                    # [16. 12.] 1080.0
print(-res.ineqlin.marginals)                             # [24.  2.]  shadow prices

# %% [markdown]
# For the simplest queue, one server with random (Poisson) arrivals and service, the M/M/1 model gives the average time in the system in closed form.

# %%
lam, mu = 8.0, 10.0                                       # arrivals/hour, service rate/hour
rho = lam / mu                                            # utilization = 0.8
L = rho / (1 - rho)                                       # average number in system
W = 1 / (mu - lam)                                        # average time in system (hours)
print(round(L, 2), round(W * 60), round(lam * W, 2))      # 4.0  30  4.0  (L == lambda*W)
Lq, Wq = L - rho, W - 1 / mu                              # queue only: exclude the one being served
print(round(Lq, 1), round(Wq * 60))                       # 3.2  24

# %% [markdown]
# The block builds the M/M/1 chain with steps of 0.05 hour, cut off at 59 customers, starts it empty, multiplies by the transition matrix 5,000 times, and compares where the probability settles with the formula.

# %%
import numpy as np

lam, mu, dt, N = 8.0, 10.0, 0.05, 60          # rates per hour, a short time step, states 0 to 59
Q = np.zeros((N, N))                          # transition matrix of the chain, rows are "now"
for n in range(N):
    up = lam * dt if n < N - 1 else 0.0       # an arrival during the step
    down = mu * dt if n > 0 else 0.0          # a departure, possible only if someone is there
    if n < N - 1:
        Q[n, n + 1] = up
    if n > 0:
        Q[n, n - 1] = down
    Q[n, n] = 1 - up - down                   # nothing happens
pi = np.zeros(N)
pi[0] = 1.0                                   # start with an empty system
for _ in range(5000):
    pi = pi @ Q                               # step the chain forward, as in 2.3.8
rho = lam / mu
print(np.round(pi[:4], 4))                    # long-run fractions of time with 0, 1, 2, 3 customers
print(np.round((1 - rho) * rho ** np.arange(4), 4))    # the formula (1 - rho) rho^n
L_chain = float(np.arange(N) @ pi)            # average number in the system
print(round(L_chain, 3), round(L_chain / lam * 60, 1))  # L, then W in minutes by Little's law

# %% [markdown]
# Experiment: set lam to 5.0, so rho = 0.5; the fractions should halve from 0.5, L should be 1, and W should be 12 minutes.

# %%
lam = 5.0                                     # the arrival rate changes; mu, dt, and N stay as before
Q = np.zeros((N, N))
for n in range(N):
    up = lam * dt if n < N - 1 else 0.0
    down = mu * dt if n > 0 else 0.0
    if n < N - 1:
        Q[n, n + 1] = up
    if n > 0:
        Q[n, n - 1] = down
    Q[n, n] = 1 - up - down
pi = np.zeros(N)
pi[0] = 1.0
for _ in range(5000):
    pi = pi @ Q
rho = lam / mu
print(np.round(pi[:4], 4))                    # [0.5    0.25   0.125  0.0625]
print(np.round((1 - rho) * rho ** np.arange(4), 4))
L_chain = float(np.arange(N) @ pi)
print(round(L_chain, 3), round(L_chain / lam * 60, 1))  # 1.0 12.0

# %% [markdown]
# ## 2.6.8 When there is no gradient: derivative-free methods

# %% [markdown]
# The same uphill move is welcome early, when escaping bad valleys matters, and forbidden late, when the search should settle.

# %%
delta = 1.0                                    # the move makes the loss worse by 1
for T in (2.0, 0.5, 0.1):                      # hot, cooling, cold
    print(T, round(np.exp(-delta / T), 5))     # 2.0 0.60653   0.5 0.13534   0.1 5e-05

# %% [markdown]
# ## 2.6.9 Game theory essentials: minimax and Nash equilibrium

# %% [markdown]
# A short code check confirms both the equalization argument and the prisoner's dilemma verdict.

# %%
M = np.array([[0, -1, 1],
              [1, 0, -1],
              [-1, 1, 0]])                  # row player's payoff: rows and columns are rock, paper, scissors
sigma = np.ones(3) / 3
print(M @ sigma)                            # [0. 0. 0.]  every pure reply earns 0: no profitable deviation

R = np.array([[3, 0], [5, 1]])              # prisoner's dilemma, row player's payoff (0 = cooperate, 1 = defect)
nash = [(i, j) for i in (0, 1) for j in (0, 1)
        if R[i, j] >= R[1 - i, j] and R[j, i] >= R[1 - j, i]]
print(nash)                                 # [(1, 1)]  mutual defection is the only pure equilibrium

# %% [markdown]
# Experiment: tilt the opponent's mix away from uniform.

# %%
print(M @ np.array([0.5, 0.25, 0.25]))      # [ 0.    0.25 -0.25]  paper now earns 0.25: an exploitable opening

# %% [markdown]
# ## 2.6.10 Calculus of variations: optimizing over functions

# %% [markdown]
# Start from a wildly wiggly path between and , hold the endpoints fixed, and minimize the total length.

# %%
import numpy as np
N = 40
x = np.linspace(0, 1, N + 1)
y = np.random.default_rng(0).normal(0, 0.3, N + 1); y[0], y[-1] = 0.0, 1.0   # wiggly, fixed ends
dx = x[1] - x[0]
def length(y): return float(np.sum(np.sqrt(dx**2 + np.diff(y)**2)))           # the functional, discretized
print(round(length(y), 4))                                                    # 10.009: the wiggly start
for _ in range(5000):
    seg = np.sqrt(dx**2 + np.diff(y)**2)
    dyl = np.diff(y) / seg                                                     # gradient of length wrt each point
    g = np.zeros_like(y); g[1:] += dyl; g[:-1] -= dyl; g[0] = g[-1] = 0.0      # fix the endpoints
    y = y - 0.02 * g
print(round(length(y), 4), round(float(np.sqrt(2)), 4))                       # 1.4142 1.4142
print(round(float(np.max(np.abs(y - x))), 4))                                 # 0.0001: it became the line y = x

# %% [markdown]
# ## 2.7.4 Why the type decides the math

# %% [markdown]
# A short, optional check shows the encoding rule in action.

# %%
import pandas as pd

df = pd.DataFrame({"country": ["US", "Brazil", "US"], "size": ["S", "L", "M"]})
print(pd.get_dummies(df["country"], prefix="country", dtype=int).to_numpy().tolist())
# nominal -> one-hot: [[0, 1], [1, 0], [0, 1]]  one 0/1 column per country, no false order

order = {"S": 0, "M": 1, "L": 2}
print(df["size"].map(order).tolist())
# ordinal -> ordered integers: [0, 2, 1]  small<medium<large preserved

# %% [markdown]
# ## 2.7.7 Worked example: one messy dataset, every column labeled

# %% [markdown]
# A short pandas block puts the two readings side by side.

# %%
import pandas as pd
from sklearn.datasets import fetch_openml

# titanic3, OpenML dataset 40945: one row per passenger (Harrell, 2002).
# The first call downloads it from openml.org; later calls read scikit-learn's local cache.
t3 = fetch_openml(data_id=40945, as_frame=True).frame
print(t3.shape, t3.select_dtypes("number").columns.tolist())

# 1. Let the storage type decide: get_dummies one-hot encodes every text or category
#    column and passes every number through unchanged.
print(pd.get_dummies(t3).shape)

# 2. Read the columns by meaning instead, the way Table 2-9 did.
has_body = t3["body"].notna()          # the number of an identified victim
print(has_body.sum(), round(t3["body"].mean(), 1), t3.loc[has_body, "survived"].unique().tolist())
has_boat = t3["boat"].notna()          # the boat that rescued a survivor
print(has_boat.sum(), (t3.loc[has_boat, "survived"] == "1").sum())
print(round(t3["pclass"].mean(), 3), t3["pclass"].median())   # ordinal codes 1, 2, 3
print(round(t3["sibsp"].mean(), 3), t3["sibsp"].median())     # a count of relatives aboard
print(t3["name"].nunique(), t3["ticket"].nunique(), t3["home.dest"].nunique())   # text codes and free text

# 3. The plan by meaning: set aside the target, the identifiers, the two leaks, and the
#    free text; keep pclass as ordered integers; one-hot encode only sex and embarked.
features = t3.drop(columns=["survived", "name", "ticket", "cabin", "boat", "home.dest", "body"])
print(pd.get_dummies(features, dtype=int).columns.tolist())

# %% [markdown]
# ## 2.8.1 Propositional logic: statements, connectives, and truth tables

# %% [markdown]
# Three of the five connectives are already Python operators (and, or, not); the other two are one line each, and from there a small helper prints any two-variable table.

# %%
from itertools import product

def IMPLIES(p, q): return (not p) or q          # p -> q is false ONLY when p is true and q is false
def IFF(p, q):     return p == q                 # p <-> q is true when both sides agree

def truth_table(name, fn):
    print(f"{name}")
    print("   p      q   | out")
    for p, q in product([False, True], repeat=2):
        print(f"  {str(p):5}  {str(q):5} | {fn(p, q)}")

connectives = [("AND (p and q)", lambda p, q: p and q),    # Python's own operators
               ("OR (p or q)", lambda p, q: p or q),
               ("IMPLIES (p -> q)", IMPLIES),              # the two defined above
               ("IFF (p <-> q)", IFF)]
for name, fn in connectives:
    truth_table(name, fn)

print("NOT (not p)")                                       # one input, so two rows
print("   p    | out")
for p in [False, True]:
    print(f"  {str(p):5}  | {not p}")

# %% [markdown]
# ## 2.8.2 Equivalence, tautology, and the laws of reasoning

# %% [markdown]
# Equivalence, tautology, and contradiction are each a one-line exhaustive check over the 2^n rows of the truth table.

# %%
def equivalent(f, g, n):
    # two formulas of n variables are equivalent if they agree on ALL 2^n assignments
    return all(f(*v) == g(*v) for v in product([False, True], repeat=n))
def is_tautology(f, n):     return all(f(*v) for v in product([False, True], repeat=n))
def is_contradiction(f, n): return not any(f(*v) for v in product([False, True], repeat=n))

IMPLIES_TABLE = {(False, False): True, (False, True): True,    # the four rows of the 2.8.1 table:
                 (True, False): False, (True, True): True}     # the definition of p -> q itself
demorgan   = equivalent(lambda p, q: not (p and q), lambda p, q: (not p) or (not q), 2)
impl_as_or = equivalent(lambda p, q: IMPLIES_TABLE[(p, q)], lambda p, q: (not p) or q, 2)
print(f"De Morgan: not(p and q) == (not p) or (not q)?  {demorgan}")
print(f"implication rewrite: (p -> q) == (not p) or q?    {impl_as_or}")
print(f"law of excluded middle (p or not p) a tautology?  {is_tautology(lambda p: p or (not p), 1)}")
print(f"(p and not p) a contradiction?                    {is_contradiction(lambda p: p and (not p), 1)}")

# %% [markdown]
# ## 2.8.3 Inference and satisfiability

# %% [markdown]
# Modus ponens is sound: in every possible world where both premises hold, the conclusion holds too, and with only two variables we can check every world outright.

# %%
# soundness of modus ponens: collect every assignment where both premises hold,
# then confirm the conclusion q holds in all of them
premises_hold = [(p, q) for p, q in product([False, True], repeat=2) if p and IMPLIES(p, q)]
print(premises_hold)                     # [(True, True)]: the only world with both premises
print(all(q for p, q in premises_hold))  # True: q holds in every such world, so the rule is sound

# %% [markdown]
# Satisfiability (SAT) asks whether any assignment of true and false to a formula's variables makes the whole formula true. The simplest solver tries every assignment.

# %%
def sat(formula, n):
    # the simplest SAT solver: try every assignment, return the first that satisfies the formula
    for v in product([False, True], repeat=n):
        if formula(*v): return v
    return None
# (p OR q) AND (NOT p OR r) AND (NOT q OR NOT r), a 3-variable formula in conjunctive normal form
clauses = lambda p, q, r: (p or q) and ((not p) or r) and ((not q) or (not r))
sol = sat(clauses, 3)
print(f"SAT for (p|q)&(~p|r)&(~q|~r): satisfying assignment (p,q,r) = {sol}")
print(f"  checking {2**3} assignments here; for n variables it is 2^n. SAT is the original NP-complete problem.")

# %% [markdown]
# ## 2.8.4 First-order logic: predicates and quantifiers

# %% [markdown]
# The universal quantifier reads "for all x, P(x) holds," and the existential quantifier reads "there exists an x for which P(x) holds." In Python, over a finite domain, these are exactly all(...) and any(...).

# %%
domain = list(range(1, 11))                      # the universe of discourse: 1..10
def is_even(x): return x % 2 == 0
forall_even = all(is_even(x) for x in domain)    # for-all x: is_even(x)
exists_even = any(is_even(x) for x in domain)    # there-exists x: is_even(x)
print(f"for-all x, even(x)      : {forall_even}   (false: 1 is a counterexample)")
print(f"there-exists x, even(x) : {exists_even}    (true: 2 witnesses it)")

# %% [markdown]
# ## 2.8.5 Beyond true and false: fuzzy, probabilistic, and temporal logic

# %% [markdown]
# AND becomes the minimum of the two values, OR becomes the maximum, and NOT becomes one minus the value.

# %%
def fuzzy_and(a, b): return min(a, b)            # AND = the weaker of the two
def fuzzy_or(a, b):  return max(a, b)            # OR  = the stronger of the two
def fuzzy_not(a):    return 1.0 - a
hot, warm = 0.8, 0.6
print(f"fuzzy: 'hot'={hot}, 'warm'={warm}")
print(f"  hot AND warm = min = {fuzzy_and(hot, warm)}")
print(f"  hot OR warm  = max = {fuzzy_or(hot, warm)}")
print(f"  NOT hot      = 1-x = {fuzzy_not(hot):.1f}")

# %% [markdown]
# The block builds three joint tables with the same two marginal probabilities and reads the conjunction off each.

# %%
import numpy as np

pA, pB = 0.9, 0.8
# three joint tables, rows A and not A, columns B and not B, all with P(A) = 0.9 and P(B) = 0.8
tables = {"independent": np.outer([pA, 1 - pA], [pB, 1 - pB]),
          "most overlap": np.array([[0.8, 0.1], [0.0, 0.1]]),
          "least overlap": np.array([[0.7, 0.2], [0.1, 0.0]])}
for name, J in tables.items():
    print(f"{name:13s}  P(A)={J[0].sum():.2f}  P(B)={J[:, 0].sum():.2f}  P(A and B)={J[0, 0]:.2f}")
print(f"fuzzy AND of 0.9 and 0.8 = {min(pA, pB):.2f}")

# %% [markdown]
# On a recorded run, judged from its start, always and eventually are the quantifiers of 2.8.4 with time as the domain, so all and any compute them.

# %%
trace = ["idle", "pick", "move", "place", "idle", "pick", "move"]   # a robot's state at each step

def always(prop, run):         # G prop: prop holds at every step of the run
    return all(prop(state) for state in run)

def eventually(prop, run):     # F prop: prop holds at some step of the run
    return any(prop(state) for state in run)

def is_safe(state):
    return state != "collide"

def is_place(state):
    return state == "place"

def every_pick_placed(run):    # G(pick -> F place): from each pick onward, a place must come
    return all(eventually(is_place, run[i:]) for i, state in enumerate(run) if state == "pick")

print(always(is_safe, trace), eventually(is_place, trace))
print(every_pick_placed(trace), every_pick_placed(trace + ["place"]))

# %% [markdown]
# ## 2.9.1 Complex numbers: rotations and the eigenvalues of everything else

# %% [markdown]
# Complex arithmetic: the product rule, the modulus, two quarter turns, and Euler's formula from its Taylor series.

# %%
import cmath
import math

z, w = 1 + 2j, 3 + 1j                         # Python writes the imaginary unit i as j
print(z * w)                                  # the product rule (ac - bd) + (ad + bc)i
print(abs(3 + 4j))                            # abs() returns the modulus sqrt(3**2 + 4**2)
print(round(abs(z) * abs(w), 4), round(abs(z * w), 4))   # lengths multiply
print(1j * (1 + 0j), 1j * 1j)                 # multiplying by i: two quarter turns
theta = math.pi / 2
series = sum((1j * theta) ** n / math.factorial(n) for n in range(20))   # 20 Taylor terms of e^(i theta)
print(f"{abs(series - 1j):.1e}")              # how far the 20-term sum lands from i
print(cmath.exp(1j * theta))                  # the library's e^(i pi/2)

# %% [markdown]
# The spectral theorem of 2.1.12 promised that symmetric matrices have real eigenvalues, but a general matrix need not, and the cleanest example is a pure rotation.

# %%
import numpy as np, cmath
R = np.array([[0., -1.], [1., 0.]])              # rotate the plane by 90 degrees
vals, _ = np.linalg.eig(R)
print(np.round(vals, 3))                          # [0.+1.j 0.-1.j]: the eigenvalues are i and -i
print(f"{abs(cmath.exp(1j * np.pi) + 1):.2e}")    # 1.22e-16: Euler's identity e^(i*pi) + 1 = 0

# %% [markdown]
# A rotation by any angle theta has eigenvalues cos(theta) +- i sin(theta) = e^(+-i theta); check it at theta = pi/6, a 30-degree turn.

# %%
theta = np.pi / 6                                 # a 30-degree rotation
R30 = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]])
vals30, _ = np.linalg.eig(R30)
print(np.round(vals30, 3))                        # cos(theta) + i sin(theta) and its conjugate
print(np.round(cmath.exp(1j * theta), 3), np.round(np.abs(vals30), 3))   # e^(i theta), then both moduli

# %% [markdown]
# ## 2.9.2 The Fourier transform, a light touch

# %% [markdown]
# Convolution two ways, by the flip-and-slide sum and by the convolution theorem (zero-pad, transform, multiply pointwise, then transform back), then an asymmetric kernel that shows the flip, and the distribution of the sum of two dice.

# %%
import numpy as np
f = np.array([1.0, 2.0, 3.0])
g = np.array([1.0, 1.0])
print(np.convolve(f, g))                        # [1. 3. 5. 3.]  the by-hand flip-and-slide
n = len(f) + len(g) - 1                         # zero-pad to 4 so circular = ordinary
via_fft = np.fft.ifft(np.fft.fft(f, n) * np.fft.fft(g, n)).real
print(np.round(via_fft, 12))                    # [1. 3. 5. 3.]  the convolution theorem at work
k = np.array([1.0, -1.0])                       # an asymmetric kernel, so the flip shows
print(np.convolve(f, k), np.correlate(f, k, "full"))  # [ 1.  1.  1. -3.] [-1. -1. -1.  3.]
die = np.ones(6) / 6                            # a fair die as a probability vector
two = np.convolve(die, die)                     # distribution of the sum of two dice
print(round(float(two.max()), 4), int(np.argmax(two)) + 2)   # 0.1667 7

# %% [markdown]
# Build one second of signal from two sines, a 5 Hz wave of amplitude 1 plus a 40 Hz wave of amplitude 0.5, transform it, and ask where the spectrum peaks.

# %%
t = np.arange(500) / 500                                   # one second, 500 samples
sig = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 40 * t)
X = np.fft.rfft(sig)                                       # the DFT bins k = 0, 1, ..., 250
amp = 2 * np.abs(X) / len(t)                               # amplitude at 0, 1, 2, ... Hz
top = np.sort(np.argsort(amp)[-2:])                        # the two strongest frequencies
print(len(X), np.round(np.abs(X[top]), 3))                 # 251 [250. 125.]: raw magnitudes N*A/2
print(top, np.round(amp[top], 3))                          # [ 5 40] [1.  0.5]
print(f"{np.sort(amp)[-3]:.1e}")                           # the third-largest amplitude, the biggest leftover

# %% [markdown]
# ## 2.9.3 Differential equations: how things change, and the diffusion behind generative models

# %% [markdown]
# Both can be solved numerically by adapting the move of 2.6.10, which discretized a function into points and then optimized those points. Here we discretize time, and for the PDE space as well, and then step the equation forward from one grid point to the next, a discretize-then-step scheme in place of the discretize-then-optimize one.

# %%
# ODE dy/dt = -y by Euler's method
t = np.linspace(0, 5, 501); dt = t[1] - t[0]
y = np.zeros_like(t); y[0] = 1.0
for k in range(len(t) - 1):
    y[k + 1] = y[k] + dt * (-y[k])               # step forward: y_{k+1} = y_k + dt * y'(y_k)
print(f"{y[-1]:.4f} vs e^-5 {np.exp(-5):.4f}")    # 0.0066 vs 0.0067

# heat-equation PDE by finite differences
nx = 60; dx = 1.0 / nx; alpha = 1.0
dt_pde = 0.4 * dx**2 / alpha                       # stability needs dt <= dx^2 / (2 alpha)
u = np.zeros(nx); u[nx//2 - 2: nx//2 + 2] = 1.0    # a hot block in the middle
for _ in range(4000):
    lap = np.zeros_like(u)
    lap[1:-1] = u[2:] - 2 * u[1:-1] + u[:-2]        # discrete second derivative (the Laplacian)
    u = u + alpha * dt_pde / dx**2 * lap            # explicit update
print(f"peak {u.max():.3f}")                        # peak fell from 1.0 to 0.001: spread out and drained at the ends

# %% [markdown]
# The boundary condition is the other variation worth running, because it changes the answer.

# %%
# the first heat-equation run held both end cells at 0: how much of the starting heat (4 cells of 1.0) is left?
print(f"heat left with cold ends: {u.sum() * dx:.4f} of {4 * dx:.4f}")

# the same hot block between insulated ends: each end cell trades heat only with its one neighbor
u = np.zeros(nx); u[nx//2 - 2: nx//2 + 2] = 1.0
for _ in range(4000):
    lap = np.zeros_like(u)
    lap[1:-1] = u[2:] - 2 * u[1:-1] + u[:-2]
    lap[0], lap[-1] = u[1] - u[0], u[-2] - u[-1]    # no heat crosses the outer edges
    u = u + alpha * dt_pde / dx**2 * lap
print(f"insulated ends: peak {u.max():.3f}, heat {u.sum() * dx:.4f}")

# %% [markdown]
# ## 2.10 Hands-on lab: gradient descent from scratch

# %% [markdown]
# Everything in Example 2-9 is just these formulas in a loop.

# %%
import numpy as np

# 1. Make noisy data from the true line y = 2x + 1, so we know the right answer.
rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=200)
y = 2 * X + 1 + rng.normal(0, 1, size=200)     # true slope 2, true intercept 1

# 2. Start the slope and intercept at zero, and pick a learning rate.
w, b = 0.0, 0.0
eta = 0.05
losses = []                                    # the loss before each step, for Figure 2-15

# 3. Gradient descent: predict, measure error, step downhill, repeat.
for step in range(1000):
    y_hat = w * X + b                          # the linear model from 2.1
    error = y_hat - y                          # how wrong each prediction is
    losses.append(np.mean(error ** 2))         # the loss L(w, b), the mean squared error
    grad_w = (2 / len(X)) * np.sum(error * X)  # dL/dw by the chain rule (2.2.3)
    grad_b = (2 / len(X)) * np.sum(error)      # dL/db
    w -= eta * grad_w                          # the descent step from 2.6.2
    b -= eta * grad_b

print(round(w, 3), round(b, 3))                # 1.958 0.921
print(round(losses[0], 3), round(losses[-1], 3))   # 15.495 1.042

# %% [markdown]
# Example 2-10 runs the same update loop on the standardized values.

# %%
import numpy as np
from sklearn.datasets import load_diabetes

# 1. Load real data: 442 diabetes patients (Efron et al., 2004), bundled with scikit-learn.
#    scaled=False keeps the raw measurements; column 2 is body mass index (BMI).
data = load_diabetes(scaled=False)
bmi = data.data[:, 2]
y = data.target                                # disease progression one year after baseline

# 2. Standardize BMI so the loss bowl is round. np.std divides by n, so X has
#    mean 0 and the average of X**2 equals 1 up to rounding.
X = (bmi - bmi.mean()) / bmi.std()

# 3. The update loop from Example 2-9, without its loss record.
w, b = 0.0, 0.0
eta = 0.05
for step in range(1000):
    y_hat = w * X + b
    error = y_hat - y
    grad_w = (2 / len(X)) * np.sum(error * X)
    grad_b = (2 / len(X)) * np.sum(error)
    w -= eta * grad_w
    b -= eta * grad_b

print(len(X), round(w, 3), round(b, 3))           # 442 45.16 152.133

# 4. The referee: the exact least-squares line from the normal equation of 2.1.11.
A = np.column_stack([X, np.ones(len(X))])          # one column for the slope, one for the intercept
print(np.linalg.solve(A.T @ A, A.T @ y).round(3))  # [ 45.16  152.133]

# 5. The loss floor against the spread of y around its mean.
mse = np.mean((w * X + b - y) ** 2)
print(round(mse, 1), round(y.var(), 1))            # 3890.5 5929.9

# %% [markdown]
# The starter block for these tasks loads both datasets of the lab and lays out the three functions they need, with bodies for you to write.

# %%
import numpy as np
from sklearn.datasets import load_diabetes

# The 200 synthetic points of Example 2-9: the same seed gives the same points.
rng = np.random.default_rng(0)
X_syn = rng.uniform(-3, 3, size=200)
y_syn = 2 * X_syn + 1 + rng.normal(0, 1, size=200)

# The 442 patients of Example 2-10 (Efron et al., 2004): raw BMI, standardized BMI, and progression.
data = load_diabetes(scaled=False)
bmi, y_db = data.data[:, 2], data.target
X_db = (bmi - bmi.mean()) / bmi.std()

def gradient_descent(X, y, eta, steps=1000):
    """Run the lab's loop from w = b = 0; return w, b, and the list of losses before each step."""
    ...

def least_squares_line(X, y):
    """Return the exact slope and intercept from the normal equation of 2.1.11."""
    ...

def minibatch_gradient_descent(X, y, eta, batch_size, steps=1000, seed=0):
    """Run the same loop, but compute each step's gradient on batch_size points drawn at random."""
    ...

print(X_syn.shape, X_db.shape, gradient_descent(X_syn, y_syn, eta=0.05))

# %% [markdown]
# ## 2.11 Interview questions and coding exercises
# Each coding exercise starter block runs as given; the full solutions are in Appendix H.

# %% [markdown]
# Coding exercise 1, A dense layer by hand: the starter block.

# %%
import numpy as np

def dense_layer(W, x, b):
    """Return W x + b computed with explicit loops over rows and columns, without @ or broadcasting."""
    ...

rng = np.random.default_rng(0)               # the seed of the worked solution, so your numbers can match it
W = rng.standard_normal((4, 3))
x = rng.standard_normal(3)
b = rng.standard_normal(4)
print(W.shape, x.shape, b.shape, dense_layer(W, x, b))

# %% [markdown]
# Coding exercise 2, Ill conditioning and the ridge cure: the starter block.

# %%
import numpy as np

A = np.array([[1.0, 1.0], [1.0, 1.0001]])    # the nearly singular matrix of 2.1.10
b = np.array([2.0, 2.0001])                   # a right-hand side to solve for, and later to perturb

def ridge_solve(A, b, lam):
    """Return the solution x of the ridge system (A^T A + lam I) x = A^T b."""
    ...

print(A.shape, b.shape, ridge_solve(A, b, lam=0.01))

# %% [markdown]
# Coding exercise 3, PCA from scratch: the starter block.

# %%
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

wine = load_wine()                            # the wine data of 2.1.13

def explained_variance_ratio(X):
    """Return the explained-variance ratios of the standardized columns of X, largest first, via np.linalg.eigh."""
    ...

print(wine.data.shape, explained_variance_ratio(wine.data))

# %% [markdown]
# Coding exercise 4, SVD image compression: the starter block.

# %%
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
img = digits.images[0]                        # the first image, as an array of pixel intensities

def relative_error(A, k):
    """Return ||A - A_k||_F / ||A||_F, where A_k is rebuilt from the top k singular values of A."""
    ...

print(img.shape, digits.target[0], relative_error(img, 1))

# %% [markdown]
# Coding exercise 5, Cosine-similarity nearest neighbor: the starter block.

# %%
import numpy as np
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)           # the digits of exercise 4, one row of pixel values per image

def most_similar(Xn, i):
    """Return the index of the row of the unit-length matrix Xn most similar to row i, excluding i itself."""
    ...

def one_nn_accuracy(Xn, y):
    """Return the accuracy of labeling each row of Xn with the label of its most similar other row."""
    ...

print(X.shape, y[:10])

# %% [markdown]
# Coding exercise 6, Gradient descent linear regression, end to end: the starter block.

# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)         # the 442 patients of Example 2-10 (Efron et al., 2004)

def mse_gradient(X, y, w):
    """Return the analytic gradient (2/n) X^T (X w - y) of the mean squared error at w."""
    ...

def numerical_gradient(f, w, eps=1e-5):
    """Return the central-difference estimate of the gradient of f at w, one coordinate at a time."""
    ...

def gradient_descent(X, y, eta, steps):
    """Run gradient descent on the mean squared error from w = 0 and return the final w."""
    ...

print(X.shape, y.shape)

# %% [markdown]
# Coding exercise 7, Bayes for a rare disease: the starter block.

# %%
def posterior(prior, sensitivity, specificity):
    """Return P(disease | positive test) by Bayes' theorem."""
    ...

print(posterior(0.001, 0.99, 0.99))

# %% [markdown]
# Coding exercise 8, Bootstrap versus formula standard error: the starter block.

# %%
import numpy as np
from sklearn.datasets import load_diabetes

y = load_diabetes().target                    # the progression scores of the 442 patients of Example 2-10

def formula_se(x):
    """Return the standard error of the mean, s / sqrt(n), with s the sample standard deviation."""
    ...

def bootstrap_se(x, n_boot=2000, seed=42):
    """Return the standard deviation of the means of n_boot resamples of x drawn with replacement."""
    ...

print(y.shape, y[:5])

# %% [markdown]
# Coding exercise 9, Softmax and cross-entropy from scratch: the starter block.

# %%
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)

def cross_entropy(y_true, probs, eps=1e-15):
    """Return the mean of -log of the probability given to each true class, with probs clipped to [eps, 1 - eps]."""
    ...

def softmax(z):
    """Return the softmax of the score vector z, subtracting max(z) before exponentiating."""
    ...

print(X.shape, np.bincount(y))

# %% [markdown]
# Coding exercise 10, A two-sample t-test by hand: the starter block.

# %%
import numpy as np
from scipy import stats
from sklearn.datasets import load_iris

iris = load_iris()
petal_length = iris.data[:, 2]                # column 2 is petal length in centimeters

def welch_t(a, b):
    """Return Welch's t-statistic for the difference between the means of samples a and b."""
    ...

print(iris.data.shape, iris.target_names)

# %% [markdown]
# Coding exercise 11, Projection by the hat matrix: the starter block.

# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

X, y = load_diabetes(return_X_y=True)        # 442 patients: 10 baseline measurements and a progression score
X = np.column_stack([np.ones(len(y)), X])    # the intercept column first, as in 2.1.4

def hat_matrix(X):
    """Return H = X (X^T X)^{-1} X^T, using np.linalg.solve instead of an explicit inverse."""
    ...

def hat_matrix_qr(X):
    """Return the same projection built as Q Q^T from the reduced QR factorization of X."""
    ...

print(X.shape, y.shape)

# %% [markdown]
# Coding exercise 12, The pseudoinverse and the shortest solution: the starter block.

# %%
import numpy as np
from scipy.linalg import null_space
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)                  # the 442 patients of exercise 11
bmi_copy = X[:, 2] / 2.54                               # BMI is feature 2; the same column in new units
X = np.column_stack([np.ones(len(y)), X, bmi_copy])     # intercept, ten features, and the redundant copy

def pinv_from_svd(X, rtol=1e-10):
    """Return V Sigma+ U^T, inverting only the singular values above rtol times the largest."""
    ...

print(X.shape, np.linalg.matrix_rank(X))

# %% [markdown]
# Coding exercise 13, Least squares by QR, ridge by Cholesky: the starter block.

# %%
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)        # the 442 patients of exercise 11
X = StandardScaler().fit_transform(X)        # z-scores, as in exercise 6
y = y - y.mean()                             # a centered target needs no intercept column

def gram_schmidt_qr(X):
    """Return Q with orthonormal columns and upper-triangular R with X = Q R, by classical Gram-Schmidt."""
    ...

def back_substitute(R, c):
    """Solve the upper-triangular system R w = c, starting from the last row."""
    ...

print(X.shape, y.shape)
