# Linear Algebra for Machine Learning

```math
\boxed{y = Wx + b}
```

Nearly every operation in deep learning is a **linear transformation** followed by a nonlinearity. Matrix multiplication is the fundamental operation—understanding it geometrically and computationally unlocks everything from embeddings to attention.

Prerequisites: Basic algebra (variables, equations). Code: `numpy`.

---

## Vectors

### What Is a Vector?

A **vector** is an ordered list of numbers:

```math
\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}
```

In ML, vectors represent:
- **Features**: A data point with $n$ measurements
- **Embeddings**: A word, token, or concept as a point in high-dimensional space
- **Weights**: Parameters connecting neurons

**What this means:** A vector is just a list of numbers, but thinking of it as a *point* or *direction* in space builds intuition. A 768-dimensional embedding is a point in 768-dimensional space—hard to visualize, but the math works the same as 2D or 3D.

### Vector Operations

**Addition:** Element-wise

```math
\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \end{pmatrix}
```

**Scalar multiplication:** Scale each element

```math
c\mathbf{v} = \begin{pmatrix} cv_1 \\ cv_2 \\ \vdots \end{pmatrix}
```

**What this means:** Adding vectors combines their effects. Scaling a vector makes it longer (or shorter, or flips it). These operations preserve "linearity"—the core property that makes linear algebra tractable.

### In Code

```python
import numpy as np

# Vectors are 1D arrays
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Operations
print(x + y)      # [5, 7, 9]
print(2 * x)      # [2, 4, 6]
print(x.shape)    # (3,)
```

## The Dot Product

The **dot product** of two vectors:

```math
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n
```

This single number measures **similarity**.

### Geometric Interpretation

```math
\mathbf{u} \cdot \mathbf{v} = |\mathbf{u}| |\mathbf{v}| \cos\theta
```

where $\theta$ is the angle between the vectors.

| Dot Product | Meaning |
|-------------|---------|
| Large positive | Vectors point in similar directions |
| Zero | Vectors are perpendicular (orthogonal) |
| Large negative | Vectors point in opposite directions |

**What this means:** The dot product is how attention works. When a query and key have a large dot product, they're "similar" in embedding space, so the attention weight is high. When they're orthogonal, the weight approaches zero.

### Cosine Similarity

To compare direction regardless of magnitude:

```math
\text{cosine similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{|\mathbf{u}| |\mathbf{v}|}
```

This is the dot product of *normalized* vectors (unit length).

### In Code

```python
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Dot product
dot = np.dot(u, v)  # or u @ v, or (u * v).sum()
print(dot)  # 32

# Cosine similarity
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

print(cosine_similarity(u, v))  # 0.974
```

## Matrices

### What Is a Matrix?

A **matrix** is a 2D array of numbers:

```math
A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}
```

An $m \times n$ matrix has $m$ rows and $n$ columns.

In ML, matrices represent:
- **Weight matrices**: Transformations between layers
- **Data batches**: Multiple samples stacked together
- **Embeddings**: Vocabulary of vectors

### Matrix-Vector Multiplication

```math
\mathbf{y} = A\mathbf{x}
```

Each element of the output is a dot product:

```math
y_i = \sum_j A_{ij} x_j = (\text{row } i \text{ of } A) \cdot \mathbf{x}
```

**What this means:** Matrix-vector multiplication applies the same operation to get each output element: take a row, dot with the input. Each row of $A$ defines a "detector"—what pattern in the input activates that output?

### Geometric View

A matrix is a **linear transformation**. It can:
- **Rotate** (orthogonal matrices)
- **Scale** (diagonal matrices)
- **Shear** (general matrices)
- **Project** (reduce dimensions)

The columns of $A$ show where the standard basis vectors go.

### In Code

```python
# Matrix-vector multiplication
A = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2 matrix
x = np.array([1, 2])                     # 2-element vector

y = A @ x  # or np.dot(A, x)
print(y)        # [5, 11, 17]
print(y.shape)  # (3,) - input was 2D, output is 3D
```

## Matrix-Matrix Multiplication

For $A$ ($m \times n$) and $B$ ($n \times p$), the product $C = AB$ is $m \times p$:

```math
C_{ij} = \sum_{k=1}^n A_{ik} B_{kj}
```

**Key insight:** Matrix multiplication is *composing* linear transformations. If $A$ transforms a vector, and $B$ transforms it further, then $AB$ does both in one step.

### The Inner Dimensions Must Match

```math
\underbrace{A}_{m \times \mathbf{n}} \times \underbrace{B}_{\mathbf{n} \times p} = \underbrace{C}_{m \times p}
```

The inner dimensions ($n$) must be equal.

### In Code

```python
A = np.array([[1, 2], [3, 4]])      # 2x2
B = np.array([[5, 6], [7, 8]])      # 2x2

C = A @ B
print(C)
# [[19, 22],
#  [43, 50]]

# Matrix multiplication is NOT commutative
print(A @ B)  # different from
print(B @ A)  # this!
```

### Batched Operations

In deep learning, we often process multiple samples at once:

```python
# X: batch of inputs, shape (batch_size, input_dim)
# W: weight matrix, shape (input_dim, output_dim)
# Y: batch of outputs, shape (batch_size, output_dim)

X = np.random.randn(32, 768)   # 32 samples, 768 features
W = np.random.randn(768, 256)  # project to 256 dims

Y = X @ W  # (32, 256) - all 32 samples transformed at once
```

**What this means:** A single matrix multiplication transforms an entire batch. This is why GPUs are fast for deep learning—they're optimized for exactly this operation.

## Transpose

The **transpose** flips a matrix across its diagonal:

```math
(A^T)_{ij} = A_{ji}
```

Rows become columns, columns become rows.

### Properties

- $(A^T)^T = A$
- $(AB)^T = B^T A^T$ (note the order reversal)
- $(A + B)^T = A^T + B^T$

### In Code

```python
A = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
print(A.T)  # or np.transpose(A)
# [[1, 4],
#  [2, 5],
#  [3, 6]]  # 3x2
```

### Why It Matters

The dot product can be written as matrix multiplication:

```math
\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v}
```

This is why attention uses $QK^T$—each row of $Q$ dots with each row of $K$.

## Norms: Measuring Size

The **norm** measures the "length" or magnitude of a vector.

### L2 Norm (Euclidean)

```math
|\mathbf{x}|_2 = \sqrt{\sum_i x_i^2}
```

This is the usual notion of distance/length.

### L1 Norm (Manhattan)

```math
|\mathbf{x}|_1 = \sum_i |x_i|
```

Sum of absolute values. Used in regularization (encourages sparsity).

### In Code

```python
x = np.array([3, 4])

l2_norm = np.linalg.norm(x)       # 5.0 (Euclidean distance)
l1_norm = np.linalg.norm(x, 1)    # 7.0 (sum of absolute values)

# Normalize to unit length
x_normalized = x / np.linalg.norm(x)
print(np.linalg.norm(x_normalized))  # 1.0
```

## Eigenvalues and Eigenvectors

An **eigenvector** of matrix $A$ is a vector that only gets scaled (not rotated) when $A$ is applied:

```math
\boxed{A\mathbf{v} = \lambda\mathbf{v}}
```

where $\lambda$ is the **eigenvalue**.

**What this means:** Most vectors get both scaled and rotated by a matrix. Eigenvectors are special—they point along the "natural axes" of the transformation. The eigenvalue tells you the scaling factor along that axis.

### Finding Eigenvalues

Eigenvalues satisfy:

```math
\det(A - \lambda I) = 0
```

For a 2×2 matrix, this gives a quadratic equation.

### Why It Matters in ML

1. **PCA**: Principal components are eigenvectors of the covariance matrix
2. **Spectral analysis**: Understanding network behavior
3. **Initialization**: Eigenvalue distributions affect training dynamics

### In Code

```python
A = np.array([[4, 2], [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)   # [5., 2.]
print(eigenvectors)  # columns are eigenvectors

# Verify: A @ v = lambda * v
v = eigenvectors[:, 0]
lam = eigenvalues[0]
print(np.allclose(A @ v, lam * v))  # True
```

## Broadcasting

NumPy (and PyTorch) **broadcast** arrays of different shapes:

```python
# Add a vector to each row of a matrix
X = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
b = np.array([10, 20])                   # 2

print(X + b)
# [[11, 22],
#  [13, 24],
#  [15, 26]]
```

**Rules:**
1. Compare shapes from right to left
2. Dimensions match if equal OR one of them is 1
3. Missing dimensions are treated as 1

**What this means:** Broadcasting lets you write `y = X @ W + b` without explicitly repeating `b` for each sample. The bias is automatically added to each row.

## Practical Patterns

### The Linear Layer

The fundamental building block:

```math
\mathbf{y} = W\mathbf{x} + \mathbf{b}
```

```python
def linear(x, W, b):
    """
    x: input, shape (batch_size, in_features)
    W: weights, shape (in_features, out_features)
    b: bias, shape (out_features,)
    """
    return x @ W + b

# Example
batch_size, in_features, out_features = 32, 768, 256
x = np.random.randn(batch_size, in_features)
W = np.random.randn(in_features, out_features)
b = np.random.randn(out_features)

y = linear(x, W, b)  # (32, 256)
```

### Embeddings as Matrix Lookup

An embedding layer is a matrix where row $i$ is the vector for token $i$:

```python
vocab_size, embed_dim = 10000, 768
embeddings = np.random.randn(vocab_size, embed_dim)

# Get embeddings for tokens [5, 42, 100]
token_ids = np.array([5, 42, 100])
token_vectors = embeddings[token_ids]  # (3, 768)
```

**What this means:** "Looking up" an embedding is just indexing into a matrix. During training, backprop updates the rows that were accessed.

### Attention as Matrix Operations

The attention mechanism is pure linear algebra:

```python
def attention_scores(Q, K):
    """
    Q: queries, shape (seq_len, d_k)
    K: keys, shape (seq_len, d_k)
    Returns: scores, shape (seq_len, seq_len)
    """
    return Q @ K.T / np.sqrt(K.shape[-1])
```

Each entry $(i, j)$ is the dot product of query $i$ with key $j$, measuring similarity.

## Summary

| Concept | Formula | Code | ML Usage |
|---------|---------|------|----------|
| Dot product | $\sum u_i v_i$ | `u @ v` | Attention scores |
| Matrix-vector | $y_i = \sum_j A_{ij} x_j$ | `A @ x` | Linear layer |
| Matrix-matrix | $C_{ij} = \sum_k A_{ik} B_{kj}$ | `A @ B` | Batched operations |
| Transpose | $(A^T)_{ij} = A_{ji}$ | `A.T` | Attention $QK^T$ |
| Norm | $\sqrt{\sum x_i^2}$ | `np.linalg.norm(x)` | Regularization, normalization |
| Eigenvalues | $A\mathbf{v} = \lambda\mathbf{v}$ | `np.linalg.eig(A)` | PCA, analysis |

**The essential insight:** Neural networks are compositions of linear transformations (matrix multiplications) with nonlinearities in between. Understanding matrices geometrically—as transformations that rotate, scale, and project—builds intuition for what networks do to data. The dot product as similarity measure is the key to attention.

**Next:** [Calculus](calculus.md) for derivatives, gradients, and the chain rule that makes backpropagation work.

**Notebook:** [01-numpy-neural-net.ipynb](../notebooks/01-numpy-neural-net.ipynb) applies these operations to build a neural network from scratch.
