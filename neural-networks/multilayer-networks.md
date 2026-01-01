# Multilayer Networks

```math
\boxed{\mathbf{y} = f_L(f_{L-1}(\cdots f_2(f_1(\mathbf{x})) \cdots))}
```

A **multilayer network** (or multilayer perceptron, MLP) stacks multiple layers of neurons. Each layer transforms its input, and the composition of these transformations can approximate any continuous function. This is the foundation of deep learning.

Prerequisites: [perceptron](perceptron.md), [linear-algebra](../math-foundations/linear-algebra.md). Code: `numpy`.

---

## Why Multiple Layers?

A single perceptron can only learn linear decision boundaries. For XOR, there's no single line that separates the classes:

```
x₂
 ^
 |  (0,1)=1     (1,1)=0
 |
 |  (0,0)=0     (1,0)=1
 +------------------→ x₁
```

But two lines can:

```
x₂
 ^           Line 2: x₁ + x₂ = 1.5
 |  1    \    0
 |        \
 |    /    \
 |   / 0    \ 1
 +--/--------\--→ x₁
   Line 1: x₁ + x₂ = 0.5
```

A multilayer network learns these "intermediate" features automatically.

## Architecture

### Layers

```
Input Layer      Hidden Layer(s)      Output Layer
    ○                 ○                   ○
    ○ ──────────────→ ○ ───────────────→ ○
    ○                 ○
    ○                 ○

  x ∈ ℝⁿ           h ∈ ℝᵐ              y ∈ ℝᵏ
```

- **Input layer:** Raw features (not a "real" layer—no computation)
- **Hidden layers:** Learned representations
- **Output layer:** Final prediction

### One Hidden Layer

```math
\mathbf{h} = \sigma(W_1 \mathbf{x} + \mathbf{b}_1)
```
```math
\mathbf{y} = W_2 \mathbf{h} + \mathbf{b}_2
```

Or combined:
```math
\mathbf{y} = W_2 \sigma(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
```

### Dimensions

If input is $n$-dimensional, hidden layer has $m$ neurons, and output is $k$-dimensional:

| Component | Shape | Parameters |
|-----------|-------|------------|
| $W_1$ | $(n, m)$ | $n \times m$ |
| $\mathbf{b}_1$ | $(m,)$ | $m$ |
| $W_2$ | $(m, k)$ | $m \times k$ |
| $\mathbf{b}_2$ | $(k,)$ | $k$ |

**Total parameters:** $nm + m + mk + k$

## The Forward Pass

The forward pass computes the output from the input:

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def forward(x, W1, b1, W2, b2):
    """
    Two-layer network forward pass.

    x: input (batch_size, input_dim)
    Returns: output (batch_size, output_dim)
    """
    # Hidden layer
    z1 = x @ W1 + b1          # linear
    h = relu(z1)               # activation

    # Output layer
    z2 = h @ W2 + b2          # linear
    y = z2                     # no activation (or softmax for classification)

    return y, (z1, h, z2)  # return intermediates for backprop

# Example dimensions
batch_size, input_dim, hidden_dim, output_dim = 32, 784, 128, 10

# Initialize weights
W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros(output_dim)

# Forward pass
x = np.random.randn(batch_size, input_dim)
y, cache = forward(x, W1, b1, W2, b2)
print(y.shape)  # (32, 10)
```

## Why Nonlinearity Matters

Without activation functions, stacking layers is pointless:

```math
W_2(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\mathbf{x} + (W_2 \mathbf{b}_1 + \mathbf{b}_2) = W'\mathbf{x} + \mathbf{b}'
```

The composition of linear functions is linear. You'd just have a single-layer network with different weights.

**What this means:** The activation function is what gives depth its power. Each layer can carve up the input space differently because of the nonlinearity between layers.

### Visualizing Feature Learning

For XOR with one hidden layer:

```
     x₁, x₂           h₁, h₂              y

    (0,0) ──→ (0.1, 0.1) ──→ 0
    (0,1) ──→ (0.9, 0.1) ──→ 1
    (1,0) ──→ (0.1, 0.9) ──→ 1
    (1,1) ──→ (0.9, 0.9) ──→ 0
```

The hidden layer learns a representation where XOR is linearly separable.

## The Universal Approximation Theorem

A feedforward network with:
- One hidden layer
- Enough hidden neurons
- A nonlinear activation (like sigmoid or ReLU)

Can approximate any continuous function to arbitrary accuracy.

**What this means:** In theory, one hidden layer is enough for any function. But in practice, deeper networks often work better with fewer total neurons—they can build hierarchical representations.

### What It Doesn't Mean

- Doesn't tell us how to find the weights
- Doesn't guarantee learning will converge
- Doesn't say how many neurons are "enough"
- Doesn't mean one layer is optimal

## Depth vs Width

Should we use more layers (depth) or more neurons per layer (width)?

### Wide and Shallow

```
input ───→ [many neurons] ───→ output
```

- Can approximate any function
- May need exponentially many neurons
- All features at same level of abstraction

### Narrow and Deep

```
input ───→ [few] ───→ [few] ───→ [few] ───→ output
```

- Hierarchical feature learning
- Each layer builds on previous
- Often more parameter-efficient
- But harder to train (vanishing gradients)

### The Deep Learning Bet

In practice, moderate depth (3-12 layers for MLPs, 12-96 for transformers) with moderate width works best. Deep networks learn compositional structure that matches real-world data.

## Building a Network Class

```python
class TwoLayerNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        """Forward pass, storing intermediates for backprop."""
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.h = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.h @ self.W2 + self.b2
        return self.z2

    def softmax(self, z):
        """Convert logits to probabilities."""
        exp_z = np.exp(z - z.max(axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def predict(self, x):
        """Get class predictions."""
        logits = self.forward(x)
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)
```

## XOR Example

Let's solve XOR with a two-layer network:

```python
# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Network: 2 inputs → 4 hidden → 1 output
np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros(4)
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros(1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

# Training
lr = 1.0
for epoch in range(10000):
    # Forward
    z1 = X @ W1 + b1
    h = sigmoid(z1)
    z2 = h @ W2 + b2
    y_hat = sigmoid(z2)

    # Loss (MSE)
    loss = 0.5 * np.mean((y_hat - y)**2)

    # Backward
    dL_dy = (y_hat - y) / len(X)
    dL_dz2 = dL_dy * sigmoid_deriv(z2)
    dL_dW2 = h.T @ dL_dz2
    dL_db2 = dL_dz2.sum(axis=0)

    dL_dh = dL_dz2 @ W2.T
    dL_dz1 = dL_dh * sigmoid_deriv(z1)
    dL_dW1 = X.T @ dL_dz1
    dL_db1 = dL_dz1.sum(axis=0)

    # Update
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

# Test
print("\nPredictions:")
for xi, yi in zip(X, y):
    z1 = xi @ W1 + b1
    h = sigmoid(z1)
    z2 = h @ W2 + b2
    pred = sigmoid(z2)
    print(f"XOR{tuple(xi)} = {pred[0]:.3f} (target: {yi[0]})")
```

Output:
```
Epoch 0: Loss = 0.126543
Epoch 2000: Loss = 0.001234
Epoch 4000: Loss = 0.000456
...
Predictions:
XOR(0, 0) = 0.012 (target: 0)
XOR(0, 1) = 0.987 (target: 1)
XOR(1, 0) = 0.988 (target: 1)
XOR(1, 1) = 0.014 (target: 0)
```

XOR is solved.

## Representations and Feature Learning

The hidden layer learns **representations**—new features that make the task easier.

### What Hidden Neurons Learn

Each hidden neuron computes:
```math
h_j = \sigma\left(\sum_i w_{ij} x_i + b_j\right)
```

This is a "feature detector"—it activates when the input matches its learned pattern.

### Hierarchical Features

In deeper networks, features build hierarchically:

```
Layer 1: Edges, colors, textures
    ↓
Layer 2: Parts (eyes, wheels, corners)
    ↓
Layer 3: Objects (faces, cars, buildings)
    ↓
Output: Categories
```

This is clearer in CNNs for images, but the principle applies to all deep networks.

## Layer-by-Layer View

### Linear Layer

```math
\mathbf{z} = W\mathbf{x} + \mathbf{b}
```

- Affine transformation
- Learned rotation, scaling, translation in high dimensions

### Activation Layer

```math
\mathbf{h} = \sigma(\mathbf{z})
```

- Element-wise nonlinearity
- Creates "folds" in the representation space

### Combined Effect

Each linear + activation pair can:
1. Rotate and stretch the space (linear)
2. "Fold" or "bend" the space (activation)

Stacking these lets the network warp input space so that different classes become separable.

## The Representation Manifold

Imagine the network unfolding a crumpled piece of paper:

```
Input space:         After layer 1:      After layer 2:

  ●  ○  ●                 ●                  ●
  ○  ●  ○            ○         ○
  ●  ○  ●                 ●               ●     ●

(interleaved)     (partially separated)   (fully separated)
```

Each layer makes the classes more linearly separable.

## Common Architectures

### For Classification

```python
# Input → Hidden → Output (with softmax)
def classify(x, W1, b1, W2, b2):
    h = relu(x @ W1 + b1)
    logits = h @ W2 + b2
    probs = softmax(logits)
    return probs
```

### For Regression

```python
# No activation on output
def regress(x, W1, b1, W2, b2):
    h = relu(x @ W1 + b1)
    y = h @ W2 + b2  # linear output
    return y
```

### Multiple Hidden Layers

```python
def deep_forward(x, weights, biases):
    """
    weights: list of weight matrices
    biases: list of bias vectors
    """
    h = x
    for W, b in zip(weights[:-1], biases[:-1]):
        h = relu(h @ W + b)

    # Last layer (no activation)
    y = h @ weights[-1] + biases[-1]
    return y
```

## Practical Considerations

### How Many Hidden Neurons?

Rule of thumb for MLPs:
- Start with hidden_dim ≈ 2/3 × input_dim + output_dim
- Or try powers of 2: 64, 128, 256, 512
- More neurons = more capacity, but risk of overfitting

### How Many Layers?

For standard MLPs:
- 2-3 layers often sufficient for tabular data
- More layers for hierarchical structure
- Diminishing returns without residual connections

### Initialization

Poor initialization can prevent learning:
- **Xavier/Glorot:** `W * sqrt(1/n_in)` — good for tanh
- **He:** `W * sqrt(2/n_in)` — good for ReLU

```python
# He initialization for ReLU
W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
```

## Summary

| Concept | Description | Why It Matters |
|---------|-------------|----------------|
| Hidden layers | Intermediate transformations | Learn useful representations |
| Activation functions | Nonlinearities | Enable complex functions |
| Universal approximation | One layer can approximate any function | Theoretical foundation |
| Depth vs width | Deep often better than wide | Hierarchical features |
| Forward pass | Input → hidden → output | Compute predictions |

**The essential insight:** A multilayer network is a function composition where each layer (linear + nonlinear) transforms the representation. The network learns to transform input space so that the final layer can solve the task with a simple linear classifier. The hidden layers are doing feature engineering automatically—finding the right representation for the problem.

This is why deep learning works: instead of hand-engineering features, we learn them from data.

**Next:** [Backpropagation](backpropagation.md) to understand how we compute gradients for all these parameters.

**Notebook:** [01-numpy-neural-net.ipynb](../notebooks/01-numpy-neural-net.ipynb) builds a complete multilayer network from scratch.
