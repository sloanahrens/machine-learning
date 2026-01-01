# The Perceptron

$$
\boxed{y = \sigma(w \cdot x + b)}
$$

The **perceptron** is a single artificial neuron—the simplest possible neural network. It takes inputs, weights them, adds a bias, and passes the result through an activation function. Understanding the perceptron unlocks everything that comes after.

Prerequisites: [linear-algebra](../math-foundations/linear-algebra.md) (dot products), [calculus](../math-foundations/calculus.md) (derivatives). Code: `numpy`.

---

## The Biological Inspiration

Real neurons:
1. Receive signals from other neurons through **dendrites**
2. Sum the incoming signals in the cell body (**soma**)
3. If the total exceeds a threshold, fire a signal along the **axon**

The perceptron mimics this: weighted sum → threshold → output.

**What this means:** We're not simulating biology—neurons are far more complex. But the core abstraction (weighted inputs → nonlinear output) is powerful enough to learn useful functions.

## Anatomy of a Perceptron

```
    x₁ ───w₁───┐
               │
    x₂ ───w₂───┼──[Σ + b]───[σ]───→ y
               │
    x₃ ───w₃───┘
```

Components:
- **Inputs** $x_1, x_2, \ldots, x_n$: The features
- **Weights** $w_1, w_2, \ldots, w_n$: Learned importance of each input
- **Bias** $b$: Learned threshold adjustment
- **Activation** $\sigma$: Nonlinear function

### The Math

**Step 1: Weighted sum (linear combination)**
$$
z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b = \mathbf{w} \cdot \mathbf{x} + b
$$

**Step 2: Activation**
$$
y = \sigma(z)
$$

### In Code

```python
import numpy as np

def perceptron(x, w, b, activation=lambda z: z):
    """
    x: input vector (n,)
    w: weight vector (n,)
    b: bias (scalar)
    activation: activation function
    """
    z = np.dot(w, x) + b  # weighted sum
    y = activation(z)      # apply nonlinearity
    return y

# Example
x = np.array([1.0, 2.0, 3.0])
w = np.array([0.5, -0.5, 0.2])
b = 0.1

# Linear (no activation)
print(perceptron(x, w, b))  # 0.5*1 - 0.5*2 + 0.2*3 + 0.1 = 0.2
```

## Activation Functions

Without an activation function, a perceptron is just a linear function. The activation introduces **nonlinearity**—the key to learning complex patterns.

### Step Function (Original Perceptron)

$$
\sigma(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}
$$

Binary output: fire or don't fire.

**Problem:** Not differentiable—can't use gradient descent.

### Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Smooth output between 0 and 1. Interpretable as probability.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Sigmoid squashes any input to (0, 1)
print(sigmoid(-10))  # ≈ 0
print(sigmoid(0))    # 0.5
print(sigmoid(10))   # ≈ 1
```

### ReLU

$$
\sigma(z) = \max(0, z)
$$

Simple, fast, and works well in practice. See [activation-functions](activation-functions.md) for more.

```python
def relu(z):
    return np.maximum(0, z)
```

## The Perceptron as a Linear Classifier

With a step function, the perceptron divides input space with a **hyperplane**:

$$
\mathbf{w} \cdot \mathbf{x} + b = 0
$$

Points on one side → class 1, points on the other → class 0.

### Example: AND Gate

| $x_1$ | $x_2$ | AND |
|-------|-------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

A perceptron can learn this:

```python
# AND gate - handpicked weights
w = np.array([1.0, 1.0])
b = -1.5  # threshold

def step(z):
    return (z > 0).astype(float)

# Test all inputs
for x1 in [0, 1]:
    for x2 in [0, 1]:
        x = np.array([x1, x2])
        y = perceptron(x, w, b, step)
        print(f"AND({x1}, {x2}) = {int(y)}")

# Output:
# AND(0, 0) = 0  (0 + 0 - 1.5 = -1.5 < 0)
# AND(0, 1) = 0  (0 + 1 - 1.5 = -0.5 < 0)
# AND(1, 0) = 0  (1 + 0 - 1.5 = -0.5 < 0)
# AND(1, 1) = 1  (1 + 1 - 1.5 = 0.5 > 0)
```

### Visualizing the Decision Boundary

```
x₂
 ^
 |     1,1 ← class 1 (above line)
 |    /
 |   /  w·x + b = 0
 |  /   (decision boundary)
 | /
 +---0,0---0,1---→ x₁
        ↑
    class 0 (below line)
```

The line $x_1 + x_2 - 1.5 = 0$ separates the classes.

## The XOR Problem

Not all problems are **linearly separable**:

| $x_1$ | $x_2$ | XOR |
|-------|-------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

```
x₂
 ^
 |  0,1(1)     1,1(0)
 |
 |  0,0(0)     1,0(1)
 +------------------→ x₁
```

No single line can separate the 1s from the 0s.

**What this means:** A single perceptron can only solve linearly separable problems. For XOR (and most real problems), we need multiple layers—see [multilayer-networks](multilayer-networks.md).

## Training: The Perceptron Learning Algorithm

For the binary perceptron, there's a simple update rule:

```
For each training example (x, y_true):
    y_pred = predict(x)
    if y_pred != y_true:
        w = w + (y_true - y_pred) * x
        b = b + (y_true - y_pred)
```

**What this means:**
- If correct: do nothing
- If predicted 0 but should be 1: add $x$ to weights (move toward $x$)
- If predicted 1 but should be 0: subtract $x$ from weights (move away from $x$)

### In Code

```python
def train_perceptron(X, y, epochs=100):
    """
    X: inputs (n_samples, n_features)
    y: labels (n_samples,) - binary 0/1
    """
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z = np.dot(w, xi) + b
            y_pred = 1 if z > 0 else 0

            if y_pred != yi:
                error = yi - y_pred  # +1 or -1
                w = w + error * xi
                b = b + error

    return w, b

# Train on AND data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND labels

w, b = train_perceptron(X, y)
print(f"Learned: w = {w}, b = {b}")
```

### Convergence Theorem

**If the data is linearly separable, the perceptron algorithm converges in finite time.**

This is a remarkable guarantee—the algorithm will find a separating hyperplane if one exists. But if the data isn't linearly separable, it will never converge.

## Gradient Descent Training

For differentiable activations (sigmoid, ReLU), we use gradient descent:

### Forward Pass

$$
z = \mathbf{w} \cdot \mathbf{x} + b
$$
$$
\hat{y} = \sigma(z)
$$

### Loss Function

Mean squared error:
$$
L = \frac{1}{2}(\hat{y} - y)^2
$$

### Backward Pass (Gradients)

Using the chain rule:

$$
\frac{\partial L}{\partial \hat{y}} = \hat{y} - y
$$

$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \sigma'(z)
$$

$$
\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial z} \cdot x_i
$$

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}
$$

### Update Rule

$$
w_i \leftarrow w_i - \eta \frac{\partial L}{\partial w_i}
$$
$$
b \leftarrow b - \eta \frac{\partial L}{\partial b}
$$

### In Code

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def train_perceptron_gradient(X, y, epochs=1000, lr=0.1):
    """Train perceptron with gradient descent."""
    n_samples, n_features = X.shape
    w = np.random.randn(n_features) * 0.01
    b = 0.0

    for epoch in range(epochs):
        total_loss = 0

        for xi, yi in zip(X, y):
            # Forward pass
            z = np.dot(w, xi) + b
            y_hat = sigmoid(z)

            # Loss
            loss = 0.5 * (y_hat - yi)**2
            total_loss += loss

            # Backward pass
            dL_dy = y_hat - yi
            dL_dz = dL_dy * sigmoid_derivative(z)
            dL_dw = dL_dz * xi
            dL_db = dL_dz

            # Update
            w = w - lr * dL_dw
            b = b - lr * dL_db

        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    return w, b

# Train
w, b = train_perceptron_gradient(X, y)

# Test
for xi, yi in zip(X, y):
    y_hat = sigmoid(np.dot(w, xi) + b)
    print(f"Input: {xi}, True: {yi}, Predicted: {y_hat:.3f}")
```

## From Perceptron to Neural Network

A single perceptron is limited. But we can:

1. **Stack layers:** Output of one perceptron feeds into another
2. **Use multiple perceptrons per layer:** Each learns different features
3. **Combine results:** Later layers integrate earlier features

This gives us **multilayer networks**—see the [next document](multilayer-networks.md).

### Preview: Two-Layer Network

```
Input      Hidden Layer     Output
  x₁ ────┬───→ h₁ ───┬
         ├───→ h₂ ───┼───→ y
  x₂ ────┴───→ h₃ ───┘
```

Each hidden unit is a perceptron. Together, they can solve XOR and much more.

## Geometric Intuition

### What the Weights Mean

- $w_i > 0$: Feature $x_i$ pushes toward class 1
- $w_i < 0$: Feature $x_i$ pushes toward class 0
- $|w_i|$ large: Feature $x_i$ is important
- $|w_i|$ small: Feature $x_i$ is less important

### What the Bias Means

The bias shifts the decision boundary:
- $b > 0$: More likely to predict class 1
- $b < 0$: More likely to predict class 0

It's like a "default tendency" before seeing the input.

## Practical Considerations

### Feature Scaling

Gradient descent works better when features have similar scales:

```python
# Standardize features to zero mean, unit variance
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
```

### Weight Initialization

Starting with all zeros is bad—all gradients would be the same. Use small random values:

```python
w = np.random.randn(n_features) * 0.01
```

### Learning Rate

- Too large: Overshoots, oscillates, may diverge
- Too small: Converges slowly
- Just right: Smooth decrease in loss

Typical starting values: 0.01 to 0.1

## Summary

| Concept | Formula | Purpose |
|---------|---------|---------|
| Weighted sum | $z = \mathbf{w} \cdot \mathbf{x} + b$ | Combine inputs |
| Activation | $y = \sigma(z)$ | Introduce nonlinearity |
| Decision boundary | $\mathbf{w} \cdot \mathbf{x} + b = 0$ | Separate classes |
| Gradient | $\frac{\partial L}{\partial w_i}$ | Direction to update |
| Update rule | $w \leftarrow w - \eta \nabla L$ | Learn from errors |

**The essential insight:** A perceptron computes a linear function of its inputs, then applies a nonlinearity. It can only learn linearly separable functions—but that's actually a lot. The key limitation is solved by stacking multiple layers, letting the network learn the right *features* to separate.

The perceptron's weighted sum $\mathbf{w} \cdot \mathbf{x}$ is the same dot product that powers attention in transformers. The activation function is what lets networks approximate complex functions. Everything builds from here.

**Next:** [Multilayer Networks](multilayer-networks.md) to break the linear separability barrier.

**Notebook:** [01-numpy-neural-net.ipynb](../notebooks/01-numpy-neural-net.ipynb) implements a full perceptron from scratch.
