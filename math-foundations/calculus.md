# Calculus for Machine Learning

```math
\boxed{\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}}
```

The **chain rule** is the heart of deep learning. It lets us compute how changing any weight affects the final loss, even through dozens of layers. Backpropagation is just the chain rule applied systematically.

Prerequisites: Basic algebra. Code: `numpy`.

---

## Derivatives: Rates of Change

### The Basic Idea

The **derivative** of $f(x)$ measures how fast $f$ changes as $x$ changes:

```math
f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
```

**What this means:** If you nudge $x$ by a tiny amount $h$, the output changes by approximately $f'(x) \cdot h$. The derivative is the "sensitivity" of output to input.

### Common Derivatives

| Function | Derivative | Notes |
|----------|------------|-------|
| $x^n$ | $nx^{n-1}$ | Power rule |
| $e^x$ | $e^x$ | The exponential is its own derivative |
| $\ln(x)$ | $1/x$ | Natural log |
| $\sin(x)$ | $\cos(x)$ | |
| $\cos(x)$ | $-\sin(x)$ | |

### Derivative Rules

**Sum rule:** $(f + g)' = f' + g'$

**Product rule:** $(fg)' = f'g + fg'$

**Quotient rule:** $(f/g)' = (f'g - fg')/g^2$

**Chain rule:** $(f(g(x)))' = f'(g(x)) \cdot g'(x)$

### Numerical Derivatives

We can approximate derivatives numerically:

```python
def numerical_derivative(f, x, h=1e-5):
    """Central difference approximation."""
    return (f(x + h) - f(x - h)) / (2 * h)

# Example: derivative of x^2 at x=3
f = lambda x: x**2
print(numerical_derivative(f, 3))  # ≈ 6.0 (exact: 2*3 = 6)
```

**What this means:** When analytical derivatives are hard to derive, we can always approximate them numerically. This is useful for checking your analytical gradients ("gradient checking").

## The Chain Rule

The **chain rule** is the most important concept for deep learning:

```math
\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)
```

Or in Leibniz notation:

```math
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
```

**What this means:** If $y$ depends on $u$, and $u$ depends on $x$, then the effect of $x$ on $y$ is the product of the intermediate effects. This chains together arbitrarily many layers.

### Example: Nested Functions

Let $y = (3x + 2)^4$. Let $u = 3x + 2$, so $y = u^4$.

```math
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 4u^3 \cdot 3 = 12(3x + 2)^3
```

### Example: Neural Network Layer

For a single neuron: $y = \sigma(wx + b)$ where $\sigma$ is the activation.

How does $y$ change with $w$?

```math
\frac{\partial y}{\partial w} = \sigma'(wx + b) \cdot x
```

The gradient depends on:
1. The derivative of the activation at the current input
2. The input $x$ itself

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)  # derivative of sigmoid

# Forward pass
x, w, b = 2.0, 0.5, 0.1
z = w * x + b
y = sigmoid(z)

# Gradient of y with respect to w
dy_dw = sigmoid_derivative(z) * x
print(f"dy/dw = {dy_dw:.4f}")
```

## Partial Derivatives

When $f$ has multiple inputs, we take **partial derivatives**—derivatives with respect to one variable while holding others constant.

```math
f(x, y) = x^2 + 3xy + y^2
```

```math
\frac{\partial f}{\partial x} = 2x + 3y \quad \text{(treat } y \text{ as constant)}
```

```math
\frac{\partial f}{\partial y} = 3x + 2y \quad \text{(treat } x \text{ as constant)}
```

**What this means:** In neural networks, we have thousands or millions of parameters. Partial derivatives tell us how changing *each one individually* affects the loss.

## The Gradient

The **gradient** collects all partial derivatives into a vector:

```math
\nabla f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}
```

### Key Properties

1. **Direction of steepest ascent:** The gradient points in the direction where $f$ increases fastest
2. **Magnitude:** $|\nabla f|$ is the rate of increase in that direction
3. **Perpendicular to level sets:** The gradient is perpendicular to contour lines

**What this means:** To minimize a loss function, we move *opposite* to the gradient direction—that's gradient descent.

### In Code

```python
def f(x):
    """f(x, y) = x^2 + y^2"""
    return x[0]**2 + x[1]**2

def gradient_f(x):
    """Gradient of f"""
    return np.array([2*x[0], 2*x[1]])

# At point (3, 4)
x = np.array([3.0, 4.0])
print(f"f(x) = {f(x)}")           # 25
print(f"∇f(x) = {gradient_f(x)}")  # [6, 8] - points toward origin
```

## Gradient Descent

To minimize a function $L(\theta)$:

```math
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
```

where $\eta$ is the **learning rate**.

**What this means:** Take a step in the direction opposite to the gradient (downhill). The learning rate controls step size—too large and you overshoot, too small and you're slow.

### Simple Example

```python
def loss(w):
    """Simple quadratic loss: L(w) = (w - 3)^2"""
    return (w - 3)**2

def grad_loss(w):
    """Gradient: dL/dw = 2(w - 3)"""
    return 2 * (w - 3)

# Gradient descent
w = 0.0  # starting point
lr = 0.1  # learning rate

for step in range(20):
    g = grad_loss(w)
    w = w - lr * g
    print(f"Step {step}: w = {w:.4f}, loss = {loss(w):.4f}")

# w converges to 3.0 (the minimum)
```

### Visualization of the Path

```
Loss
 ^
 |    *  <- starting point
 |   *
 |  *
 | *
 |*________ <- minimum at w=3
 +---------> w
```

## The Jacobian Matrix

When a function has multiple inputs AND multiple outputs, we need the **Jacobian**:

```math
\mathbf{y} = f(\mathbf{x}) \quad \text{where } \mathbf{x} \in \mathbb{R}^n, \mathbf{y} \in \mathbb{R}^m
```

```math
J = \begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{pmatrix}
```

Entry $(i, j)$ is $\frac{\partial y_i}{\partial x_j}$.

**What this means:** The Jacobian generalizes the derivative to vector functions. Each row is the gradient of one output. For a neural network layer, the Jacobian captures how all outputs change with all inputs.

### Example: Linear Layer

For $\mathbf{y} = W\mathbf{x}$:

```math
\frac{\partial y_i}{\partial x_j} = W_{ij}
```

So the Jacobian of a linear layer is just the weight matrix $W$ itself.

## Chain Rule with Vectors

For composed vector functions $\mathbf{z} = g(f(\mathbf{x}))$:

```math
\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}
```

This is matrix multiplication of Jacobians.

**What this means:** Backpropagation through a network is repeated Jacobian multiplication. Each layer contributes its Jacobian, and we multiply them together to get the gradient with respect to any earlier layer.

### Backprop in Matrix Form

For a network with layers:
```math
\mathbf{x} \xrightarrow{W_1} \mathbf{h}_1 \xrightarrow{\sigma} \mathbf{a}_1 \xrightarrow{W_2} \mathbf{h}_2 \xrightarrow{\sigma} \mathbf{y} \xrightarrow{L} \text{loss}
```

The gradient of loss with respect to $W_1$ involves:

```math
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{h}_2} \cdot \frac{\partial \mathbf{h}_2}{\partial \mathbf{a}_1} \cdot \frac{\partial \mathbf{a}_1}{\partial \mathbf{h}_1} \cdot \frac{\partial \mathbf{h}_1}{\partial W_1}
```

Each term is a local Jacobian. See [backpropagation](../neural-networks/backpropagation.md) for the full algorithm.

## Computing Gradients: Analytical vs Numerical

### Analytical Gradients

Derive the formula using calculus rules:

```python
def mse_loss(y_pred, y_true):
    """Mean squared error: L = mean((y_pred - y_true)^2)"""
    return np.mean((y_pred - y_true)**2)

def mse_gradient(y_pred, y_true):
    """Gradient: dL/dy_pred = 2(y_pred - y_true) / n"""
    n = len(y_pred)
    return 2 * (y_pred - y_true) / n
```

### Numerical Gradients (for checking)

```python
def numerical_gradient(f, x, h=1e-5):
    """Compute gradient numerically for any function."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Gradient check
y_pred = np.array([1.0, 2.0, 3.0])
y_true = np.array([1.1, 1.9, 3.2])

analytical = mse_gradient(y_pred, y_true)
numerical = numerical_gradient(lambda y: mse_loss(y, y_true), y_pred)

print(f"Analytical: {analytical}")
print(f"Numerical:  {numerical}")
print(f"Difference: {np.abs(analytical - numerical).max():.2e}")
```

**What this means:** Always check your analytical gradients against numerical ones during development. A bug in gradient computation will silently break training.

## Common Derivatives in ML

### Sigmoid

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

```math
\sigma'(x) = \sigma(x)(1 - \sigma(x))
```

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)
```

### ReLU

```math
\text{ReLU}(x) = \max(0, x)
```

```math
\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
```

```python
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)
```

### Softmax

```math
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
```

The Jacobian is more complex:

```math
\frac{\partial \text{softmax}_i}{\partial x_j} = \text{softmax}_i(\delta_{ij} - \text{softmax}_j)
```

where $\delta_{ij} = 1$ if $i = j$, else 0.

### Cross-Entropy Loss

Combined with softmax (the common case):

```math
L = -\sum_i y_i \log(\hat{y}_i)
```

For softmax output with one-hot target, the gradient simplifies beautifully:

```math
\frac{\partial L}{\partial x_i} = \hat{y}_i - y_i
```

**What this means:** The gradient is just "prediction minus target"—incredibly simple despite the complex-looking loss function. This is why cross-entropy + softmax is the standard classification setup.

## Taylor Series: Approximating Functions

Any smooth function can be approximated as a polynomial:

```math
f(x) \approx f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots
```

### First-Order Approximation

```math
f(x + \Delta x) \approx f(x) + f'(x) \Delta x
```

**What this means:** This is the foundation of gradient descent. We approximate the loss near our current point with a linear function, then step to minimize that approximation.

### Second-Order Approximation

```math
f(x + \Delta x) \approx f(x) + f'(x) \Delta x + \frac{1}{2} f''(x) (\Delta x)^2
```

The second derivative (Hessian for multiple variables) captures curvature. Second-order optimizers like Newton's method use this, but are expensive in high dimensions.

## Summary

| Concept | Formula | ML Application |
|---------|---------|----------------|
| Derivative | $\frac{df}{dx}$ | Sensitivity of output to input |
| Chain rule | $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$ | Backpropagation |
| Gradient | $\nabla f$ | Direction of steepest ascent |
| Gradient descent | $\theta \leftarrow \theta - \eta \nabla L$ | Training neural networks |
| Jacobian | $J_{ij} = \frac{\partial y_i}{\partial x_j}$ | Layer-wise derivatives |

**The essential insight:** Deep learning is optimization. Calculus gives us the tools to find how changing each parameter affects the loss. The chain rule lets us do this efficiently, layer by layer, even for networks with billions of parameters. The gradient points uphill, so we go downhill to minimize loss.

**Next:** [Probability](probability.md) for distributions, entropy, and why cross-entropy is the right loss function.

**Notebook:** [02-backprop-from-scratch.ipynb](../notebooks/02-backprop-from-scratch.ipynb) implements the chain rule to train a neural network.
