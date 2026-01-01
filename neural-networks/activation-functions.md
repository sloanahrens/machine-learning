# Activation Functions

```math
\boxed{a = \sigma(z) = \sigma(Wx + b)}
```

**Activation functions** introduce nonlinearity into neural networks. Without them, any network would collapse to a single linear transformation. The choice of activation affects training dynamics, gradient flow, and final performance.

Prerequisites: [calculus](../math-foundations/calculus.md), [backpropagation](backpropagation.md). Code: `numpy`.

---

## Why Nonlinearity?

A linear function of a linear function is still linear:

```math
f_2(f_1(x)) = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2)
```

No matter how many layers, without activation functions you just get:
```math
y = W'x + b'
```

**What this means:** Depth is useless without nonlinearity. Activations let each layer carve up input space differently, enabling complex decision boundaries.

## The Classic: Sigmoid

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

### Properties
- Output range: $(0, 1)$
- Smooth and differentiable
- Centered at 0.5

### Derivative

```math
\sigma'(z) = \sigma(z)(1 - \sigma(z))
```

Maximum derivative is 0.25 at $z = 0$.

### In Code

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Plot
z = np.linspace(-6, 6, 100)
# sigmoid(z) -> S-curve from 0 to 1
# sigmoid_derivative(z) -> bell curve, max at 0
```

### Problems

1. **Vanishing gradients:** For $|z| > 4$, $\sigma'(z) \approx 0$. Gradients vanish in deep networks.

2. **Not zero-centered:** Outputs are always positive. This makes gradient updates have the same sign, causing zig-zag optimization paths.

3. **Expensive:** Exponential computation.

**When to use:** Output layer for binary classification (probability interpretation). Rarely in hidden layers of modern networks.

## Hyperbolic Tangent (Tanh)

```math
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1
```

### Properties
- Output range: $(-1, 1)$
- Zero-centered
- Steeper than sigmoid

### Derivative

```math
\tanh'(z) = 1 - \tanh^2(z)
```

Maximum derivative is 1 at $z = 0$ (better than sigmoid's 0.25).

### In Code

```python
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2
```

### Problems

Still has vanishing gradients for large $|z|$, but less severe than sigmoid.

**When to use:** Historically common in RNNs/LSTMs. Still used in some architectures, but largely replaced by ReLU variants.

## ReLU: The Modern Default

```math
\text{ReLU}(z) = \max(0, z)
```

### Properties
- Output range: $[0, \infty)$
- Not differentiable at $z = 0$ (we use 0 there)
- Extremely simple

### Derivative

```math
\text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}
```

### In Code

```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```

### Why ReLU Works

1. **No vanishing gradient (for positive inputs):** Derivative is exactly 1.

2. **Sparse activation:** Many neurons output 0, creating efficient representations.

3. **Computational efficiency:** Just a comparison—no exponentials.

4. **Biological plausibility:** Neurons can't have negative firing rates.

### The Dead ReLU Problem

If a neuron's input is always negative, it outputs 0 forever:
- Gradient is 0, so weights never update
- The neuron is "dead"

**Causes:**
- Large negative bias
- Large learning rate causing weights to go negative
- Unlucky initialization

**How common:** In practice, 10-40% of neurons can die during training.

## Leaky ReLU

```math
\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}
```

where $\alpha$ is a small constant (typically 0.01).

### Properties
- Small gradient for negative inputs prevents dead neurons
- Still simple and efficient
- Output range: $(-\infty, \infty)$

### Derivative

```math
\text{LeakyReLU}'(z) = \begin{cases} 1 & z > 0 \\ \alpha & z \leq 0 \end{cases}
```

### In Code

```python
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)
```

### Parametric ReLU (PReLU)

Make $\alpha$ a learnable parameter:

```math
\text{PReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}
```

The network learns the optimal slope for negative values.

## ELU (Exponential Linear Unit)

```math
\text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases}
```

### Properties
- Smooth at $z = 0$
- Mean activations closer to zero
- More robust to noise

### Derivative

```math
\text{ELU}'(z) = \begin{cases} 1 & z > 0 \\ \text{ELU}(z) + \alpha & z \leq 0 \end{cases}
```

### In Code

```python
def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

def elu_derivative(z, alpha=1.0):
    return np.where(z > 0, 1, elu(z, alpha) + alpha)
```

## GELU (Gaussian Error Linear Unit)

```math
\text{GELU}(z) = z \cdot \Phi(z)
```

where $\Phi(z)$ is the CDF of the standard Gaussian.

### Approximation

```math
\text{GELU}(z) \approx 0.5z\left(1 + \tanh\left[\sqrt{2/\pi}(z + 0.044715z^3)\right]\right)
```

### Properties
- Smooth everywhere
- Weights inputs by their "probability" of being positive
- Used in BERT, GPT, and most modern transformers

### In Code

```python
def gelu(z):
    # Exact version using error function
    return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))

# Or with scipy
from scipy.special import erf
def gelu_exact(z):
    return 0.5 * z * (1 + erf(z / np.sqrt(2)))
```

### Why GELU for Transformers?

GELU is smoother than ReLU, which can help with:
- Gradient-based optimization
- Attention weight distributions
- Layer normalization interactions

The probabilistic interpretation: GELU randomly drops inputs close to zero, keeping large positive values and zeroing large negative values.

## Swish / SiLU

```math
\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}
```

### Properties
- Smooth
- Non-monotonic (slightly dips below 0 for negative $z$)
- Self-gated (output depends on input times sigmoid of input)

### Derivative

```math
\text{Swish}'(z) = \sigma(z) + z \cdot \sigma(z)(1 - \sigma(z)) = \sigma(z)(1 + z(1 - \sigma(z)))
```

### In Code

```python
def swish(z):
    return z * sigmoid(z)

def swish_derivative(z):
    s = sigmoid(z)
    return s + z * s * (1 - s)
```

Often slightly better than ReLU on deep networks.

## Softmax (For Outputs)

```math
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
```

Not a pointwise activation—it depends on all inputs.

### Properties
- Outputs sum to 1 (probability distribution)
- Exaggerates differences (larger values get more probability)
- Numerically unstable without max subtraction

### In Code

```python
def softmax(z):
    exp_z = np.exp(z - z.max(axis=-1, keepdims=True))  # stability
    return exp_z / exp_z.sum(axis=-1, keepdims=True)
```

### When to Use

Output layer for multi-class classification only.

## Comparison

### Visual Comparison

```
        |
    1   |          _______________  sigmoid
        |      ___/
    0   |-----/-------------------
        |
   -1   |_________________________
        -4   -2    0    2    4


        |
    1   |          _______________  tanh
        |      ___/
    0   |-----/-------------------
        |___/
   -1   |_________________________
        -4   -2    0    2    4


        |              /          ReLU
        |             /
    0   |------------/
        |
        |_________________________
        -4   -2    0    2    4


        |              /          Leaky ReLU
        |             /
    0   |__,--,-----/
        |  `-'
        |_________________________
        -4   -2    0    2    4
```

### Performance Comparison

| Activation | Gradient Flow | Computation | Typical Use |
|------------|---------------|-------------|-------------|
| Sigmoid | Poor (vanishes) | Expensive | Binary output |
| Tanh | Moderate | Expensive | RNNs, some outputs |
| ReLU | Good (positive) | Fast | Default hidden layers |
| Leaky ReLU | Good (all) | Fast | When dead ReLU is problem |
| GELU | Good | Moderate | Transformers |
| Swish | Good | Moderate | Deep networks |

## Choosing an Activation

### For Hidden Layers

1. **Start with ReLU** — fast, works well, easy to debug
2. **If dead ReLU is a problem** → Leaky ReLU
3. **For transformers** → GELU (or Swish)
4. **For deep networks** → GELU, Swish, or ELU

### For Output Layers

| Task | Activation | Output |
|------|------------|--------|
| Binary classification | Sigmoid | Probability in $(0, 1)$ |
| Multi-class classification | Softmax | Probability distribution |
| Multi-label classification | Sigmoid (element-wise) | Independent probabilities |
| Regression | None (linear) | Unbounded real values |
| Regression (bounded) | Sigmoid or Tanh | Bounded values |

## Practical Tips

### Initialization Matters

Different activations need different initialization:

| Activation | Initialization | Why |
|------------|----------------|-----|
| Sigmoid/Tanh | Xavier/Glorot | Preserves variance |
| ReLU | He | Accounts for ReLU halving variance |
| GELU/Swish | He or Xavier | Either works |

```python
# Xavier initialization
W = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)

# He initialization
W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
```

### Watch for Saturation

Monitor activation statistics during training:

```python
def activation_stats(activations):
    """Check for saturation."""
    dead = (activations == 0).mean()
    saturated = (np.abs(activations) > 0.99).mean()  # for sigmoid/tanh
    print(f"Dead: {dead:.1%}, Saturated: {saturated:.1%}")
```

### Gradient Magnitude

Healthy gradients should have reasonable magnitude:

```python
def gradient_stats(gradients):
    """Check for vanishing/exploding gradients."""
    mean = np.abs(gradients).mean()
    max_val = np.abs(gradients).max()
    print(f"Mean |grad|: {mean:.2e}, Max: {max_val:.2e}")
```

## Summary

| Activation | Formula | Derivative | Range |
|------------|---------|------------|-------|
| Sigmoid | $1/(1+e^{-z})$ | $\sigma(1-\sigma)$ | $(0,1)$ |
| Tanh | $(e^z-e^{-z})/(e^z+e^{-z})$ | $1-\tanh^2$ | $(-1,1)$ |
| ReLU | $\max(0,z)$ | $\mathbf{1}_{z>0}$ | $[0,\infty)$ |
| Leaky ReLU | $\max(\alpha z, z)$ | $1$ or $\alpha$ | $(-\infty,\infty)$ |
| GELU | $z\Phi(z)$ | (complex) | $\approx(-0.17,\infty)$ |
| Swish | $z\sigma(z)$ | $\sigma + z\sigma(1-\sigma)$ | $\approx(-0.28,\infty)$ |

**The essential insight:** Activations are what make deep learning "deep." Without nonlinearity, layers collapse into one. ReLU revolutionized deep learning by solving the vanishing gradient problem for positive inputs. Modern activations like GELU provide smoother gradients for transformer-scale models. The choice matters less than you'd think—most reasonable activations work, but matching to your architecture and task can give a few percentage points.

**Next:** [Attention](../transformers/attention.md) to see how transformers use these building blocks.

**Notebook:** [03-activations-visualized.ipynb](../notebooks/03-activations-visualized.ipynb) compares activations visually and on real tasks.
