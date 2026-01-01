# Backpropagation

$$
\boxed{\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}}
$$

**Backpropagation** is the algorithm that makes deep learning possible. It computes gradients efficiently by applying the chain rule backwards through the network. Without it, training networks with millions of parameters would be computationally infeasible.

Prerequisites: [calculus](../math-foundations/calculus.md) (chain rule), [multilayer-networks](multilayer-networks.md). Code: `numpy`.

---

## The Problem

We want to minimize a loss function $L$ by adjusting weights $\mathbf{w}$.

Gradient descent requires:
$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial L}{\partial \mathbf{w}}
$$

For a network with millions of parameters, we need efficient gradient computation.

**Naive approach:** Numerically compute each gradient independently.
- Perturb each weight slightly
- Measure change in loss
- Cost: $O(P)$ forward passes per gradient step, where $P$ is parameter count

**Backpropagation:** Compute all gradients in one backward pass.
- Cost: $O(1)$ backward pass (same order as forward pass)

## The Chain Rule Revisited

For a composition $L = f(g(h(w)))$:

$$
\frac{\partial L}{\partial w} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial w}
$$

The key insight: we can compute these intermediate derivatives once and reuse them.

### Example: Simple Network

$$
x \xrightarrow{w} z=wx \xrightarrow{\sigma} a=\sigma(z) \xrightarrow{L} L=(a-y)^2
$$

Working backward:

$$
\frac{\partial L}{\partial a} = 2(a - y)
$$

$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} = 2(a-y) \cdot \sigma'(z)
$$

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w} = 2(a-y) \cdot \sigma'(z) \cdot x
$$

**What this means:** We compute gradients from output to input. Each step reuses the gradient computed in the previous step.

## Computational Graphs

We represent computations as **directed acyclic graphs** (DAGs):

```
    x ─────┐
           ├──→ [*] ──→ z ──→ [σ] ──→ a ──→ [L] ──→ loss
    w ─────┘                        y ───────┘
```

Nodes are operations, edges are data flow.

### Forward Pass

Traverse graph from inputs to outputs, computing values.

### Backward Pass

Traverse graph from outputs to inputs, computing gradients.

## The Backpropagation Algorithm

### Forward Pass

For each layer $l = 1, 2, \ldots, L$:
$$
\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
$$
$$
\mathbf{a}^{(l)} = \sigma^{(l)}(\mathbf{z}^{(l)})
$$

Store all intermediate values for the backward pass.

### Compute Loss

$$
L = \text{Loss}(\mathbf{a}^{(L)}, \mathbf{y})
$$

### Backward Pass

**Step 1:** Gradient at output
$$
\delta^{(L)} = \frac{\partial L}{\partial \mathbf{z}^{(L)}} = \frac{\partial L}{\partial \mathbf{a}^{(L)}} \odot \sigma'^{(L)}(\mathbf{z}^{(L)})
$$

**Step 2:** Propagate backwards for $l = L-1, L-2, \ldots, 1$:
$$
\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'^{(l)}(\mathbf{z}^{(l)})
$$

**Step 3:** Compute parameter gradients:
$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (\mathbf{a}^{(l-1)})^T
$$
$$
\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \delta^{(l)}
$$

### The Pattern

```
Forward:   x ──→ z₁ ──→ a₁ ──→ z₂ ──→ a₂ ──→ L
                ↓        ↓        ↓        ↓
Backward:      dz₁ ←── da₁ ←── dz₂ ←── da₂ ←── dL
```

Each backward step multiplies by the local gradient and passes the result upstream.

## Implementation: Two-Layer Network

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_backward(dout, z):
    """Gradient of ReLU."""
    return dout * (z > 0)

def softmax(z):
    exp_z = np.exp(z - z.max(axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, y):
    """Cross-entropy loss. y is integer class labels."""
    n = len(y)
    log_probs = -np.log(probs[np.arange(n), y] + 1e-10)
    return log_probs.mean()

class TwoLayerNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2/hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, X):
        """
        Forward pass. Store intermediates for backprop.

        X: (batch_size, input_dim)
        Returns: (batch_size, output_dim) logits
        """
        self.X = X

        # Layer 1
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2

        return self.z2  # logits

    def backward(self, y):
        """
        Backward pass. Compute gradients.

        y: (batch_size,) integer class labels
        Returns: dictionary of gradients
        """
        batch_size = len(y)

        # Softmax + cross-entropy gradient (combined for simplicity)
        probs = softmax(self.z2)
        dz2 = probs.copy()
        dz2[np.arange(batch_size), y] -= 1
        dz2 /= batch_size  # average over batch

        # Layer 2 gradients
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)

        # Backprop to layer 1
        da1 = dz2 @ self.W2.T
        dz1 = relu_backward(da1, self.z1)

        # Layer 1 gradients
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def update(self, grads, lr):
        """Update parameters with gradients."""
        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']
```

## Training Loop

```python
# Create network
net = TwoLayerNet(input_dim=784, hidden_dim=256, output_dim=10)

# Training loop
for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        # Forward pass
        logits = net.forward(X_batch)
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, y_batch)

        # Backward pass
        grads = net.backward(y_batch)

        # Update
        net.update(grads, lr=0.01)

    print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

## Understanding Each Gradient

### Output Layer: Softmax + Cross-Entropy

The gradient is remarkably simple:

$$
\frac{\partial L}{\partial z_i} = p_i - y_i
$$

where $p$ is the softmax output and $y$ is one-hot target.

**What this means:** The gradient is "prediction minus target." If we predicted 0.9 for the true class, gradient is 0.9 - 1 = -0.1 (small, correct direction). If we predicted 0.1, gradient is 0.1 - 1 = -0.9 (large correction needed).

### Hidden Layer: ReLU

For ReLU: $\sigma(z) = \max(0, z)$

$$
\frac{\partial a}{\partial z} = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}
$$

**What this means:** Gradients flow through for positive inputs, are blocked for negative inputs. This creates "dead" neurons that never update if they always output 0.

### Weight Gradient

$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (\mathbf{a}^{(l-1)})^T
$$

**What this means:** The weight gradient is an outer product of the error signal $\delta$ and the activations $a$. Weights connecting active neurons to high-error outputs get large updates.

### Bias Gradient

$$
\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \delta^{(l)}
$$

**What this means:** The bias gradient is just the error signal. No dependence on input—the bias is a learned constant.

## Gradient Flow Visualization

```
Forward Pass:
X ──[W1]──→ z1 ──[ReLU]──→ a1 ──[W2]──→ z2 ──[softmax]──→ p ──[CE]──→ L
    ↓           ↓              ↓           ↓
   store       store          store       store

Backward Pass:
   dW1 ←──── dz1 ←─[ReLU']─── da1 ←─[W2.T]── dz2 ←──[p-y]──── dL=1
   db1 ←──┘                   ↓
                             dW2
                             db2
```

## Gradient Checking

Always verify gradients numerically during development:

```python
def numerical_gradient(f, x, h=1e-5):
    """Compute gradient numerically."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + h
        f_plus = f(x)

        x[idx] = old_val - h
        f_minus = f(x)

        grad[idx] = (f_plus - f_minus) / (2 * h)
        x[idx] = old_val
        it.iternext()

    return grad

def check_gradient(net, X, y, param_name, epsilon=1e-5, tolerance=1e-5):
    """Check analytical gradient against numerical."""
    # Get analytical gradient
    net.forward(X)
    grads = net.backward(y)
    analytical = grads[param_name].flatten()

    # Get numerical gradient
    param = getattr(net, param_name)
    numerical = np.zeros_like(param.flatten())

    for i in range(len(numerical)):
        param_flat = param.flatten()
        old_val = param_flat[i]

        param_flat[i] = old_val + epsilon
        param.flat[i] = param_flat[i]
        net.forward(X)
        loss_plus = cross_entropy_loss(softmax(net.z2), y)

        param_flat[i] = old_val - epsilon
        param.flat[i] = param_flat[i]
        net.forward(X)
        loss_minus = cross_entropy_loss(softmax(net.z2), y)

        numerical[i] = (loss_plus - loss_minus) / (2 * epsilon)
        param.flat[i] = old_val

    # Compare
    diff = np.abs(analytical - numerical).max()
    rel_diff = diff / (np.abs(analytical).max() + 1e-10)

    print(f"{param_name}: max diff = {diff:.2e}, relative = {rel_diff:.2e}")
    return rel_diff < tolerance
```

## Common Issues

### Vanishing Gradients

In deep networks, gradients can shrink exponentially:

$$
\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \prod_{l=2}^{L} W^{(l)} \cdot \sigma'(\cdot)
$$

If each $W \cdot \sigma'$ has eigenvalues < 1, the product shrinks.

**Symptoms:**
- Early layers learn very slowly
- Loss plateaus
- Gradients are near zero

**Solutions:**
- ReLU activation (gradient is 1 for positive inputs)
- Residual connections (skip connections)
- Batch normalization
- Better initialization

### Exploding Gradients

Gradients can also grow exponentially.

**Symptoms:**
- Loss becomes NaN
- Weights blow up
- Gradients are huge

**Solutions:**
- Gradient clipping: `grad = np.clip(grad, -max_val, max_val)`
- Lower learning rate
- Better initialization

### Dead ReLU

Neurons that always output 0 never get gradients:

$$
\text{ReLU}'(z) = 0 \quad \text{if } z \leq 0
$$

**Solutions:**
- Leaky ReLU: small slope for negative inputs
- Careful initialization
- Lower learning rate initially

## Automatic Differentiation

Modern frameworks (PyTorch, JAX) implement autodiff:

```python
import torch

# Define parameters
W = torch.randn(10, 5, requires_grad=True)
x = torch.randn(32, 10)
y = torch.randint(0, 5, (32,))

# Forward pass
logits = x @ W
loss = torch.nn.functional.cross_entropy(logits, y)

# Backward pass - gradients computed automatically!
loss.backward()

# Gradients available
print(W.grad.shape)  # (10, 5)
```

**What this means:** You only write the forward pass. The framework builds the computational graph and computes gradients automatically. This is why implementing backprop by hand is valuable—you understand what the framework does.

## Backprop Through Time (BPTT)

For recurrent networks, unroll through time:

```
x₁ → h₁ → x₂ → h₂ → x₃ → h₃ → L
      ↓         ↓         ↓
      W         W         W  (shared weights)
```

Gradients accumulate across time steps:

$$
\frac{\partial L}{\partial W} = \sum_t \frac{\partial L}{\partial W_t}
$$

This is just backprop on the unrolled graph, but gradients must flow through many steps (vanishing/exploding gradient problem is severe).

## Summary

| Concept | Description | Formula |
|---------|-------------|---------|
| Forward pass | Compute outputs | $\mathbf{a}^{(l)} = \sigma(W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)})$ |
| Error signal | Gradient w.r.t. pre-activation | $\delta^{(l)} = \frac{\partial L}{\partial \mathbf{z}^{(l)}}$ |
| Backprop | Propagate error backward | $\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'$ |
| Weight gradient | Error × activation | $\nabla_W L = \delta \cdot \mathbf{a}^T$ |
| Gradient checking | Verify analytically | $(f(x+h) - f(x-h)) / 2h$ |

**The essential insight:** Backpropagation is the chain rule applied efficiently. By working backwards and reusing intermediate gradients, we compute all gradients in time proportional to one forward pass. This algorithmic efficiency—not the math—is what makes training large networks feasible.

Understanding backprop deeply helps debug training issues, design better architectures, and understand why certain techniques (residual connections, normalization) work.

**Next:** [Activation Functions](activation-functions.md) for the nonlinearities that make it all work.

**Notebook:** [02-backprop-from-scratch.ipynb](../notebooks/02-backprop-from-scratch.ipynb) implements backprop step by step.
