# Optimizers

```math
\boxed{\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t}
```

**Optimizers** are the algorithms that update model parameters to minimize loss. While basic gradient descent works, modern optimizers use adaptive learning rates, momentum, and other techniques to train faster and more reliably. Adam is the default choice for most deep learning, but understanding the alternatives helps you tune training.

Prerequisites: [optimization](../math-foundations/optimization.md), [backpropagation](../neural-networks/backpropagation.md). Code: `numpy`.

---

## From SGD to Adam

### The Evolution

```
SGD → SGD+Momentum → Adagrad → RMSprop → Adam → AdamW
        ↓               ↓          ↓
     Accumulate     Per-param    Running
      velocity      learning     average
                      rate       of v
```

Each step addresses a limitation of the previous:
- **Momentum:** Accelerate convergence, dampen oscillations
- **Adaptive rates:** Scale updates by parameter gradient history
- **Combination:** Momentum + adaptive rates = Adam

### The Optimizer Interface

```python
import numpy as np

class Optimizer:
    def __init__(self, params, lr=0.001):
        self.params = params  # List of parameter arrays
        self.lr = lr

    def step(self, grads):
        """Update parameters using gradients."""
        raise NotImplementedError

    def zero_grad(self):
        """Reset any accumulated state if needed."""
        pass
```

## SGD with Momentum

### Algorithm

Momentum accumulates a velocity that smooths out gradient noise:

```math
v_t = \beta v_{t-1} + g_t
```
```math
\theta_t = \theta_{t-1} - \eta v_t
```

```python
class SGD:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            param -= self.lr * self.velocities[i]
```

### Nesterov Momentum

Look ahead before computing gradient:

```math
v_t = \beta v_{t-1} + g(\theta_{t-1} - \eta \beta v_{t-1})
```

```python
class SGDNesterov:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p) for p in params]

    def step(self, grads):
        """
        Note: Assumes grads were computed at (param - lr*momentum*velocity)
        In practice, we use an equivalent formulation.
        """
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            v_prev = self.velocities[i].copy()
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            # Nesterov correction
            param -= self.lr * (self.momentum * self.velocities[i] + grad)
```

**What this means:** Nesterov momentum evaluates the gradient at the "lookahead" position, providing a correction if momentum is overshooting. Often converges slightly faster than standard momentum.

## Adagrad

### The Problem with Fixed Learning Rates

Different parameters may need different learning rates:
- Sparse features: rare updates → need larger steps
- Frequent features: many updates → can use smaller steps

### Algorithm

Accumulate squared gradients and scale by their inverse:

```math
G_t = G_{t-1} + g_t^2
```
```math
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t
```

```python
class Adagrad:
    def __init__(self, params, lr=0.01, eps=1e-8):
        self.params = params
        self.lr = lr
        self.eps = eps
        self.G = [np.zeros_like(p) for p in params]  # Accumulated squared gradients

    def step(self, grads):
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.G[i] += grad ** 2
            param -= self.lr * grad / (np.sqrt(self.G[i]) + self.eps)
```

### The Problem

$G_t$ only grows, so learning rate continually shrinks. Training can stall before convergence.

**What this means:** Adagrad was designed for sparse data (NLP, recommender systems) where some features appear rarely. For dense deep learning, the forever-decreasing learning rate is often too aggressive.

## RMSprop

### Algorithm

Use exponential moving average instead of sum:

```math
v_t = \beta v_{t-1} + (1 - \beta) g_t^2
```
```math
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon} g_t
```

```python
class RMSprop:
    def __init__(self, params, lr=0.001, beta=0.9, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * grad ** 2
            param -= self.lr * grad / (np.sqrt(self.v[i]) + self.eps)
```

**What this means:** RMSprop "forgets" old gradients, keeping the effective learning rate from shrinking to zero. The decay rate $\beta$ controls how much history matters.

## Adam

### Algorithm

Combine momentum (first moment) with adaptive rates (second moment):

```math
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
```
```math
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
```

Bias correction (crucial for early training):

```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
```

Update:

```math
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
```

```python
class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # Timestep
        self.m = [np.zeros_like(p) for p in params]  # First moment
        self.v = [np.zeros_like(p) for p in params]  # Second moment

    def step(self, grads):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

### Why Bias Correction?

At $t=1$ with $\beta_2=0.999$:

```math
v_1 = 0.001 \cdot g_1^2
```

This severely underestimates the true variance. Dividing by $(1 - 0.999^1) = 0.001$ corrects this.

```python
def visualize_bias_correction():
    """Show effect of bias correction in early training."""
    beta2 = 0.999
    steps = np.arange(1, 1001)

    uncorrected_scale = 1 - beta2 ** steps  # How much of true variance we've accumulated
    corrected_scale = np.ones_like(steps, dtype=float)  # After correction

    print(f"Step 1: uncorrected sees {uncorrected_scale[0]:.4f} of variance")
    print(f"Step 100: uncorrected sees {uncorrected_scale[99]:.4f} of variance")
    print(f"Step 1000: uncorrected sees {uncorrected_scale[999]:.4f} of variance")
```

### Default Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| $\eta$ (lr) | 0.001 | Learning rate |
| $\beta_1$ | 0.9 | Momentum decay |
| $\beta_2$ | 0.999 | Variance decay |
| $\epsilon$ | 1e-8 | Numerical stability |

**What this means:** Adam combines the best of momentum (smooth, directional) and adaptive learning rates (per-parameter scaling). It's robust across many problems with default settings.

## AdamW

### The Weight Decay Fix

Standard Adam applies weight decay incorrectly. L2 regularization adds $\lambda\theta$ to the gradient, which then gets scaled by Adam's adaptive rate. This means weight decay is applied inconsistently across parameters.

**Decoupled weight decay** applies it directly:

```math
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t - \eta \lambda \theta_{t-1}
```

```python
class AdamW:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update moments (without weight decay in gradient)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update with decoupled weight decay
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            param -= self.lr * self.weight_decay * param
```

**What this means:** AdamW is the standard for training transformers. The decoupled weight decay works as intended regardless of the adaptive learning rate, leading to better generalization.

## Learning Rate Schedules

### Warmup + Decay

Standard pattern for transformers:

```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.lr
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1

        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )

        self.optimizer.lr = lr
        return lr
```

### Learning Rate Finder

Find good learning rate by training with exponentially increasing rates:

```python
def find_learning_rate(model, data, optimizer, min_lr=1e-7, max_lr=10, steps=100):
    """
    Sweep learning rates to find good range.

    Returns:
        lrs: Learning rates tried
        losses: Loss at each learning rate
    """
    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), steps)
    losses = []

    # Save initial weights
    initial_weights = [p.copy() for p in model.params]

    for lr in lrs:
        optimizer.lr = lr

        # Train one step
        x, y = next(iter(data))
        y_pred = model.forward(x)
        loss = model.loss(y_pred, y)
        grads = model.backward(y)
        optimizer.step(grads)

        losses.append(loss)

        # Stop if loss explodes
        if loss > 10 * losses[0] or np.isnan(loss):
            break

    # Restore weights
    for p, w in zip(model.params, initial_weights):
        p[:] = w

    return lrs[:len(losses)], losses
```

**What this means:** Plot losses vs learning rate. Choose a learning rate where loss is decreasing rapidly, typically 1/10 of the rate where loss starts increasing.

## Gradient Accumulation

Train with effective batch sizes larger than memory allows:

```python
class GradientAccumulator:
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.accumulated_grads = None
        self.step_count = 0

    def backward(self, loss):
        # Compute gradients
        grads = self.model.backward(loss)

        # Accumulate
        if self.accumulated_grads is None:
            self.accumulated_grads = [g / self.accumulation_steps for g in grads]
        else:
            for i, g in enumerate(grads):
                self.accumulated_grads[i] += g / self.accumulation_steps

        self.step_count += 1

        # Update when we've accumulated enough
        if self.step_count % self.accumulation_steps == 0:
            self.optimizer.step(self.accumulated_grads)
            self.accumulated_grads = None
            return True  # Did update
        return False  # Accumulated only
```

## Comparison

### Optimizer Selection

| Optimizer | Best For | Typical LR |
|-----------|----------|------------|
| SGD + Momentum | CNNs, when tuning is possible | 0.01-0.1 |
| Adam | Quick experiments, NLP, general | 0.001 |
| AdamW | Transformers, best generalization | 0.0001-0.001 |
| RMSprop | RNNs, non-stationary problems | 0.001 |

### Complete Training Example

```python
def train_model(model, train_data, val_data, epochs=100):
    """Full training loop with best practices."""

    # AdamW optimizer
    optimizer = AdamW(model.params, lr=1e-4, weight_decay=0.01)

    # Cosine schedule with warmup
    total_steps = len(train_data) * epochs
    warmup_steps = total_steps // 10
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    # Gradient clipping
    max_grad_norm = 1.0

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train_mode()
        train_losses = []

        for batch in train_data:
            x, y = batch

            # Forward
            y_pred = model.forward(x)
            loss = model.loss(y_pred, y)
            train_losses.append(loss)

            # Backward
            grads = model.backward(y)

            # Gradient clipping
            grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
            if grad_norm > max_grad_norm:
                grads = [g * max_grad_norm / grad_norm for g in grads]

            # Update
            optimizer.step(grads)
            scheduler.step()

        # Validation
        model.eval_mode()
        val_losses = []
        for batch in val_data:
            x, y = batch
            y_pred = model.forward(x)
            val_losses.append(model.loss(y_pred, y))

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, "
              f"lr={optimizer.lr:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = [p.copy() for p in model.params]

    # Restore best weights
    for p, w in zip(model.params, best_weights):
        p[:] = w

    return model
```

## Summary

| Algorithm | Update Rule | Key Feature |
|-----------|-------------|-------------|
| SGD | $\theta - \eta g$ | Simple baseline |
| Momentum | $\theta - \eta v$, $v = \beta v + g$ | Smooth, accelerate |
| Adagrad | $\theta - \frac{\eta}{\sqrt{G}} g$ | Per-param rates |
| RMSprop | $\theta - \frac{\eta}{\sqrt{v}} g$, $v = \beta v + (1-\beta)g^2$ | Decaying average |
| Adam | $\theta - \frac{\eta}{\sqrt{\hat{v}}} \hat{m}$ | Momentum + adaptive |
| AdamW | Adam + decoupled decay | Best for transformers |

**The essential insight:** Optimizers adapt the learning process to the loss landscape. Momentum helps with narrow valleys and noisy gradients. Adaptive rates help when parameters have different sensitivities. Adam combines both. AdamW fixes weight decay for proper regularization. Start with AdamW, tune learning rate, add warmup and cosine decay for best results.

**Next:** [RNNs](../architectures/rnns.md) covers recurrent architectures for sequences.
