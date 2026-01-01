# Optimization

$$
\boxed{\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)}
$$

**Optimization** is how neural networks learn. Given a loss function that measures how wrong our predictions are, we adjust parameters to minimize that loss. The landscape of loss functions in deep learning is complex—high-dimensional, non-convex, full of saddle points—yet gradient descent works remarkably well.

Prerequisites: [calculus](calculus.md) (gradients), [backpropagation](../neural-networks/backpropagation.md). Code: `numpy`.

---

## The Optimization Problem

We want to find parameters $\theta$ that minimize a loss function $L$:

$$
\theta^* = \arg\min_\theta L(\theta)
$$

In deep learning:
- $\theta$ = millions to billions of weights and biases
- $L$ = some measure of prediction error (cross-entropy, MSE, etc.)
- $L$ is computed over a dataset $\mathcal{D} = \{(x_i, y_i)\}$

**What this means:** Training a neural network is an optimization problem. We're searching through a space with millions of dimensions for parameters that make good predictions.

## Gradient Descent

The gradient $\nabla L$ points in the direction of steepest increase. To minimize, we go the opposite direction:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

where $\eta$ is the **learning rate**.

### Intuition

Imagine standing on a mountainside in fog. You can't see the bottom, but you can feel which way is downhill. Take a step downhill. Repeat.

```
Loss
  ^
  |    *  ← Start here
  |   /|
  |  / |
  | /  ↓  ← Follow gradient
  |/   *
  |    |\
  |    | \
  |    |  *  ← Getting lower
  +----------→ θ
```

### In Code

```python
import numpy as np

def gradient_descent(f, grad_f, theta_init, learning_rate, n_steps):
    """
    Basic gradient descent.

    Args:
        f: Loss function
        grad_f: Gradient of loss function
        theta_init: Starting parameters
        learning_rate: Step size
        n_steps: Number of iterations
    """
    theta = theta_init.copy()
    history = [f(theta)]

    for _ in range(n_steps):
        gradient = grad_f(theta)
        theta = theta - learning_rate * gradient
        history.append(f(theta))

    return theta, history

# Example: minimize f(x) = x^2
def f(x):
    return x ** 2

def grad_f(x):
    return 2 * x

theta_opt, history = gradient_descent(f, grad_f, theta_init=np.array([5.0]),
                                       learning_rate=0.1, n_steps=20)
print(f"Optimal: {theta_opt[0]:.6f}")  # Should be near 0
```

## The Learning Rate

The learning rate $\eta$ controls step size:

| Learning Rate | Effect |
|---------------|--------|
| Too small | Slow convergence, may get stuck |
| Too large | Overshoots, oscillates, may diverge |
| Just right | Steady decrease toward minimum |

```python
def visualize_learning_rates():
    """Show effect of different learning rates."""
    import matplotlib.pyplot as plt

    rates = [0.01, 0.1, 0.5, 0.9]

    plt.figure(figsize=(10, 4))
    for lr in rates:
        _, history = gradient_descent(f, grad_f, np.array([5.0]), lr, 30)
        plt.plot(history, label=f'lr={lr}')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Effect of Learning Rate')
    plt.legend()
    plt.yscale('log')
    plt.show()
```

### Adaptive Learning Rates

Modern optimizers (Adam, AdaGrad) adapt the learning rate per-parameter. See [optimizers](../training/optimizers.md).

## Stochastic Gradient Descent (SGD)

Computing the gradient over the full dataset is expensive:

$$
\nabla L = \frac{1}{N} \sum_{i=1}^{N} \nabla L_i
$$

**SGD** uses random samples (mini-batches) instead:

$$
\nabla L \approx \frac{1}{B} \sum_{i \in \text{batch}} \nabla L_i
$$

### Why It Works

1. **Efficiency:** Process subset instead of full dataset
2. **Noise helps:** Stochastic gradients add noise that can escape local minima
3. **Memory:** Only need to hold one batch in memory

### Batch Size Trade-offs

| Batch Size | Gradient Quality | Speed | Generalization |
|------------|------------------|-------|----------------|
| 1 (online) | Very noisy | Slow | Often good |
| 32-256 | Reasonable noise | Fast | Good |
| 1000+ | Low noise | Very fast | May overfit |

```python
def sgd(f_batch, grad_f_batch, theta_init, X, y,
        learning_rate, batch_size, epochs):
    """
    Stochastic gradient descent.
    """
    theta = theta_init.copy()
    n = len(X)

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            gradient = grad_f_batch(theta, X_batch, y_batch)
            theta = theta - learning_rate * gradient

    return theta
```

## The Loss Landscape

### Convex vs Non-Convex

**Convex:** Bowl-shaped. Any local minimum is the global minimum.

$$
f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y)
$$

**Non-convex:** Multiple minima, saddle points, complex terrain.

Neural network loss surfaces are **highly non-convex**. Yet gradient descent finds good solutions.

### Local Minima

Points where the gradient is zero and Hessian is positive definite:

```
     *          *
    / \        / \
   /   \      /   \
  /     \____/     \
 /       local      \
/        min         \
         (not global)
```

**Empirical finding:** In high dimensions, most local minima are nearly as good as the global minimum. The bigger problem is saddle points.

### Saddle Points

Points where gradient is zero but Hessian has mixed signs:

```
        ↗ increases
       /
      /
  ___*___  ← saddle point: zero gradient
      \
       \
        ↘ decreases
```

In $d$ dimensions, a random critical point has probability $(1/2)^d$ of being a local minimum. For $d = 1000$, almost all critical points are saddle points.

**What this means:** Gradient descent slows down near saddle points (gradients are small), but doesn't truly get stuck because noise in SGD helps escape.

## Momentum

Momentum accelerates descent by accumulating past gradients:

$$
v_t = \beta v_{t-1} + \nabla L(\theta_t)
$$
$$
\theta_{t+1} = \theta_t - \eta v_t
$$

### Intuition

Like a ball rolling downhill—it builds up speed and can roll through small bumps.

```
Without momentum:     With momentum:
    *                     *
   /|                    /|
  / |                   / |
 /  | slow             /  | → → →
/   ↓                 /       fast!
```

### In Code

```python
def sgd_momentum(grad_f, theta_init, learning_rate, momentum, n_steps):
    """SGD with momentum."""
    theta = theta_init.copy()
    v = np.zeros_like(theta)

    for _ in range(n_steps):
        gradient = grad_f(theta)
        v = momentum * v + gradient
        theta = theta - learning_rate * v

    return theta
```

### Typical Values

- $\beta = 0.9$ is standard
- Higher momentum (0.99) for noisy gradients
- Lower momentum (0.5) early in training

## Nesterov Momentum

Look ahead before computing gradient:

$$
v_t = \beta v_{t-1} + \nabla L(\theta_t - \beta v_{t-1})
$$

**What this means:** Evaluate the gradient at where momentum would take us, not where we are. This provides a "correction" if momentum is taking us too far.

## Learning Rate Schedules

### Step Decay

Reduce learning rate at fixed intervals:

```python
def step_decay(epoch, initial_lr, drop=0.5, epochs_drop=10):
    return initial_lr * (drop ** (epoch // epochs_drop))
```

### Exponential Decay

$$
\eta_t = \eta_0 \cdot e^{-kt}
$$

### Cosine Annealing

Smooth decay following a cosine curve:

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))
$$

### Warmup

Start with small learning rate, gradually increase:

```python
def warmup_schedule(step, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr
```

**Why warmup?** Early gradients can be unstable. Small steps initially help establish reasonable parameter values.

## Gradient Clipping

Prevent exploding gradients by limiting their magnitude:

### By Value

```python
def clip_by_value(gradient, max_val):
    return np.clip(gradient, -max_val, max_val)
```

### By Norm

```python
def clip_by_norm(gradient, max_norm):
    norm = np.linalg.norm(gradient)
    if norm > max_norm:
        gradient = gradient * max_norm / norm
    return gradient
```

**When to use:** Essential for RNNs, helpful for transformers, especially early in training.

## Convergence

### For Convex Functions

With appropriate learning rate, gradient descent converges to the global minimum:

$$
L(\theta_T) - L(\theta^*) \leq O(1/T)
$$

### For Non-Convex Functions

We can only guarantee convergence to a **stationary point** (where gradient is zero):

$$
\min_{t \leq T} \|\nabla L(\theta_t)\|^2 \leq O(1/\sqrt{T})
$$

**What this means:** Gradient descent will find *some* point where the loss isn't changing much. Whether it's a good minimum depends on the loss landscape and initialization.

## Practical Tips

### Monitoring Training

```python
def train_with_monitoring(model, data, epochs):
    """Track training health."""
    for epoch in range(epochs):
        losses = []
        grad_norms = []

        for batch in data:
            loss = model.forward(batch)
            grads = model.backward()

            losses.append(loss)
            grad_norms.append(np.linalg.norm(grads))

            model.update(grads)

        avg_loss = np.mean(losses)
        avg_grad = np.mean(grad_norms)

        # Warning signs:
        if np.isnan(avg_loss):
            print("NaN loss! Lower learning rate.")
        if avg_grad > 100:
            print("Large gradients! Consider clipping.")
        if avg_grad < 1e-7:
            print("Vanishing gradients! Check architecture.")
```

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss NaN | Learning rate too high | Reduce lr, add gradient clipping |
| Loss plateaus early | Stuck in saddle point | Add momentum, reduce lr |
| Loss oscillates | Learning rate too high | Reduce lr |
| Training loss drops, val doesn't | Overfitting | Regularization, more data |
| Very slow progress | Learning rate too small | Increase lr, use Adam |

### Hyperparameter Starting Points

| Hyperparameter | Starting Value | Typical Range |
|----------------|----------------|---------------|
| Learning rate | 0.001 | 1e-5 to 0.1 |
| Batch size | 32-128 | 8 to 2048 |
| Momentum | 0.9 | 0.5 to 0.99 |
| Weight decay | 0.01 | 1e-5 to 0.1 |

## Second-Order Methods (Brief)

Gradient descent uses only first derivatives. **Second-order methods** use the Hessian (second derivatives):

$$
\theta_{t+1} = \theta_t - H^{-1} \nabla L
$$

### Newton's Method

Converges faster (quadratically vs linearly) but:
- Hessian is $O(d^2)$ parameters
- Inverting is $O(d^3)$ operations
- Impractical for millions of parameters

### Approximations

- **L-BFGS:** Approximates inverse Hessian, useful for smaller models
- **Natural gradient:** Uses Fisher information matrix
- **K-FAC:** Kronecker-factored approximation

In practice, first-order methods (Adam) dominate deep learning due to scalability.

## Summary

| Concept | Formula | Purpose |
|---------|---------|---------|
| Gradient descent | $\theta \leftarrow \theta - \eta \nabla L$ | Basic optimization |
| SGD | Sample mini-batches | Efficient, noisy |
| Momentum | $v \leftarrow \beta v + \nabla L$ | Accelerate, smooth |
| Learning rate | $\eta$ | Control step size |
| Gradient clipping | $\|g\| \leq c$ | Prevent explosion |

**The essential insight:** Optimization in deep learning works despite the non-convex landscape because: (1) SGD noise helps escape bad regions, (2) high-dimensional saddle points are more common than local minima, (3) most local minima are nearly as good as the global minimum, and (4) overparameterization creates many paths to good solutions.

The learning rate is the most important hyperparameter. Start with a reasonable value, use warmup, decay during training, and consider adaptive methods like Adam.

**Next:** [Optimizers](../training/optimizers.md) covers Adam, AdamW, and other adaptive methods.

**Notebook:** [04-optimization-landscape.ipynb](../notebooks/04-optimization-landscape.ipynb) visualizes loss surfaces and optimizer behavior.
