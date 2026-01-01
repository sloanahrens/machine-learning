# Regularization

```math
\boxed{L_{\text{total}} = L_{\text{data}} + \lambda R(\theta)}
```

**Regularization** prevents overfitting by constraining what the model can learn. Neural networks have millions of parameters—far more than needed to memorize training data. Regularization techniques add bias toward simpler solutions, improving generalization to unseen data.

Prerequisites: [multilayer networks](multilayer-networks.md), [optimization](../math-foundations/optimization.md). Code: `numpy`.

---

## The Overfitting Problem

### Bias-Variance Tradeoff

Model error = Bias² + Variance + Irreducible noise

| Term | Meaning | Symptom |
|------|---------|---------|
| High bias | Model too simple | Underfits (high training error) |
| High variance | Model too complex | Overfits (training << validation) |

```
Error
  ^
  |   \           /
  |    \  total  /
  |     \       /
  |      \_____/  ← sweet spot
  |    bias  variance
  +------------------→ Model complexity
```

### Why Neural Networks Overfit

With $d$ parameters and $n$ training examples:
- If $d \gg n$: model can memorize training data
- Modern networks often have $d > 10^6$ and $n < 10^6$
- Overparameterization enables memorization

```python
import numpy as np

def train_test_gap_demo():
    """Show overfitting with model complexity."""
    np.random.seed(42)

    # True function: simple quadratic + noise
    n = 50
    x = np.linspace(0, 1, n)
    y_true = 0.5 + 2*x - 3*x**2
    y = y_true + np.random.randn(n) * 0.2

    # Fit polynomials of increasing degree
    degrees = [1, 3, 15]

    for d in degrees:
        # Polynomial features
        X = np.column_stack([x**i for i in range(d+1)])
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

        # Training error
        y_pred = X @ coeffs
        train_err = np.mean((y - y_pred)**2)

        # Generalization (using true function as proxy)
        gen_err = np.mean((y_true - y_pred)**2)

        print(f"Degree {d:2d}: Train MSE = {train_err:.4f}, Gen MSE = {gen_err:.4f}")
```

**What this means:** A degree-15 polynomial can pass through all 50 points perfectly (zero training error) but generalizes terribly. Regularization prevents this.

## Weight Decay (L2 Regularization)

### The Idea

Add penalty on weight magnitude to the loss:

```math
L_{\text{total}} = L_{\text{data}} + \frac{\lambda}{2} \|\theta\|^2_2 = L_{\text{data}} + \frac{\lambda}{2} \sum_i \theta_i^2
```

### Effect on Optimization

The gradient becomes:

```math
\nabla L_{\text{total}} = \nabla L_{\text{data}} + \lambda \theta
```

Update rule:

```math
\theta_{t+1} = \theta_t - \eta(\nabla L_{\text{data}} + \lambda \theta_t) = (1 - \eta\lambda)\theta_t - \eta \nabla L_{\text{data}}
```

Weights decay toward zero each step (hence the name).

```python
def train_with_weight_decay(model, X, y, learning_rate, weight_decay, epochs):
    """Training loop with L2 regularization."""
    for epoch in range(epochs):
        # Forward pass
        y_pred = model.forward(X)
        data_loss = np.mean((y_pred - y)**2)

        # L2 penalty
        l2_loss = 0
        for param in model.params:
            l2_loss += np.sum(param**2)
        l2_loss *= weight_decay / 2

        total_loss = data_loss + l2_loss

        # Backward pass (gradients include regularization)
        model.backward(y)
        for param, grad in zip(model.params, model.grads):
            grad += weight_decay * param  # Add regularization gradient

        # Update
        for param, grad in zip(model.params, model.grads):
            param -= learning_rate * grad

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
```

### Why It Works

**Bayesian interpretation:** L2 regularization is equivalent to a Gaussian prior on weights:

```math
p(\theta) = \mathcal{N}(0, \sigma^2 I), \quad \lambda = \frac{1}{2\sigma^2}
```

**Geometric interpretation:** Constrains weights to lie in a ball of radius $\sqrt{c/\lambda}$.

```
       weight space
      ___________
     /           \
    |   feasible  |
    |    region   |
    |     (ball)  |
     \___________/

Larger λ → smaller ball → simpler model
```

### Typical Values

| Context | Weight Decay |
|---------|--------------|
| Computer vision | 1e-4 to 1e-3 |
| NLP/Transformers | 0.01 to 0.1 |
| Fine-tuning | 0.01 |

## L1 Regularization

### Definition

```math
L_{\text{total}} = L_{\text{data}} + \lambda \|\theta\|_1 = L_{\text{data}} + \lambda \sum_i |\theta_i|
```

### L1 vs L2

| Property | L1 (Lasso) | L2 (Ridge) |
|----------|------------|------------|
| Penalty | $\sum \|\theta_i\|$ | $\sum \theta_i^2$ |
| Gradient | $\lambda \cdot \text{sign}(\theta)$ | $2\lambda\theta$ |
| Effect | Sparse weights (many zeros) | Small weights |
| Use case | Feature selection | General regularization |

```python
def l1_gradient(theta, lambda_reg):
    """Gradient of L1 penalty (subgradient at 0)."""
    return lambda_reg * np.sign(theta)

def l2_gradient(theta, lambda_reg):
    """Gradient of L2 penalty."""
    return lambda_reg * theta

# L1 pushes weights to exactly zero
# L2 shrinks weights but rarely to exactly zero
```

**What this means:** L1 is useful when you expect few features to matter—it automatically selects important ones. L2 is preferred when all features contribute somewhat.

### Elastic Net

Combine both:

```math
L = L_{\text{data}} + \lambda_1 \|\theta\|_1 + \lambda_2 \|\theta\|_2^2
```

Gets sparsity of L1 with the stability of L2.

## Dropout

### The Technique

During training, randomly set activations to zero:

```math
h' = m \odot h, \quad m_i \sim \text{Bernoulli}(p)
```

Then scale by $1/(1-p)$ to maintain expected value.

```python
class Dropout:
    def __init__(self, p=0.5):
        """
        Args:
            p: Probability of keeping a unit (not dropping)
        """
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, self.p, x.shape) / self.p
            return x * self.mask
        else:
            return x  # No dropout at test time

    def backward(self, dout):
        return dout * self.mask
```

### Why It Works

**Ensemble interpretation:** Dropout trains an exponential number of subnetworks. At test time, we approximate the ensemble average.

```
Full network:     o---o---o
                   \ / \ /
                    o---o
                   / \ / \
                  o---o---o

Dropout samples: Each forward pass uses a different subnetwork
```

**Co-adaptation prevention:** Neurons can't rely on specific other neurons, forcing redundant representations.

### Placement

Typical pattern:

```python
class MLPWithDropout:
    def __init__(self, sizes, dropout_p=0.5):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:  # Not after last layer
                self.layers.append(ReLU())
                self.layers.append(Dropout(dropout_p))

    def forward(self, x, training=True):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, training)
            else:
                x = layer.forward(x)
        return x
```

### Dropout Rate Guidelines

| Layer Type | Typical Rate |
|------------|--------------|
| Input layer | 0.8-0.9 (keep) |
| Hidden layers | 0.5 (keep) |
| Convolutional | 0.7-0.9 (keep) |
| Transformers | Often use 0.1 drop rate |

**What this means:** Dropout with p=0.5 means each neuron has 50% chance of being active during any training step. This adds noise that acts as regularization.

## Batch Normalization

### The Technique

Normalize activations to zero mean and unit variance, then apply learnable scale and shift:

```math
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
```
```math
y_i = \gamma \hat{x}_i + \beta
```

```python
class BatchNorm:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.momentum = momentum
        self.eps = eps

        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):
        if training:
            # Batch statistics
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            # Normalize
            self.x_norm = (x - mu) / np.sqrt(var + self.eps)

            # Save for backward
            self.mu, self.var, self.x = mu, var, x
        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * self.x_norm + self.beta

    def backward(self, dout):
        N = dout.shape[0]

        # Gradients for gamma and beta
        self.dgamma = np.sum(dout * self.x_norm, axis=0)
        self.dbeta = np.sum(dout, axis=0)

        # Gradient for input
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (self.x - self.mu) * -0.5 * (self.var + self.eps)**(-1.5), axis=0)
        dmu = np.sum(dx_norm * -1/np.sqrt(self.var + self.eps), axis=0)
        dx = dx_norm / np.sqrt(self.var + self.eps) + dvar * 2 * (self.x - self.mu) / N + dmu / N

        return dx
```

### Why It Regularizes

1. **Internal covariate shift:** Stabilizes distribution of layer inputs
2. **Noise injection:** Batch statistics add stochasticity during training
3. **Smoother loss landscape:** Enables higher learning rates

### Layer Norm vs Batch Norm

| Normalization | Statistics Over | Use Case |
|---------------|-----------------|----------|
| Batch norm | Batch dimension | CNNs, fixed batch size |
| Layer norm | Feature dimension | RNNs, Transformers, variable batch |

```python
class LayerNorm:
    def __init__(self, num_features, eps=1e-5):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.eps = eps

    def forward(self, x):
        # Normalize over features (last dimension)
        mu = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mu) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

**What this means:** Batch norm normalizes across the batch (good when batch is representative). Layer norm normalizes within each sample (good for sequences of varying length). Transformers use layer norm.

## Early Stopping

### The Technique

Stop training when validation loss stops improving:

```python
def train_with_early_stopping(model, train_data, val_data, patience=10):
    """
    Stop training when validation loss plateaus.

    Args:
        patience: Number of epochs to wait for improvement
    """
    best_val_loss = float('inf')
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(1000):  # Max epochs
        # Train one epoch
        train_loss = train_epoch(model, train_data)

        # Validate
        val_loss = evaluate(model, val_data)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy_weights(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            restore_weights(model, best_weights)
            break

    return model
```

### Why It Works

Training trajectory:

```
Loss
  ^
  |   train
  |    ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
  |   _____________________
  |  /
  | /     val
  |/       ↓ ↓ ↓     ↗ ↗
  |        _______/
  |              ↑
  |         stop here
  +-------------------------→ Epoch
```

Early stopping finds the point of minimum validation error, before the model overfits.

**What this means:** Early stopping is implicit regularization—it limits the effective capacity by constraining training time rather than model size.

## Data Augmentation

### The Idea

Artificially expand training set with transformed examples:

```python
def augment_image(image):
    """Random augmentations for image data."""
    augmented = image.copy()

    # Random horizontal flip
    if np.random.random() > 0.5:
        augmented = np.fliplr(augmented)

    # Random rotation (-15 to 15 degrees)
    angle = np.random.uniform(-15, 15)
    # (rotation code would go here)

    # Random brightness
    factor = np.random.uniform(0.8, 1.2)
    augmented = np.clip(augmented * factor, 0, 1)

    # Random crop and resize
    # (cropping code would go here)

    return augmented
```

### Common Augmentations

**Images:**
- Flips, rotations, crops
- Color jitter, brightness, contrast
- Mixup: blend two images and labels
- Cutout: mask random rectangles

**Text:**
- Synonym replacement
- Random deletion/insertion
- Back-translation
- Token masking (BERT-style)

**Why it works:** Augmentation encodes invariances—if a flipped cat is still a cat, the model should learn flip-invariance rather than memorizing specific orientations.

## Putting It Together

### Modern Recipe

```python
class RegularizedNetwork:
    """Example combining multiple regularization techniques."""

    def __init__(self, sizes, dropout_p=0.1, weight_decay=0.01):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                self.layers.append(LayerNorm(sizes[i+1]))
                self.layers.append(ReLU())
                self.layers.append(Dropout(1 - dropout_p))

        self.weight_decay = weight_decay

    def forward(self, x, training=True):
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                if isinstance(layer, Dropout):
                    x = layer.forward(x, training)
                else:
                    x = layer.forward(x)
        return x

    def get_l2_loss(self):
        """L2 regularization loss."""
        l2 = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                l2 += np.sum(layer.W ** 2)
        return 0.5 * self.weight_decay * l2
```

### What To Use When

| Situation | Recommended |
|-----------|-------------|
| Small dataset | Strong regularization, augmentation |
| Transformers | Dropout 0.1, weight decay 0.01-0.1, layer norm |
| CNNs | Batch norm, dropout, augmentation |
| Overfitting | Increase dropout, weight decay, add augmentation |
| Underfitting | Reduce regularization, increase model size |

## Summary

| Technique | Mechanism | Typical Settings |
|-----------|-----------|------------------|
| Weight decay | Penalize large weights | $\lambda \sim 10^{-4}$ to $10^{-1}$ |
| Dropout | Random neuron masking | $p \sim 0.1$ to $0.5$ |
| Batch norm | Normalize activations | Momentum 0.1 |
| Layer norm | Per-sample normalization | Default for transformers |
| Early stopping | Stop before overfit | Patience 5-20 epochs |
| Data augmentation | Artificial examples | Task-specific |

**The essential insight:** Regularization is about encoding prior knowledge—that simpler models generalize better, that representations should be redundant, that flip-invariance matters. Different techniques encode different priors. The art is matching regularization to your problem structure.

**Next:** [Loss Functions](../training/loss-functions.md) covers the objectives we optimize.
