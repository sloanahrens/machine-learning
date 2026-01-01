# Loss Functions

$$
\boxed{L = -\sum_{c} y_c \log \hat{y}_c}
$$

**Loss functions** measure how wrong our predictions are. The choice of loss function defines what "good" means for a model—it encodes our objective and shapes the entire optimization landscape. Different tasks require different losses, and understanding why helps you choose (or design) the right one.

Prerequisites: [information theory](../math-foundations/information-theory.md), [backpropagation](../neural-networks/backpropagation.md). Code: `numpy`.

---

## The Role of Loss Functions

### What Loss Functions Do

1. **Measure prediction quality:** Scalar summarizing how wrong the model is
2. **Provide gradients:** Enable optimization via backpropagation
3. **Encode objectives:** Define what behavior we want

```python
import numpy as np

def training_loop(model, loss_fn, data, learning_rate):
    """Generic training loop."""
    for x, y in data:
        # Forward
        y_pred = model.forward(x)
        loss = loss_fn.forward(y_pred, y)

        # Backward
        dL = loss_fn.backward()  # Gradient of loss
        model.backward(dL)

        # Update
        model.update(learning_rate)
```

### Properties of Good Loss Functions

1. **Differentiable:** Must compute gradients for optimization
2. **Bounded below:** Has a minimum (can't go to $-\infty$)
3. **Informative gradients:** Provides useful direction even far from optimum
4. **Matches the task:** Measures what we actually care about

## Mean Squared Error (MSE)

### Definition

$$
L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

```python
class MSELoss:
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Predictions [batch_size, ...]
            y_true: Ground truth [batch_size, ...]
        """
        self.y_pred = y_pred
        self.y_true = y_true
        self.diff = y_pred - y_true
        return np.mean(self.diff ** 2)

    def backward(self):
        """Gradient w.r.t. y_pred."""
        n = self.y_pred.size
        return 2 * self.diff / n
```

### Properties

| Property | MSE |
|----------|-----|
| Range | $[0, \infty)$ |
| Gradient | $\frac{2}{N}(\hat{y} - y)$ |
| Sensitive to | Outliers (squared penalty) |
| Probabilistic view | Gaussian likelihood |

### When to Use

- **Regression tasks:** Predicting continuous values
- **When outliers matter:** Large errors should be penalized heavily
- **Gaussian assumption:** Output distribution is roughly normal

```python
# Example: Linear regression
def linear_regression_demo():
    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    w_true = np.array([1.5, -2.0, 0.5])
    y = X @ w_true + np.random.randn(100) * 0.1

    # Train with MSE
    w = np.zeros(3)
    lr = 0.01
    loss_fn = MSELoss()

    for epoch in range(100):
        y_pred = X @ w
        loss = loss_fn.forward(y_pred, y)
        grad = loss_fn.backward()
        w -= lr * (X.T @ grad)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")

    print(f"Learned weights: {w}")
    print(f"True weights: {w_true}")
```

**What this means:** MSE penalizes large errors quadratically—an error of 2 costs 4x more than an error of 1. This makes the model prioritize reducing large errors first.

## Mean Absolute Error (MAE)

### Definition

$$
L = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

```python
class MAELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.diff = y_pred - y_true
        return np.mean(np.abs(self.diff))

    def backward(self):
        n = self.y_pred.size
        return np.sign(self.diff) / n
```

### MSE vs MAE

| Property | MSE | MAE |
|----------|-----|-----|
| Outlier sensitivity | High | Low |
| Gradient magnitude | Proportional to error | Constant |
| Median vs mean | Estimates mean | Estimates median |
| Optimization | Smooth | Non-smooth at 0 |

```python
def compare_mse_mae():
    """Show robustness of MAE to outliers."""
    # Data with outlier
    y_true = np.array([1.0, 2.0, 3.0, 100.0])  # 100 is outlier
    y_pred = np.array([1.5, 2.5, 3.5, 4.0])

    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))

    print(f"MSE: {mse:.1f}")  # Dominated by outlier
    print(f"MAE: {mae:.1f}")  # More robust
```

### Huber Loss (Smooth L1)

Combines MSE (small errors) and MAE (large errors):

$$
L_\delta(a) = \begin{cases}
\frac{1}{2}a^2 & \text{if } |a| \leq \delta \\
\delta(|a| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

```python
class HuberLoss:
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, y_pred, y_true):
        self.diff = y_pred - y_true
        abs_diff = np.abs(self.diff)

        quadratic = np.minimum(abs_diff, self.delta)
        linear = abs_diff - quadratic

        return np.mean(0.5 * quadratic**2 + self.delta * linear)

    def backward(self):
        n = self.diff.size
        grad = np.where(np.abs(self.diff) <= self.delta,
                        self.diff,
                        self.delta * np.sign(self.diff))
        return grad / n
```

**What this means:** Huber loss gives you the best of both worlds—smooth gradients for small errors (like MSE) and bounded gradients for outliers (like MAE).

## Cross-Entropy Loss

### Binary Cross-Entropy

For binary classification with sigmoid output:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i)]
$$

```python
class BinaryCrossEntropyLoss:
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Sigmoid probabilities [batch_size]
            y_true: Binary labels {0, 1} [batch_size]
        """
        self.y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        self.y_true = y_true

        loss = -(y_true * np.log(self.y_pred) +
                 (1 - y_true) * np.log(1 - self.y_pred))
        return np.mean(loss)

    def backward(self):
        n = self.y_pred.size
        grad = -(self.y_true / self.y_pred -
                 (1 - self.y_true) / (1 - self.y_pred))
        return grad / n
```

### Categorical Cross-Entropy

For multi-class classification with softmax output:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c}
$$

For one-hot labels, simplifies to:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{y}_{i, c_i}
$$

where $c_i$ is the true class for sample $i$.

```python
class CrossEntropyLoss:
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Softmax probabilities [batch_size, num_classes]
            y_true: One-hot labels [batch_size, num_classes]
        """
        self.y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        self.y_true = y_true
        self.batch_size = y_pred.shape[0]

        # Sum over classes, mean over batch
        loss = -np.sum(y_true * np.log(self.y_pred), axis=-1)
        return np.mean(loss)

    def backward(self):
        return -(self.y_true / self.y_pred) / self.batch_size
```

### Softmax + Cross-Entropy Combined

For numerical stability, combine softmax and cross-entropy:

```python
class SoftmaxCrossEntropyLoss:
    """Numerically stable softmax + cross-entropy."""

    def forward(self, logits, y_true):
        """
        Args:
            logits: Raw scores before softmax [batch_size, num_classes]
            y_true: One-hot labels [batch_size, num_classes]
        """
        self.y_true = y_true
        self.batch_size = logits.shape[0]

        # Stable softmax
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_shifted = np.exp(shifted)
        self.y_pred = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

        # Cross-entropy
        log_probs = shifted - np.log(np.sum(exp_shifted, axis=-1, keepdims=True))
        loss = -np.sum(y_true * log_probs, axis=-1)
        return np.mean(loss)

    def backward(self):
        """Gradient simplifies to (y_pred - y_true)."""
        return (self.y_pred - self.y_true) / self.batch_size
```

**What this means:** The combined gradient is beautifully simple: $\nabla_{\text{logits}} L = \hat{y} - y$. The probability error directly becomes the gradient.

### Label Smoothing

Prevent overconfidence by softening targets:

```python
class LabelSmoothingCrossEntropy:
    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing

    def forward(self, logits, y_true):
        num_classes = logits.shape[-1]

        # Smooth labels
        y_smooth = (1 - self.smoothing) * y_true + self.smoothing / num_classes

        # Use smoothed labels for cross-entropy
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))

        return -np.mean(np.sum(y_smooth * log_probs, axis=-1))
```

## Specialized Loss Functions

### Focal Loss

For class imbalance—down-weight easy examples:

$$
L = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

```python
class FocalLoss:
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, epsilon=1e-15):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # p_t = p if y=1 else 1-p
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        # Focal term
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        loss = -alpha_t * focal_weight * np.log(p_t)
        return np.mean(loss)
```

**What this means:** Focal loss reduces the contribution of easy examples (high $p_t$), focusing training on hard cases. With $\gamma=2$, an easy example with $p_t=0.9$ contributes 100x less than one with $p_t=0.5$.

### Contrastive Loss

For learning embeddings:

$$
L = (1-y) \frac{1}{2} d^2 + y \frac{1}{2} \max(0, m - d)^2
$$

```python
class ContrastiveLoss:
    def __init__(self, margin=1.0):
        self.margin = margin

    def forward(self, embedding1, embedding2, labels):
        """
        Args:
            embedding1, embedding2: Feature vectors [batch_size, dim]
            labels: 1 if same class, 0 if different
        """
        distances = np.sqrt(np.sum((embedding1 - embedding2)**2, axis=-1))

        # Same class: minimize distance
        # Different class: maximize distance (up to margin)
        same_loss = labels * distances**2
        diff_loss = (1 - labels) * np.maximum(0, self.margin - distances)**2

        return np.mean(0.5 * (same_loss + diff_loss))
```

### Triplet Loss

Pull anchor-positive pairs together, push anchor-negative apart:

$$
L = \max(0, d(a, p) - d(a, n) + m)
$$

```python
class TripletLoss:
    def __init__(self, margin=0.2):
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor, positive, negative: Embeddings [batch_size, dim]
        """
        dist_pos = np.sqrt(np.sum((anchor - positive)**2, axis=-1))
        dist_neg = np.sqrt(np.sum((anchor - negative)**2, axis=-1))

        loss = np.maximum(0, dist_pos - dist_neg + self.margin)
        return np.mean(loss)
```

### InfoNCE Loss

Contrastive learning with multiple negatives:

$$
L = -\log \frac{\exp(q \cdot k_+ / \tau)}{\sum_{i} \exp(q \cdot k_i / \tau)}
$$

```python
class InfoNCELoss:
    def __init__(self, temperature=0.07):
        self.temperature = temperature

    def forward(self, query, keys, positive_idx=0):
        """
        Args:
            query: Query embedding [batch_size, dim]
            keys: Key embeddings [batch_size, num_keys, dim]
            positive_idx: Index of positive key
        """
        # Similarity scores
        logits = np.sum(query[:, None, :] * keys, axis=-1) / self.temperature

        # Softmax cross-entropy with positive as target
        log_probs = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
        loss = -log_probs[:, positive_idx]

        return np.mean(loss)
```

## Loss for Sequence Tasks

### Negative Log-Likelihood for Language Models

$$
L = -\frac{1}{T} \sum_{t=1}^{T} \log p(x_t | x_{<t})
$$

```python
def language_model_loss(logits, targets):
    """
    Cross-entropy for autoregressive language modeling.

    Args:
        logits: [batch_size, seq_len, vocab_size]
        targets: [batch_size, seq_len]
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape for cross-entropy
    logits_flat = logits.reshape(-1, vocab_size)

    # One-hot encode targets
    targets_flat = targets.reshape(-1)
    targets_onehot = np.zeros((batch_size * seq_len, vocab_size))
    targets_onehot[np.arange(batch_size * seq_len), targets_flat] = 1

    # Cross-entropy
    probs = softmax(logits_flat)
    log_probs = np.log(np.clip(probs, 1e-15, 1))
    loss = -np.sum(targets_onehot * log_probs) / (batch_size * seq_len)

    return loss

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

### Perplexity

Perplexity is the exponentiated average cross-entropy:

$$
\text{PPL} = \exp\left(-\frac{1}{T} \sum_t \log p(x_t | x_{<t})\right)
$$

```python
def perplexity(cross_entropy_loss):
    """Convert cross-entropy to perplexity."""
    return np.exp(cross_entropy_loss)

# Example: cross-entropy of 4.0 nats
ce_loss = 4.0
ppl = perplexity(ce_loss)
print(f"Cross-entropy: {ce_loss:.2f}, Perplexity: {ppl:.1f}")
# Perplexity ~54.6 means model is as uncertain as uniform over ~55 words
```

**What this means:** Perplexity is the effective vocabulary size the model is choosing from. A perplexity of 50 means the model is, on average, equally uncertain between 50 possible next words.

## Choosing Loss Functions

### Quick Reference

| Task | Loss | Output Activation |
|------|------|-------------------|
| Regression | MSE, MAE, Huber | None (linear) |
| Binary classification | Binary cross-entropy | Sigmoid |
| Multi-class (single label) | Cross-entropy | Softmax |
| Multi-class (multi-label) | Binary cross-entropy per class | Sigmoid |
| Embedding learning | Contrastive, Triplet, InfoNCE | Normalize to unit sphere |
| Language modeling | Cross-entropy | Softmax over vocab |
| Object detection | Focal loss + regression | Sigmoid + linear |

### Implementation Checklist

```python
def implement_loss_checklist():
    """Things to check when implementing loss functions."""
    checks = [
        "Numerical stability (clipping, log-sum-exp)",
        "Correct gradient formula",
        "Handling of batch dimension (mean vs sum)",
        "Compatibility with optimizer expectations",
        "Proper handling of edge cases (y_pred = 0 or 1)",
    ]
    return checks
```

## Summary

| Loss | Formula | Use Case |
|------|---------|----------|
| MSE | $\frac{1}{N}\sum(y - \hat{y})^2$ | Regression |
| MAE | $\frac{1}{N}\sum\|y - \hat{y}\|$ | Robust regression |
| Huber | Quadratic + linear | Balanced regression |
| Cross-entropy | $-\sum y \log \hat{y}$ | Classification |
| Focal | $-(1-p)^\gamma \log p$ | Imbalanced classes |
| Contrastive | Pair-based embedding | Similarity learning |
| Triplet | Anchor-pos-neg | Ranking/retrieval |

**The essential insight:** Loss functions translate model outputs into a single number measuring failure. The choice of loss encodes assumptions: MSE assumes Gaussian errors, cross-entropy assumes categorical distributions, focal loss assumes class imbalance. Match the loss to your assumptions about the problem.

**Next:** [Optimizers](optimizers.md) covers algorithms for minimizing these losses.
