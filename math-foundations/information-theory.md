# Information Theory

$$
\boxed{H(p) = -\sum_{x} p(x) \log p(x)}
$$

**Information theory** quantifies uncertainty and the cost of communication. In machine learning, it provides the foundation for loss functions, explains why certain architectures work, and gives us tools to measure what models have learned. Cross-entropy loss, KL divergence, and mutual information all come from information theory.

Prerequisites: [probability](probability.md) (distributions, expectations). Code: `numpy`.

---

## Surprise and Information

### The Core Idea

How surprised are you when an event happens? If something is certain (probability 1), no surprise. If something is rare (probability 0.001), very surprised.

**Information content** (or "surprisal") of event $x$:

$$
I(x) = -\log p(x)
$$

The negative log ensures:
- Certain events ($p = 1$): zero information
- Rare events ($p \to 0$): high information
- Independent events add: $I(x, y) = I(x) + I(y)$

```python
import numpy as np

def information(p):
    """Information content in bits."""
    return -np.log2(p)

# Examples
print(f"Fair coin heads: {information(0.5):.2f} bits")   # 1.0 bits
print(f"Loaded coin (p=0.9): {information(0.9):.2f} bits")  # 0.15 bits
print(f"Rare event (p=0.01): {information(0.01):.2f} bits")  # 6.64 bits
```

**What this means:** Information measures surprise. A fair coin flip gives 1 bit of information. An event that happens 99% of the time tells you almost nothing when it occurs.

### Bits vs Nats

The base of the logarithm determines the unit:

| Base | Unit | Common In |
|------|------|-----------|
| 2 | bits | Information theory, compression |
| $e$ | nats | Machine learning, neural networks |
| 10 | digits | Sometimes historical |

**Conversion:** 1 nat = $\log_2(e) \approx 1.44$ bits

In ML, we typically use natural log (nats) because:
- Derivatives are cleaner: $\frac{d}{dx} \ln x = \frac{1}{x}$
- Connects directly to probability via $e^x$

```python
def bits_to_nats(bits):
    return bits / np.log2(np.e)

def nats_to_bits(nats):
    return nats * np.log2(np.e)
```

## Entropy

### Shannon Entropy

The **entropy** of a distribution is the expected information content—the average surprise:

$$
H(p) = -\sum_{x} p(x) \log p(x) = \mathbb{E}_{x \sim p}[-\log p(x)]
$$

```python
def entropy(p):
    """Shannon entropy in nats."""
    # Handle p=0 (0 * log(0) = 0 by convention)
    p = np.array(p)
    p = p[p > 0]
    return -np.sum(p * np.log(p))

# Fair coin
print(f"Fair coin: {entropy([0.5, 0.5]):.4f} nats")  # 0.6931 nats

# Biased coin
print(f"Biased (90/10): {entropy([0.9, 0.1]):.4f} nats")  # 0.3251 nats

# Certain outcome
print(f"Certain: {entropy([1.0, 0.0]):.4f} nats")  # 0.0 nats
```

### Properties of Entropy

1. **Non-negative:** $H(p) \geq 0$
2. **Maximum at uniform:** For $n$ outcomes, $H \leq \log n$ with equality when uniform
3. **Zero for certainty:** $H = 0$ iff one outcome has probability 1

```python
def entropy_examples():
    """Entropy for different distributions."""
    # Uniform distribution has maximum entropy
    n = 6  # Like a die
    uniform = [1/n] * n
    print(f"Uniform (n={n}): {entropy(uniform):.4f} nats")
    print(f"Maximum possible: {np.log(n):.4f} nats")

    # More peaked = lower entropy
    peaked = [0.7, 0.1, 0.1, 0.05, 0.03, 0.02]
    print(f"Peaked: {entropy(peaked):.4f} nats")

entropy_examples()
```

**What this means:** Entropy measures uncertainty. A uniform distribution over many outcomes has high entropy—anything could happen. A peaked distribution has low entropy—you can predict what's likely.

### Continuous Entropy

For continuous distributions, we use **differential entropy**:

$$
h(p) = -\int p(x) \log p(x) \, dx
$$

Key examples:
- **Gaussian:** $h(\mathcal{N}(\mu, \sigma^2)) = \frac{1}{2} \log(2\pi e \sigma^2)$
- **Uniform on $[a, b]$:** $h = \log(b - a)$

Differential entropy can be negative (unlike discrete entropy).

## Cross-Entropy

### Definition

**Cross-entropy** measures the average bits needed to encode samples from $p$ using a code optimized for $q$:

$$
H(p, q) = -\sum_{x} p(x) \log q(x) = \mathbb{E}_{x \sim p}[-\log q(x)]
$$

```python
def cross_entropy(p, q, epsilon=1e-15):
    """Cross-entropy H(p, q)."""
    p = np.array(p)
    q = np.array(q)
    # Clip q to avoid log(0)
    q = np.clip(q, epsilon, 1 - epsilon)
    return -np.sum(p * np.log(q))

# True distribution vs model prediction
p_true = [0.7, 0.2, 0.1]  # True labels
q_model = [0.6, 0.3, 0.1]  # Model prediction

print(f"Cross-entropy: {cross_entropy(p_true, q_model):.4f}")
print(f"True entropy: {entropy(p_true):.4f}")
```

### Cross-Entropy as Loss

In classification, we use cross-entropy loss:
- True distribution $p$: one-hot (e.g., $[0, 1, 0]$ for class 1)
- Model prediction $q$: softmax output

$$
L = -\sum_{c} y_c \log \hat{y}_c
$$

For one-hot labels, this simplifies to:

$$
L = -\log \hat{y}_{\text{true class}}
$$

```python
def cross_entropy_loss(y_true, y_pred):
    """
    Cross-entropy loss for classification.

    Args:
        y_true: One-hot encoded labels [batch, classes]
        y_pred: Softmax predictions [batch, classes]
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

# Example: batch of 3 samples, 4 classes
y_true = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])
y_pred = np.array([
    [0.7, 0.1, 0.1, 0.1],  # Good prediction
    [0.1, 0.8, 0.05, 0.05], # Good prediction
    [0.25, 0.25, 0.25, 0.25] # Bad prediction (uniform)
])

print(f"Cross-entropy loss: {cross_entropy_loss(y_true, y_pred):.4f}")
```

**What this means:** Cross-entropy loss punishes confident wrong predictions heavily. If the model predicts 0.01 for the true class, the loss is $-\log(0.01) \approx 4.6$. If it predicts 0.99, the loss is only $-\log(0.99) \approx 0.01$.

## KL Divergence

### Definition

**Kullback-Leibler divergence** measures how different two distributions are:

$$
D_{KL}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)
$$

```python
def kl_divergence(p, q, epsilon=1e-15):
    """KL divergence D_KL(p || q)."""
    p = np.array(p)
    q = np.array(q)
    # Avoid division by zero and log(0)
    q = np.clip(q, epsilon, 1)
    p = np.clip(p, epsilon, 1)
    return np.sum(p * np.log(p / q))

p = [0.7, 0.2, 0.1]
q = [0.5, 0.3, 0.2]

print(f"KL(p || q): {kl_divergence(p, q):.4f}")
print(f"KL(q || p): {kl_divergence(q, p):.4f}")  # Different!
```

### Properties

1. **Non-negative:** $D_{KL}(p \| q) \geq 0$
2. **Zero iff equal:** $D_{KL}(p \| q) = 0 \Leftrightarrow p = q$
3. **Not symmetric:** $D_{KL}(p \| q) \neq D_{KL}(q \| p)$ in general
4. **Not a metric:** Doesn't satisfy triangle inequality

### Asymmetry

The asymmetry is important:

- **$D_{KL}(p \| q)$**: Cost of using $q$ to approximate $p$
- **$D_{KL}(q \| p)$**: Cost of using $p$ to approximate $q$

```
p = true distribution
q = model approximation

Forward KL: D_KL(p || q)
- Penalizes q(x) ≈ 0 where p(x) > 0 (mode-covering)
- Model tries to cover all modes of p

Reverse KL: D_KL(q || p)
- Penalizes q(x) > 0 where p(x) ≈ 0 (mode-seeking)
- Model concentrates on high-probability regions of p
```

```python
def visualize_kl_asymmetry():
    """Show how forward and reverse KL behave differently."""
    import matplotlib.pyplot as plt

    # True distribution: mixture of two Gaussians
    x = np.linspace(-5, 5, 1000)

    def mixture(x):
        return 0.5 * np.exp(-0.5 * (x + 2)**2) + 0.5 * np.exp(-0.5 * (x - 2)**2)

    p = mixture(x)
    p = p / np.trapz(p, x)  # Normalize

    # Forward KL solution: covers both modes (wide Gaussian)
    q_forward = np.exp(-0.5 * (x / 2.5)**2)
    q_forward = q_forward / np.trapz(q_forward, x)

    # Reverse KL solution: picks one mode
    q_reverse = np.exp(-0.5 * (x - 2)**2)
    q_reverse = q_reverse / np.trapz(q_reverse, x)

    plt.figure(figsize=(10, 4))
    plt.plot(x, p, 'k-', linewidth=2, label='True p(x)')
    plt.plot(x, q_forward, 'b--', label='Forward KL (mode-covering)')
    plt.plot(x, q_reverse, 'r--', label='Reverse KL (mode-seeking)')
    plt.legend()
    plt.title('KL Divergence Asymmetry')
    plt.show()
```

**What this means:** In variational inference, we minimize reverse KL (fitting q to p). In maximum likelihood, we minimize forward KL (fitting model to data). The choice affects whether models cover all modes or focus on the main ones.

### Connection to Cross-Entropy Loss

Since $D_{KL}(p \| q) = H(p, q) - H(p)$:

When optimizing $q$ (model parameters), $H(p)$ is constant, so:

$$
\min_q D_{KL}(p \| q) \equiv \min_q H(p, q)
$$

Minimizing cross-entropy = minimizing KL divergence from true distribution.

## Mutual Information

### Definition

**Mutual information** measures how much knowing one variable tells you about another:

$$
I(X; Y) = H(X) + H(Y) - H(X, Y)
$$

Equivalently:

$$
I(X; Y) = D_{KL}(p(x, y) \| p(x)p(y))
$$

```python
def mutual_information(joint_prob):
    """
    Mutual information from joint distribution.

    Args:
        joint_prob: 2D array p(x, y)
    """
    p_xy = np.array(joint_prob)

    # Marginals
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)

    # Entropies
    h_x = entropy(p_x)
    h_y = entropy(p_y)
    h_xy = entropy(p_xy.flatten())

    return h_x + h_y - h_xy

# Example: X and Y are dependent
joint = np.array([
    [0.4, 0.1],  # p(X=0, Y=0), p(X=0, Y=1)
    [0.1, 0.4]   # p(X=1, Y=0), p(X=1, Y=1)
])

print(f"Mutual information: {mutual_information(joint):.4f} nats")

# Independent case (MI = 0)
p_x = [0.5, 0.5]
p_y = [0.5, 0.5]
independent = np.outer(p_x, p_y)
print(f"Independent MI: {mutual_information(independent):.6f} nats")
```

### Properties

1. **Non-negative:** $I(X; Y) \geq 0$
2. **Symmetric:** $I(X; Y) = I(Y; X)$
3. **Zero iff independent:** $I(X; Y) = 0 \Leftrightarrow X \perp Y$
4. **Bounded:** $I(X; Y) \leq \min(H(X), H(Y))$

### Interpretation

```
I(X; Y) = H(X) - H(X|Y)
        = uncertainty in X minus uncertainty in X given Y
        = reduction in uncertainty about X from knowing Y
```

**What this means:** Mutual information quantifies the information shared between variables. In ML, it's used to measure how much a representation captures about the input, how related features are, or how much a model's output tells you about the true label.

### Applications in ML

**InfoNCE Loss** (contrastive learning):

$$
L = -\log \frac{\exp(f(x) \cdot g(x^+))}{\sum_{i} \exp(f(x) \cdot g(x^-_i))}
$$

This is a lower bound on mutual information between representations.

**Variational Autoencoders:**

$$
\text{ELBO} = \mathbb{E}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))
$$

The reconstruction term relates to mutual information between $x$ and $z$.

## Connections to Loss Functions

### Why Cross-Entropy?

Maximum likelihood estimation:

$$
\hat{\theta} = \arg\max_\theta \prod_{i} p_\theta(x_i)
$$

Taking log and negating:

$$
\hat{\theta} = \arg\min_\theta -\frac{1}{N} \sum_{i} \log p_\theta(x_i)
$$

This is the cross-entropy between empirical distribution and model.

### Why KL Divergence?

If $\hat{p}$ is the empirical distribution:

$$
\min_\theta D_{KL}(\hat{p} \| p_\theta) = \min_\theta H(\hat{p}, p_\theta) - H(\hat{p})
$$

Since $H(\hat{p})$ is constant, minimizing KL = minimizing cross-entropy = maximum likelihood.

### Loss Function Comparison

| Loss | Information-Theoretic View |
|------|---------------------------|
| Cross-entropy | Expected bits to encode labels with model's distribution |
| KL divergence | Extra bits from using wrong distribution |
| MSE | Cross-entropy assuming Gaussian distribution |
| Binary cross-entropy | Special case for Bernoulli |

```python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """Binary cross-entropy loss."""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def mse_as_nll(y_true, y_pred, sigma=1.0):
    """MSE is negative log-likelihood of Gaussian."""
    # p(y|x) = N(y; f(x), sigma^2)
    # -log p = (y - f(x))^2 / (2*sigma^2) + const
    return np.mean((y_true - y_pred)**2) / (2 * sigma**2)
```

## Practical Considerations

### Numerical Stability

Cross-entropy with softmax can overflow. Use the log-sum-exp trick:

```python
def stable_softmax(logits):
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

def stable_cross_entropy_with_logits(labels, logits):
    """
    Stable cross-entropy directly from logits.
    Avoids computing softmax separately.
    """
    # log_softmax = logits - log(sum(exp(logits)))
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    log_softmax = shifted - log_sum_exp
    return -np.sum(labels * log_softmax, axis=-1)
```

### Label Smoothing

Modify one-hot labels to prevent overconfidence:

$$
y'_i = (1 - \alpha) y_i + \frac{\alpha}{K}
$$

```python
def label_smoothing(labels, alpha=0.1):
    """Apply label smoothing."""
    n_classes = labels.shape[-1]
    return (1 - alpha) * labels + alpha / n_classes

# Example
one_hot = np.array([1, 0, 0, 0])
smoothed = label_smoothing(one_hot.reshape(1, -1), alpha=0.1)
print(f"Smoothed: {smoothed}")  # [0.925, 0.025, 0.025, 0.025]
```

**What this means:** Label smoothing adds a small amount of uncertainty to targets. This prevents the model from becoming overconfident and can improve generalization.

## Summary

| Concept | Formula | Meaning |
|---------|---------|---------|
| Information | $-\log p(x)$ | Surprise of event |
| Entropy | $-\sum p \log p$ | Average surprise |
| Cross-entropy | $-\sum p \log q$ | Bits with wrong code |
| KL divergence | $\sum p \log \frac{p}{q}$ | Extra bits from wrong dist |
| Mutual information | $H(X) + H(Y) - H(X,Y)$ | Shared information |

**The essential insight:** Information theory provides the principled foundation for machine learning loss functions. Cross-entropy loss isn't an arbitrary choice—it's the natural way to measure how well a probability distribution matches data. Understanding this connection explains why certain losses work well and guides the design of new ones.

**Next:** [Loss Functions](../training/loss-functions.md) covers the practical implementations used in neural networks.
