# Probability for Machine Learning

$$
\boxed{P(A|B) = \frac{P(B|A) P(A)}{P(B)}}
$$

**Bayes' theorem** is the foundation of probabilistic reasoning. But for deep learning, the key insight is simpler: neural network outputs are probability distributions, and training minimizes the "distance" between predicted and true distributions.

Prerequisites: Basic algebra. Code: `numpy`.

---

## Probability Basics

### What Is Probability?

A **probability** $P(A)$ is a number between 0 and 1 representing how likely event $A$ is:
- $P(A) = 0$: Impossible
- $P(A) = 1$: Certain
- $P(A) = 0.5$: Equally likely as not

### Rules

**Sum rule:** For mutually exclusive events:
$$
P(A \text{ or } B) = P(A) + P(B)
$$

**Product rule:** For independent events:
$$
P(A \text{ and } B) = P(A) \cdot P(B)
$$

**Complement:**
$$
P(\text{not } A) = 1 - P(A)
$$

### Conditional Probability

The probability of $A$ **given** $B$:

$$
P(A|B) = \frac{P(A \text{ and } B)}{P(B)}
$$

**What this means:** If we know $B$ happened, how does that change the probability of $A$? Conditioning restricts us to the world where $B$ is true.

## Probability Distributions

### Discrete Distributions

For a discrete random variable $X$ with possible values $x_1, x_2, \ldots$:

$$
P(X = x_i) = p_i \quad \text{where} \quad \sum_i p_i = 1
$$

**Example:** Rolling a fair die
```python
import numpy as np

# Probability distribution for a fair die
outcomes = [1, 2, 3, 4, 5, 6]
probs = [1/6] * 6  # uniform

print(sum(probs))  # 1.0 - probabilities sum to 1
```

### Continuous Distributions

For continuous variables, we use a **probability density function** (PDF):

$$
P(a \leq X \leq b) = \int_a^b p(x) \, dx
$$

The probability of any *exact* value is 0, but we can compute probability over intervals.

### Common Distributions

| Distribution | Use Case | Formula |
|--------------|----------|---------|
| Bernoulli | Binary outcome | $P(X=1) = p$ |
| Categorical | Multi-class | $P(X=i) = p_i$ |
| Gaussian | Continuous, symmetric | $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ |
| Uniform | Equal probability | $p(x) = \frac{1}{b-a}$ for $x \in [a,b]$ |

## Expectation and Variance

### Expectation (Mean)

The **expected value** is the probability-weighted average:

$$
E[X] = \sum_i x_i P(X = x_i) = \sum_i x_i p_i
$$

For continuous: $E[X] = \int x \, p(x) \, dx$

**What this means:** If you repeated the experiment infinitely many times and averaged the results, you'd get $E[X]$.

### Variance

**Variance** measures spread:

$$
\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
$$

**Standard deviation:** $\sigma = \sqrt{\text{Var}(X)}$

### In Code

```python
def expected_value(values, probs):
    """E[X] = sum of x * P(x)"""
    return np.sum(values * probs)

def variance(values, probs):
    """Var(X) = E[X^2] - E[X]^2"""
    mean = expected_value(values, probs)
    mean_sq = expected_value(values**2, probs)
    return mean_sq - mean**2

# Example: fair die
values = np.array([1, 2, 3, 4, 5, 6])
probs = np.array([1/6] * 6)

print(f"E[X] = {expected_value(values, probs):.2f}")  # 3.5
print(f"Var(X) = {variance(values, probs):.2f}")      # 2.92
```

## Maximum Likelihood Estimation

Given data, we want to find the parameters that make the data most likely.

### Example: Coin Flips

We flip a coin 10 times and get 7 heads. What's the most likely value of $p$ (probability of heads)?

The **likelihood** of the data given $p$:

$$
L(p) = p^7 (1-p)^3
$$

To maximize, take the derivative and set to zero:

$$
\frac{dL}{dp} = 7p^6(1-p)^3 - 3p^7(1-p)^2 = 0
$$

Solving: $p = 7/10 = 0.7$

**What this means:** The maximum likelihood estimate is the fraction of heads we observed. This is the "obvious" answer, but maximum likelihood provides the formal justification.

### Log-Likelihood

In practice, we work with **log-likelihood** (sums instead of products):

$$
\log L(p) = 7 \log(p) + 3 \log(1-p)
$$

Maximizing log-likelihood is equivalent to maximizing likelihood (log is monotonic).

**What this means:** Products become sums with logs. This prevents numerical underflow when multiplying many small probabilities, which happens constantly in ML.

## Softmax: Turning Scores into Probabilities

The **softmax** function converts arbitrary real numbers into a probability distribution:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### Properties

1. All outputs are positive (exponential)
2. Outputs sum to 1 (normalization)
3. Preserves order (larger $z_i$ → larger probability)
4. Differentiable (can backpropagate through it)

### In Code

```python
def softmax(z):
    """Convert logits to probabilities."""
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum()

# Example: model outputs (logits)
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)

print(probs)       # [0.659, 0.242, 0.099]
print(probs.sum()) # 1.0
```

### Temperature

We can control the "sharpness" of softmax with temperature $T$:

$$
\text{softmax}(z_i / T)
$$

- $T < 1$: Sharper (more confident)
- $T > 1$: Softer (more uniform)
- $T \to 0$: Approaches one-hot (argmax)
- $T \to \infty$: Approaches uniform

```python
def softmax_with_temperature(z, T=1.0):
    exp_z = np.exp((z - np.max(z)) / T)
    return exp_z / exp_z.sum()

logits = np.array([2.0, 1.0, 0.1])
print(f"T=0.5: {softmax_with_temperature(logits, 0.5)}")  # sharper
print(f"T=1.0: {softmax_with_temperature(logits, 1.0)}")  # normal
print(f"T=2.0: {softmax_with_temperature(logits, 2.0)}")  # softer
```

**What this means:** Temperature controls exploration vs exploitation. High temperature for diverse sampling, low temperature for confident predictions.

## Entropy: Measuring Uncertainty

### Information Content

The **information** (surprise) of an event with probability $p$:

$$
I(p) = -\log_2(p) = \log_2(1/p)
$$

- Rare events (small $p$) carry more information
- Certain events ($p = 1$) carry zero information

### Shannon Entropy

**Entropy** is the expected information:

$$
H(P) = -\sum_i p_i \log p_i = E[-\log p_i]
$$

(Using natural log gives nats; base 2 gives bits)

### Properties

| Distribution | Entropy |
|--------------|---------|
| Uniform | Maximum (most uncertain) |
| One-hot | Zero (completely certain) |
| Peaked | Low (confident) |
| Spread | High (uncertain) |

### In Code

```python
def entropy(probs):
    """Shannon entropy in nats."""
    # Avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))

# Examples
uniform = np.array([0.25, 0.25, 0.25, 0.25])
peaked = np.array([0.9, 0.05, 0.03, 0.02])
certain = np.array([1.0, 0.0, 0.0, 0.0])

print(f"Uniform: {entropy(uniform):.3f}")  # 1.386 (max for 4 classes)
print(f"Peaked:  {entropy(peaked):.3f}")   # 0.451
print(f"Certain: {entropy(certain):.3f}")  # 0.0
```

**What this means:** Entropy measures how "spread out" a distribution is. A model that's uncertain about its prediction has high entropy outputs.

## Cross-Entropy: Comparing Distributions

**Cross-entropy** measures how well distribution $Q$ matches true distribution $P$:

$$
H(P, Q) = -\sum_i p_i \log q_i = E_{p}[-\log q_i]
$$

### As a Loss Function

For classification, $P$ is the one-hot true label, $Q$ is the predicted distribution:

$$
L = -\sum_i y_i \log \hat{y}_i
$$

Since $y$ is one-hot (only one $y_i = 1$), this simplifies to:

$$
L = -\log \hat{y}_c
$$

where $c$ is the correct class.

**What this means:** Cross-entropy loss is high when the model assigns low probability to the correct class. It penalizes confident wrong predictions severely.

### In Code

```python
def cross_entropy_loss(y_true, y_pred):
    """
    y_true: one-hot labels (batch_size, num_classes)
    y_pred: predicted probabilities (batch_size, num_classes)
    """
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, 1e-10, 1.0)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

# Example
y_true = np.array([[1, 0, 0], [0, 1, 0]])  # classes 0 and 1
y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])

loss = cross_entropy_loss(y_true, y_pred)
print(f"Loss: {loss:.4f}")  # low loss - predictions are good
```

### Relationship to Entropy

$$
H(P, Q) = H(P) + D_{KL}(P || Q)
$$

where $D_{KL}$ is the KL divergence (below). When $P$ is a one-hot distribution, $H(P) = 0$, so cross-entropy equals KL divergence.

## KL Divergence: Distance Between Distributions

**Kullback-Leibler divergence** measures how different $Q$ is from $P$:

$$
D_{KL}(P || Q) = \sum_i p_i \log \frac{p_i}{q_i} = E_P\left[\log \frac{p_i}{q_i}\right]
$$

### Properties

- $D_{KL}(P || Q) \geq 0$ (always non-negative)
- $D_{KL}(P || Q) = 0$ iff $P = Q$
- **Not symmetric:** $D_{KL}(P || Q) \neq D_{KL}(Q || P)$ in general

### In Code

```python
def kl_divergence(p, q):
    """KL(P || Q)"""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log(p / q))

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.4, 0.4, 0.2])

print(f"KL(P||Q) = {kl_divergence(p, q):.4f}")  # 0.0410
print(f"KL(Q||P) = {kl_divergence(q, p):.4f}")  # 0.0404 (different!)
```

**What this means:** KL divergence measures the "extra bits" needed to encode samples from $P$ using a code optimized for $Q$. It's used in variational inference, knowledge distillation, and as a regularizer.

## Bayes' Theorem

$$
P(A|B) = \frac{P(B|A) P(A)}{P(B)}
$$

### Interpretation

- $P(A)$: **Prior** - belief before seeing evidence
- $P(B|A)$: **Likelihood** - probability of evidence given hypothesis
- $P(A|B)$: **Posterior** - updated belief after seeing evidence
- $P(B)$: **Evidence** - normalization constant

### Example: Spam Detection

- $P(\text{spam})$: Prior probability an email is spam (say 30%)
- $P(\text{"free"} | \text{spam})$: Probability spam contains "free" (say 80%)
- $P(\text{"free"} | \text{not spam})$: Probability normal email contains "free" (say 10%)

Given an email contains "free", what's $P(\text{spam} | \text{"free"})$?

$$
P(\text{spam} | \text{"free"}) = \frac{0.8 \times 0.3}{0.8 \times 0.3 + 0.1 \times 0.7} = \frac{0.24}{0.31} \approx 0.77
$$

The word "free" increases our belief from 30% to 77%.

## Sampling

### Why Sample?

- Generate new data (text, images)
- Estimate expectations: $E[f(X)] \approx \frac{1}{N} \sum_{i=1}^N f(x_i)$
- Explore diverse outputs

### Sampling from Discrete Distributions

```python
def sample(probs, n=1):
    """Sample n items from a categorical distribution."""
    return np.random.choice(len(probs), size=n, p=probs)

# Sample from softmax output
probs = np.array([0.7, 0.2, 0.1])
samples = sample(probs, 10)
print(samples)  # e.g., [0, 0, 1, 0, 0, 0, 2, 0, 0, 0]
```

### Top-k and Top-p Sampling

For text generation, we often restrict sampling:

**Top-k:** Only sample from the $k$ most likely tokens

**Top-p (nucleus):** Sample from the smallest set of tokens whose cumulative probability ≥ $p$

```python
def top_k_sample(logits, k=10):
    """Sample from top k tokens."""
    top_k_idx = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_idx]
    probs = softmax(top_k_logits)
    chosen_idx = np.random.choice(len(probs), p=probs)
    return top_k_idx[chosen_idx]

def top_p_sample(logits, p=0.9):
    """Sample from smallest set with cumulative prob >= p."""
    probs = softmax(logits)
    sorted_idx = np.argsort(probs)[::-1]
    cumsum = np.cumsum(probs[sorted_idx])
    cutoff = np.searchsorted(cumsum, p) + 1
    top_p_idx = sorted_idx[:cutoff]
    top_p_probs = probs[top_p_idx]
    top_p_probs /= top_p_probs.sum()  # renormalize
    chosen_idx = np.random.choice(len(top_p_probs), p=top_p_probs)
    return top_p_idx[chosen_idx]
```

**What this means:** Pure sampling can produce unlikely tokens. Top-k/top-p restrict to reasonable options while maintaining diversity.

## Putting It Together: Classification

A classifier outputs a probability distribution over classes:

```python
def classify(x, W, b):
    """
    x: input features
    W: weight matrix
    b: bias vector
    Returns: probability distribution over classes
    """
    logits = x @ W + b
    probs = softmax(logits)
    return probs

def cross_entropy(y_true, probs):
    """Cross-entropy loss for classification."""
    return -np.log(probs[y_true])

# Training loop (simplified)
def train_step(x, y_true, W, b, lr=0.01):
    # Forward pass
    probs = classify(x, W, b)
    loss = cross_entropy(y_true, probs)

    # Gradient (for softmax + cross-entropy)
    grad = probs.copy()
    grad[y_true] -= 1  # beautiful simplification

    # Update weights (simplified - ignoring full backprop)
    # ...

    return loss
```

## Summary

| Concept | Formula | ML Usage |
|---------|---------|----------|
| Softmax | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | Logits → probabilities |
| Entropy | $-\sum p_i \log p_i$ | Measure uncertainty |
| Cross-entropy | $-\sum y_i \log \hat{y}_i$ | Classification loss |
| KL divergence | $\sum p_i \log \frac{p_i}{q_i}$ | Distribution matching |
| Bayes | $P(A|B) \propto P(B|A)P(A)$ | Probabilistic reasoning |

**The essential insight:** Neural networks output probability distributions. Training minimizes cross-entropy, which pushes the predicted distribution toward the true distribution. Softmax converts raw scores to valid probabilities. Entropy measures confidence. These concepts unify classification, language modeling, and generative models.

**Next:** [Perceptron](../neural-networks/perceptron.md) to start building neural networks.

**Notebook:** [01-numpy-neural-net.ipynb](../notebooks/01-numpy-neural-net.ipynb) uses cross-entropy loss to train a classifier.
