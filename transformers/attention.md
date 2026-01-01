# Attention

$$
\boxed{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V}
$$

The **attention mechanism** lets a model dynamically focus on relevant parts of its input. It's the core innovation that makes transformers possible, replacing the sequential processing of RNNs with parallel, content-based routing.

Prerequisites: [linear-algebra](../math-foundations/linear-algebra.md) (matrix multiplication, dot products), [probability](../math-foundations/probability.md) (softmax). Code: `numpy`.

---

## The Problem: Fixed Representations

RNNs process sequences left-to-right, compressing everything into a fixed-size hidden state. For long sequences, early information gets "forgotten"—the hidden state is a bottleneck.

**What this means:** Imagine summarizing a book into a single sentence, then answering questions using only that sentence. You'd lose details. RNNs face this problem at scale—by the time you reach the end of a long sequence, information from the beginning has been repeatedly transformed and compressed.

### The Bottleneck in Numbers

For a sequence of length $n$ with hidden dimension $d$:
- Information from position 1 must pass through $n-1$ transformations to reach position $n$
- Each transformation can lose or distort information
- The hidden state has fixed capacity $d$, regardless of sequence length

## The Solution: Look Everywhere at Once

Attention computes a weighted combination of all inputs, with weights based on relevance:

$$
\text{output}_i = \sum_j \alpha_{ij} \cdot \text{value}_j
$$

where $\alpha_{ij}$ measures "how much should position $i$ attend to position $j$?"

**What this means:** Instead of forcing information through a sequential bottleneck, attention lets each position directly access every other position. The model learns which positions are relevant and weights them accordingly.

## Computing Attention Weights

Given a **query** $q$ and **keys** $k_1, \ldots, k_n$:

$$
\alpha_j = \frac{\exp(q \cdot k_j / \sqrt{d_k})}{\sum_m \exp(q \cdot k_m / \sqrt{d_k})}
$$

### Breaking This Down

1. **Dot product** $q \cdot k_j$: Measures similarity between query and key. High if they point in the same direction.

2. **Scaling** by $\sqrt{d_k}$: Prevents dot products from getting too large in high dimensions.

3. **Softmax**: Converts raw scores to probabilities that sum to 1.

**What this means:**
- The dot product $q \cdot k_j$ measures similarity (high if query and key point in the same direction in embedding space)
- Softmax converts similarities to probabilities (they sum to 1, so we get a weighted average)
- The $\sqrt{d_k}$ scaling prevents dot products from becoming too large in high dimensions, which would push softmax into regions with tiny gradients

### Why Scaling Matters

Without scaling, for random vectors in $d$ dimensions:
- Expected value of $q \cdot k$ is 0
- Variance of $q \cdot k$ is $d$

So as $d$ grows, dot products have larger magnitude, pushing softmax toward one-hot outputs (one weight near 1, rest near 0). Dividing by $\sqrt{d}$ keeps the variance at 1 regardless of dimension.

## The Query-Key-Value Framework

The full attention mechanism uses three different projections:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

where $X$ is the input and $W_Q, W_K, W_V$ are learned weight matrices.

### Why Three Separate Projections?

**Analogy:** A library search.
- **Query:** What you're looking for ("books about attention mechanisms")
- **Key:** How items are indexed ("machine learning", "neuroscience", "psychology")
- **Value:** The actual content you retrieve

Separating these lets the model learn different representations for:
- **What to match on** (query-key similarity)
- **What to return** (values)

If we used the same vectors for all three, matching and retrieval would be coupled—good matches might return bad content, or vice versa.

## In Code

```python
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    exp_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def attention(Q, K, V):
    """
    Scaled dot-product attention.

    Args:
        Q: Queries, shape (seq_len, d_k)
        K: Keys, shape (seq_len, d_k)
        V: Values, shape (seq_len, d_v)

    Returns:
        Output, shape (seq_len, d_v)
        Attention weights, shape (seq_len, seq_len)
    """
    d_k = K.shape[-1]

    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)  # (seq_len, seq_len)

    # Convert to probabilities
    weights = softmax(scores)  # each row sums to 1

    # Weighted sum of values
    output = weights @ V  # (seq_len, d_v)

    return output, weights
```

**What the code shows:**
- The entire attention mechanism is three matrix operations
- `Q @ K.T` computes all pairwise similarities at once—no loops over positions
- `weights @ V` retrieves a weighted combination of values
- The output has the same sequence length as input, but each position now contains information from all positions

### Example: Attending to a Sequence

```python
# Example: 4 tokens, dimension 8
seq_len, d_k, d_v = 4, 8, 8

# Random queries, keys, values (in practice, these come from projections)
np.random.seed(42)
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

output, weights = attention(Q, K, V)

print("Attention weights (each row sums to 1):")
print(weights.round(3))
# Each row shows how much that position attends to each other position
```

## Visualizing Attention

Attention weights form a matrix where entry $(i, j)$ shows how much position $i$ attends to position $j$:

```
        Position j →
        0     1     2     3
    ┌─────────────────────────┐
  0 │ 0.1   0.7   0.1   0.1   │  ← Position 0 mostly attends to position 1
P 1 │ 0.2   0.2   0.4   0.2   │  ← Position 1 mostly attends to position 2
o 2 │ 0.3   0.3   0.2   0.2   │  ← Position 2 spreads attention evenly
s 3 │ 0.1   0.1   0.1   0.7   │  ← Position 3 mostly attends to itself
i   └─────────────────────────┘
↓
```

**What this shows:** Each row sums to 1 (it's a probability distribution). The model learns to focus on relevant positions—"relevant" emerges from training on the task.

## Attention Patterns in Practice

Trained models develop interpretable attention patterns:

| Pattern | Description | Example |
|---------|-------------|---------|
| **Copying** | Attend strongly to one position | Repeating a word |
| **Uniform** | Attend equally to all positions | Aggregating context |
| **Local** | Attend to nearby positions | Syntax, word order |
| **Positional** | Attend to specific relative positions | Previous token, sentence start |

These patterns emerge without explicit supervision—the model discovers what's useful for the task.

## Computational Complexity

For sequence length $n$ and dimension $d$:

| Operation | Complexity | Memory |
|-----------|------------|--------|
| $QK^T$ | $O(n^2 d)$ | $O(n^2)$ |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| Output | $O(n^2 d)$ | $O(nd)$ |
| **Total** | $O(n^2 d)$ | $O(n^2)$ |

**What this means:** Attention is quadratic in sequence length. Doubling the sequence length quadruples compute and memory. This is why long-context models are expensive and why efficient attention variants (sparse, linear) are active research areas.

## Practical Considerations

### Numerical Stability

Always subtract the max before softmax:

```python
# Bad: can overflow for large scores
exp_x = np.exp(scores)

# Good: mathematically equivalent, numerically stable
exp_x = np.exp(scores - scores.max(axis=-1, keepdims=True))
```

### Masking

For autoregressive models (like GPT), we need **causal masking**—position $i$ can only attend to positions $\leq i$:

```python
def causal_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    # Create causal mask: -inf for future positions
    seq_len = Q.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
    scores = scores + mask

    weights = softmax(scores)
    return weights @ V, weights
```

Setting future positions to $-\infty$ before softmax makes their weights exactly 0.

### Padding

When batching sequences of different lengths, we pad shorter sequences and mask the padding:

```python
def attention_with_padding_mask(Q, K, V, padding_mask):
    """
    padding_mask: (seq_len,) with True for real tokens, False for padding
    """
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    # Set padding positions to -inf
    scores = np.where(padding_mask[None, :], scores, -np.inf)

    weights = softmax(scores)
    return weights @ V, weights
```

## Summary

| Concept | Formula | Code | Purpose |
|---------|---------|------|---------|
| Similarity | $q \cdot k$ | `Q @ K.T` | Measure relevance |
| Scaling | $/ \sqrt{d_k}$ | `/ np.sqrt(d_k)` | Stabilize gradients |
| Weights | softmax | `softmax(scores)` | Normalize to probabilities |
| Output | $\sum \alpha_j v_j$ | `weights @ V` | Weighted retrieval |

**The essential insight:** Attention replaces fixed sequential processing with dynamic, content-based routing. Each position can directly access any other position, weighted by learned relevance. This is why transformers handle long-range dependencies that defeated RNNs—information doesn't need to flow through a bottleneck.

The key innovation isn't the math (dot products and softmax are simple) but the *idea*: let the model decide what's relevant, rather than imposing a fixed processing order.

**Next:** [Self-Attention](self-attention.md) applies attention within a single sequence, letting each token attend to all other tokens.

**Notebook:** [07-attention-from-scratch.ipynb](../notebooks/07-attention-from-scratch.ipynb) for hands-on implementation with visualizations.
