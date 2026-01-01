# Self-Attention

```math
\boxed{\text{SelfAttention}(X) = \text{softmax}\left(\frac{(XW_Q)(XW_K)^T}{\sqrt{d_k}}\right)(XW_V)}
```

**Self-attention** applies attention within a single sequence—each position attends to all positions in the same sequence, including itself. This is the core mechanism that lets transformers model relationships between any pair of tokens, regardless of distance.

Prerequisites: [attention](attention.md) (query-key-value framework), [linear-algebra](../math-foundations/linear-algebra.md). Code: `numpy`.

---

## From Attention to Self-Attention

In general attention, queries come from one sequence and keys/values from another (e.g., decoder attending to encoder outputs).

In **self-attention**, queries, keys, and values all come from the *same* sequence:

```math
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
```

where $X \in \mathbb{R}^{n \times d}$ is the input sequence (n tokens, d-dimensional embeddings).

**What this means:** Each token asks "which other tokens in this sequence are relevant to me?" and gets a weighted combination of all tokens as its new representation. The word "self" means the sequence is talking to itself.

## Why Self-Attention Works

### The Key Insight

Consider the sentence: "The animal didn't cross the street because it was too tired."

What does "it" refer to? A human knows "it" = "animal" (not "street") because animals get tired, streets don't. To resolve this, a model must:

1. Identify that "it" needs resolution
2. Consider all candidate referents ("animal", "street")
3. Use world knowledge to select the right one

Self-attention enables this: the representation of "it" can directly attend to "animal" and "street", weighting them by relevance.

### Distance Doesn't Matter

In RNNs, connecting distant tokens requires many sequential steps. In self-attention, any token can attend to any other token in a single operation:

```
RNN: Token 1 → Token 2 → Token 3 → ... → Token 100
     (99 sequential steps to connect 1 and 100)

Self-Attention: Token 1 ←→ Token 100
               (direct connection via attention)
```

**What this means:** Self-attention has constant path length between any two positions. Long-range dependencies are just as easy to learn as short-range ones.

## The Full Computation

### Step by Step

Given input $X \in \mathbb{R}^{n \times d}$ (n tokens, d dimensions):

**Step 1: Project to Q, K, V**

```math
Q = XW_Q \in \mathbb{R}^{n \times d_k}
```
```math
K = XW_K \in \mathbb{R}^{n \times d_k}
```
```math
V = XW_V \in \mathbb{R}^{n \times d_v}
```

**Step 2: Compute attention scores**

```math
\text{scores} = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}
```

**Step 3: Apply softmax**

```math
\text{weights} = \text{softmax}(\text{scores}) \in \mathbb{R}^{n \times n}
```

**Step 4: Compute output**

```math
\text{output} = \text{weights} \cdot V \in \mathbb{R}^{n \times d_v}
```

### Dimensions Visualized

```
Input X:          Queries Q:        Keys K:           Values V:
(n × d)           (n × d_k)         (n × d_k)         (n × d_v)
┌─────────┐       ┌───────┐         ┌───────┐         ┌───────┐
│ token 1 │       │ q_1   │         │ k_1   │         │ v_1   │
│ token 2 │  W_Q  │ q_2   │   W_K   │ k_2   │   W_V   │ v_2   │
│ token 3 │ ───→  │ q_3   │  ───→   │ k_3   │  ───→   │ v_3   │
│   ...   │       │  ...  │         │  ...  │         │  ...  │
│ token n │       │ q_n   │         │ k_n   │         │ v_n   │
└─────────┘       └───────┘         └───────┘         └───────┘

Scores = Q @ K.T:        Weights:              Output:
(n × n)                  (n × n)               (n × d_v)
┌─────────────────┐      ┌─────────────────┐   ┌───────┐
│ q1·k1 q1·k2 ... │      │ α11  α12  ...   │   │ out_1 │
│ q2·k1 q2·k2 ... │ soft │ α21  α22  ...   │ @ │ out_2 │
│  ...   ...  ... │ max  │ ...  ...  ...   │ V │  ...  │
│ qn·k1 qn·k2 ... │ ───→ │ αn1  αn2  ...   │   │ out_n │
└─────────────────┘      └─────────────────┘   └───────┘
```

## In Code

```python
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    exp_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

class SelfAttention:
    def __init__(self, d_model, d_k, d_v):
        """
        Args:
            d_model: Input/output dimension
            d_k: Key/query dimension
            d_v: Value dimension
        """
        # Xavier initialization
        self.W_Q = np.random.randn(d_model, d_k) / np.sqrt(d_model)
        self.W_K = np.random.randn(d_model, d_k) / np.sqrt(d_model)
        self.W_V = np.random.randn(d_model, d_v) / np.sqrt(d_model)
        self.d_k = d_k

    def forward(self, X, mask=None):
        """
        Self-attention forward pass.

        Args:
            X: Input, shape (seq_len, d_model)
            mask: Optional mask, shape (seq_len, seq_len)
                  True = attend, False = don't attend

        Returns:
            output: Shape (seq_len, d_v)
            weights: Attention weights, shape (seq_len, seq_len)
        """
        # Project to Q, K, V
        Q = X @ self.W_Q  # (seq_len, d_k)
        K = X @ self.W_K  # (seq_len, d_k)
        V = X @ self.W_V  # (seq_len, d_v)

        # Compute attention scores
        scores = Q @ K.T / np.sqrt(self.d_k)  # (seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)

        # Softmax to get weights
        weights = softmax(scores, axis=-1)  # (seq_len, seq_len)

        # Weighted sum of values
        output = weights @ V  # (seq_len, d_v)

        return output, weights

# Example usage
d_model, d_k, d_v = 64, 32, 32
seq_len = 10

# Create self-attention layer
sa = SelfAttention(d_model, d_k, d_v)

# Random input sequence
X = np.random.randn(seq_len, d_model)

# Forward pass
output, weights = sa.forward(X)

print(f"Input shape: {X.shape}")       # (10, 64)
print(f"Output shape: {output.shape}") # (10, 32)
print(f"Weights shape: {weights.shape}") # (10, 10)
print(f"Each row sums to: {weights.sum(axis=1)}")  # All 1s
```

## Causal (Masked) Self-Attention

For autoregressive models like GPT, each position should only attend to previous positions—you can't "see the future" when generating text.

### The Causal Mask

```
Position:  0   1   2   3   4
         ┌───────────────────┐
       0 │ ✓   ✗   ✗   ✗   ✗ │  Position 0 only sees itself
       1 │ ✓   ✓   ✗   ✗   ✗ │  Position 1 sees 0, 1
       2 │ ✓   ✓   ✓   ✗   ✗ │  Position 2 sees 0, 1, 2
       3 │ ✓   ✓   ✓   ✓   ✗ │  Position 3 sees 0, 1, 2, 3
       4 │ ✓   ✓   ✓   ✓   ✓ │  Position 4 sees all
         └───────────────────┘
```

### In Code

```python
def causal_self_attention(X, W_Q, W_K, W_V, d_k):
    """
    Causal self-attention for autoregressive models.
    """
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    seq_len = X.shape[0]

    # Compute scores
    scores = Q @ K.T / np.sqrt(d_k)

    # Create causal mask (lower triangular)
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))

    # Apply mask: set future positions to -inf
    scores = np.where(mask, scores, -np.inf)

    # Softmax (future positions become 0)
    weights = softmax(scores, axis=-1)

    return weights @ V, weights

# Test causal attention
output, weights = causal_self_attention(X, sa.W_Q, sa.W_K, sa.W_V, d_k)

print("Causal attention weights (lower triangular):")
print(weights[:5, :5].round(2))  # First 5x5 block
```

**What this means:** During training, we process entire sequences in parallel, but the causal mask ensures each position only "sees" its past. During generation, we produce one token at a time, naturally never seeing the future.

## Multi-Head Self-Attention

A single attention head can only focus on one type of relationship. **Multi-head attention** runs multiple attention heads in parallel, each learning different patterns:

```math
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O
```

where each head is:

```math
\text{head}_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)
```

### Why Multiple Heads?

Different heads learn different things:
- **Head 1:** Subject-verb agreement ("dog runs", "dogs run")
- **Head 2:** Coreference ("it" → "animal")
- **Head 3:** Positional patterns (attend to previous token)
- **Head 4:** Semantic similarity (synonyms)

### In Code

```python
class MultiHeadSelfAttention:
    def __init__(self, d_model, n_heads):
        """
        Multi-head self-attention.

        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
        """
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.d_model = d_model

        # Projection matrices for all heads (packed together)
        self.W_Q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_K = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_V = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_O = np.random.randn(d_model, d_model) / np.sqrt(d_model)

    def forward(self, X, mask=None):
        """
        Args:
            X: Input, shape (seq_len, d_model)
            mask: Optional causal mask

        Returns:
            output: Shape (seq_len, d_model)
        """
        seq_len = X.shape[0]

        # Project to Q, K, V
        Q = X @ self.W_Q  # (seq_len, d_model)
        K = X @ self.W_K
        V = X @ self.W_V

        # Reshape for multi-head: (seq_len, n_heads, d_k)
        Q = Q.reshape(seq_len, self.n_heads, self.d_k)
        K = K.reshape(seq_len, self.n_heads, self.d_k)
        V = V.reshape(seq_len, self.n_heads, self.d_k)

        # Transpose for batched attention: (n_heads, seq_len, d_k)
        Q = Q.transpose(1, 0, 2)
        K = K.transpose(1, 0, 2)
        V = V.transpose(1, 0, 2)

        # Compute attention for all heads at once
        # scores: (n_heads, seq_len, seq_len)
        scores = np.einsum('hid,hjd->hij', Q, K) / np.sqrt(self.d_k)

        # Apply causal mask if provided
        if mask is not None:
            scores = np.where(mask[None, :, :], scores, -np.inf)

        weights = softmax(scores, axis=-1)

        # Compute output: (n_heads, seq_len, d_k)
        head_outputs = np.einsum('hij,hjd->hid', weights, V)

        # Concatenate heads: (seq_len, d_model)
        concat = head_outputs.transpose(1, 0, 2).reshape(seq_len, self.d_model)

        # Final projection
        output = concat @ self.W_O

        return output, weights

# Example: 8-head attention
mhsa = MultiHeadSelfAttention(d_model=64, n_heads=8)
output, weights = mhsa.forward(X)

print(f"Output shape: {output.shape}")   # (10, 64)
print(f"Weights shape: {weights.shape}") # (8, 10, 10) - one per head
```

### Head Dimension

If `d_model = 512` and `n_heads = 8`, each head has `d_k = 64`.

The total computation is similar to single-head attention with the full dimension, but:
- Multiple attention patterns can be learned in parallel
- Each head's smaller dimension forces it to specialize

## Positional Information

Self-attention is **permutation equivariant**—it treats all positions equally. Without positional encodings, the model can't distinguish "dog bites man" from "man bites dog."

### The Problem

Self-attention computes:
```math
\alpha_{ij} = \text{softmax}(q_i \cdot k_j)
```

This depends only on *content* ($q_i$, $k_j$), not *position* ($i$, $j$). Shuffling the sequence shuffles the output in the same way.

**What this means:** Self-attention sees a *bag of tokens*, not an *ordered sequence*. We need to inject position information explicitly.

### Solutions

1. **Sinusoidal positional encoding** (original Transformer):
```math
   PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})
```
```math
   PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})
```

2. **Learned positional embeddings** (BERT, GPT):
```math
   X' = X + P
```
   where $P \in \mathbb{R}^{n_{max} \times d}$ is learned.

3. **Relative positional encoding** (Transformer-XL, T5):
   Add position-dependent bias to attention scores.

4. **Rotary positional embedding (RoPE)** (LLaMA, modern LLMs):
   Rotate query and key vectors based on position.

```python
def sinusoidal_position_encoding(seq_len, d_model):
    """Generate sinusoidal positional encodings."""
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe

# Add positional encoding to input
pe = sinusoidal_position_encoding(seq_len, d_model)
X_with_position = X + pe
```

## Self-Attention vs Convolution vs Recurrence

| Property | Self-Attention | Convolution | Recurrence |
|----------|---------------|-------------|------------|
| Long-range | O(1) path length | O(n/k) path length | O(n) path length |
| Parallelization | Full | Full | Sequential |
| Computation | O(n²d) | O(knd) | O(nd²) |
| Inductive bias | None (flexible) | Local patterns | Sequential processing |

**What this means:**
- Self-attention is most flexible but most expensive
- Convolutions are efficient for local patterns
- Recurrence is efficient for sequential patterns but hard to parallelize

## Interpreting Self-Attention

Attention weights reveal what the model is "looking at":

```python
def visualize_attention(tokens, weights, layer=0, head=0):
    """
    Visualize attention patterns.

    Args:
        tokens: List of token strings
        weights: Attention weights (n_heads, seq_len, seq_len)
        layer: Which layer (for labeling)
        head: Which head to visualize
    """
    w = weights[head]
    n = len(tokens)

    print(f"Layer {layer}, Head {head}:")
    print("       " + " ".join(f"{t:>6}" for t in tokens))
    for i, tok in enumerate(tokens):
        row = " ".join(f"{w[i,j]:6.2f}" for j in range(n))
        print(f"{tok:>6} {row}")

# Example
tokens = ["The", "cat", "sat", "on", "mat"]
# Simulated weights showing "mat" attending strongly to "The"
```

### What to Look For

- **Diagonal patterns:** Attending to self or nearby tokens
- **Vertical stripes:** All positions attending to one token (often special tokens like [CLS])
- **Horizontal stripes:** One position attending broadly
- **Off-diagonal spikes:** Long-range dependencies

## Common Pitfalls

### 1. Forgetting the Scale Factor

```python
# Wrong: dot products grow with dimension
scores = Q @ K.T

# Right: scale to keep variance stable
scores = Q @ K.T / np.sqrt(d_k)
```

### 2. Incorrect Mask Application

```python
# Wrong: masking after softmax
weights = softmax(scores)
weights = weights * mask  # Doesn't work!

# Right: mask before softmax with -inf
scores = np.where(mask, scores, -np.inf)
weights = softmax(scores)
```

### 3. Dimension Mismatches with Multi-Head

```python
# d_model must be divisible by n_heads
assert d_model % n_heads == 0, f"{d_model} not divisible by {n_heads}"
```

## Computational Considerations

### Memory: The Quadratic Problem

For sequence length $n$:
- Attention weights matrix: $n \times n$ floats
- For n = 4096, that's 16M floats = 64MB per head
- With 32 heads and batch size 32: ~65GB just for attention weights

This is why long-context models use techniques like:
- **Flash Attention:** Fused kernels, reduced memory
- **Sparse Attention:** Only compute subset of attention
- **Linear Attention:** Approximate attention in O(n)

### Training vs Inference

| Phase | Behavior | Memory |
|-------|----------|--------|
| Training | Process full sequence at once | High (store all activations) |
| Inference | Can process token by token | Lower (KV cache) |

**KV Cache:** During autoregressive generation, we cache K and V from previous tokens, only computing Q for the new token. This reduces redundant computation from O(n²) to O(n) per token.

## Summary

| Concept | Formula | Purpose |
|---------|---------|---------|
| Self-attention | $\text{softmax}(QK^T/\sqrt{d_k})V$ | Relate all positions to each other |
| Causal mask | Lower triangular | Prevent seeing future |
| Multi-head | $\text{Concat(heads)}W_O$ | Learn multiple patterns |
| Position encoding | $X + PE$ | Inject order information |

**The essential insight:** Self-attention lets every position communicate with every other position in a single step. The "self" means queries, keys, and values all come from the same sequence. Combined with position encodings and multi-head parallelism, this creates a powerful mechanism for learning relationships in sequences—the foundation of modern transformers.

The attention pattern matrix itself is interpretable: entry $(i, j)$ shows how much token $i$ attends to token $j$. Trained models develop meaningful patterns—syntactic, semantic, and positional—without explicit supervision.

**Next:** [Multi-Head Attention](multi-head-attention.md) dives deeper into why multiple heads work and what different heads learn.

**Notebook:** [08-self-attention-visualized.ipynb](../notebooks/08-self-attention-visualized.ipynb) explores attention patterns on real text.
