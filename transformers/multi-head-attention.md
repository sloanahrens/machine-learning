# Multi-Head Attention

$$
\boxed{\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O}
$$

**Multi-head attention** runs multiple attention operations in parallel, each focusing on different aspects of the input. One head might capture syntactic relationships, another semantic similarities, another positional patterns. This parallel processing is central to transformer power and efficiency.

Prerequisites: [attention](attention.md), [self-attention](self-attention.md). Code: `numpy`.

---

## Why Multiple Heads?

### The Limitation of Single Attention

Single-head attention computes one set of attention weights:

```
Query: "The cat sat on the mat"
                ↓
        [Single attention]
                ↓
Each position gets ONE weighted combination
```

But words have multiple relationships:
- "cat" relates to "sat" (subject-verb)
- "cat" relates to "The" (determiner)
- "cat" relates to "mat" (rhyme, semantic category)

Single attention must average these different relationship types.

### The Multi-Head Solution

Run attention in parallel with different learned projections:

```
Query: "The cat sat on the mat"
         ↓     ↓     ↓     ↓
      [Head1][Head2][Head3][Head4]
         ↓     ↓     ↓     ↓
      syntax semantic position ...
         ↓     ↓     ↓     ↓
         [    Concatenate    ]
                  ↓
              [Project]
                  ↓
         Combined representation
```

**What this means:** Each head learns to focus on different types of relationships. The model can simultaneously capture syntax, semantics, and other patterns without forcing them into a single attention distribution.

## Architecture

### The Equations

For each head $i$:
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

Combine heads:
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

Where:
- $W_i^Q \in \mathbb{R}^{d \times d_k}$ projects to query space
- $W_i^K \in \mathbb{R}^{d \times d_k}$ projects to key space
- $W_i^V \in \mathbb{R}^{d \times d_v}$ projects to value space
- $W^O \in \mathbb{R}^{hd_v \times d}$ combines head outputs

Typically: $d_k = d_v = d/h$

### Implementation

```python
import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Projections for Q, K, V (combined for efficiency)
        scale = np.sqrt(2 / (d_model + self.d_k))
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale

        # Output projection
        self.W_o = np.random.randn(d_model, d_model) * scale

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_k, d_model]
            mask: Optional [batch_size, 1, seq_len_q, seq_len_k] or broadcastable

        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]

        # Linear projections
        Q = query @ self.W_q  # [batch, seq_q, d_model]
        K = key @ self.W_k    # [batch, seq_k, d_model]
        V = value @ self.W_v  # [batch, seq_k, d_model]

        # Reshape to separate heads
        Q = Q.reshape(batch_size, seq_len_q, self.num_heads, self.d_k)
        K = K.reshape(batch_size, seq_len_k, self.num_heads, self.d_k)
        V = V.reshape(batch_size, seq_len_k, self.num_heads, self.d_k)

        # Transpose to [batch, heads, seq, d_k]
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        # scores: [batch, heads, seq_q, seq_k]

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        attention_weights = softmax(scores, axis=-1)

        # Apply attention to values
        context = attention_weights @ V  # [batch, heads, seq_q, d_k]

        # Concatenate heads
        context = context.transpose(0, 2, 1, 3)  # [batch, seq_q, heads, d_k]
        context = context.reshape(batch_size, seq_len_q, self.d_model)

        # Output projection
        output = context @ self.W_o

        return output, attention_weights
```

### Efficient Parallel Computation

The key insight: all heads can be computed in one matrix operation.

```python
class EfficientMultiHeadAttention:
    """Same computation, clearer parallelism."""

    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Combined projection: projects to all heads at once
        self.qkv_proj = np.random.randn(d_model, 3 * d_model) * np.sqrt(2 / d_model)
        self.out_proj = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)

    def forward(self, x, mask=None):
        """Self-attention: Q, K, V all come from x."""
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V simultaneously
        qkv = x @ self.qkv_proj  # [batch, seq, 3*d_model]

        # Split into Q, K, V
        Q, K, V = np.split(qkv, 3, axis=-1)

        # Reshape for multi-head: [batch, heads, seq, d_k]
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Attention (batched across heads)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        attn = softmax(scores, axis=-1)
        context = attn @ V

        # Recombine heads
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        return context @ self.out_proj, attn
```

## What Each Head Learns

### Empirical Observations

Different heads in trained transformers specialize:

| Head Type | What It Attends To |
|-----------|-------------------|
| Positional | Adjacent tokens, fixed offsets |
| Syntactic | Subject-verb, noun-adjective pairs |
| Semantic | Synonyms, related concepts |
| Rare token | Special tokens, punctuation |
| Broad | Average of many positions |

### Visualizing Head Patterns

```python
def analyze_attention_heads(attention_weights, tokens):
    """
    Analyze what each head focuses on.

    Args:
        attention_weights: [num_heads, seq_len, seq_len]
        tokens: List of token strings
    """
    num_heads = attention_weights.shape[0]

    for head in range(num_heads):
        attn = attention_weights[head]

        # Measure attention entropy (how focused vs distributed)
        entropy = -np.sum(attn * np.log(attn + 1e-10), axis=-1).mean()

        # Measure positional bias
        positions = np.arange(len(tokens))
        pos_diff = np.abs(positions[:, None] - positions[None, :])
        avg_distance = np.sum(attn * pos_diff) / np.sum(attn)

        # Find strongest connections
        max_attn = np.unravel_index(np.argmax(attn), attn.shape)

        print(f"Head {head}:")
        print(f"  Entropy: {entropy:.3f} (lower = more focused)")
        print(f"  Avg attention distance: {avg_distance:.1f} tokens")
        print(f"  Strongest: {tokens[max_attn[0]]} → {tokens[max_attn[1]]}")
        print()
```

### Head Pruning

Research shows many heads can be removed with minimal performance loss:

```python
def head_importance(model, data):
    """
    Estimate importance of each attention head.

    Method: measure performance drop when head is masked.
    """
    num_heads = model.num_heads
    importance = []

    baseline_loss = evaluate(model, data)

    for head in range(num_heads):
        # Zero out this head's output
        mask_head(model, head)
        masked_loss = evaluate(model, data)
        unmask_head(model, head)

        importance.append(masked_loss - baseline_loss)

    return importance
```

**What this means:** Some heads are critical, others redundant. This suggests the model has capacity to learn specialized patterns and fall back on general ones.

## Cross-Attention

### Encoder-Decoder Attention

In transformers with encoder-decoder structure (like T5 or original transformer):

```
Encoder output: [E1, E2, E3, E4]  ← Keys and Values
                   ↑   ↑   ↑   ↑
                   |   |   |   |
Decoder state:  [D1, D2, D3]     ← Queries
```

```python
class CrossAttention(MultiHeadAttention):
    """Attention where Q comes from decoder, K/V from encoder."""

    def forward(self, decoder_states, encoder_outputs, mask=None):
        """
        Args:
            decoder_states: [batch, tgt_len, d_model] - Queries
            encoder_outputs: [batch, src_len, d_model] - Keys and Values
        """
        return super().forward(
            query=decoder_states,
            key=encoder_outputs,
            value=encoder_outputs,
            mask=mask
        )
```

### Attention Types Summary

| Type | Q Source | K, V Source | Use Case |
|------|----------|-------------|----------|
| Self-attention | Same input | Same input | Encoder, decoder |
| Cross-attention | Decoder | Encoder | Translation, summarization |
| Causal self-attention | Same input | Same input (masked) | Autoregressive generation |

## Memory and Compute Costs

### Complexity Analysis

For sequence length $n$, model dimension $d$, and $h$ heads:

| Operation | Time | Memory |
|-----------|------|--------|
| Q, K, V projection | $O(nd^2)$ | $O(nd)$ |
| Attention scores | $O(n^2 d)$ | $O(n^2 h)$ |
| Softmax | $O(n^2 h)$ | $O(n^2 h)$ |
| Value aggregation | $O(n^2 d)$ | $O(nd)$ |
| Output projection | $O(nd^2)$ | $O(nd)$ |

The $O(n^2)$ attention is the bottleneck for long sequences.

### Memory-Efficient Attention

```python
def chunked_attention(Q, K, V, chunk_size=512):
    """
    Process attention in chunks to reduce peak memory.

    Instead of materializing full [n, n] attention matrix,
    process query chunks one at a time.
    """
    seq_len = Q.shape[1]
    outputs = []

    for i in range(0, seq_len, chunk_size):
        q_chunk = Q[:, i:i+chunk_size]

        # Compute attention for this chunk of queries against all keys
        scores = q_chunk @ K.transpose(0, 2, 1) / np.sqrt(Q.shape[-1])
        attn = softmax(scores, axis=-1)
        out_chunk = attn @ V

        outputs.append(out_chunk)

    return np.concatenate(outputs, axis=1)
```

## Positional Information in Multi-Head Attention

### Why Positions Matter

Attention is permutation-equivariant—it doesn't know token order:

```
"Dog bites man" and "Man bites dog" produce same attention scores
(if we ignore position information)
```

### Position Encodings Interact with Heads

With learned or sinusoidal position encodings added to inputs:

```python
def attention_with_positions():
    """
    Position info flows through attention.

    Token embedding: semantic content
    Position encoding: where in sequence

    After projection:
    Q = W_q(token + position)
    K = W_k(token + position)

    Score = Q @ K^T includes:
    - token-token similarity
    - token-position interaction
    - position-position pattern
    """
    pass
```

### Relative Position Encodings

Some architectures use relative positions directly in attention:

```python
class RelativeMultiHeadAttention:
    """Attention with relative position bias."""

    def __init__(self, d_model, num_heads, max_relative_position=32):
        self.mha = MultiHeadAttention(d_model, num_heads)

        # Learnable relative position biases
        # Shape: [num_heads, 2 * max_rel + 1] for positions -max to +max
        self.rel_bias = np.zeros((num_heads, 2 * max_relative_position + 1))
        self.max_rel = max_relative_position

    def get_relative_positions(self, seq_len):
        """Compute relative position matrix."""
        positions = np.arange(seq_len)
        relative = positions[:, None] - positions[None, :]
        relative = np.clip(relative, -self.max_rel, self.max_rel)
        relative += self.max_rel  # Shift to positive indices
        return relative

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Standard attention scores
        output, attn_weights = self.mha.forward(x, x, x, mask)

        # Add relative position bias
        rel_pos = self.get_relative_positions(seq_len)
        bias = self.rel_bias[:, rel_pos]  # [heads, seq, seq]

        # Would need to integrate this into attention computation
        # (simplified here for illustration)

        return output, attn_weights
```

**What this means:** Relative position encodings let attention learn patterns like "the token 2 positions back" rather than "the token at position 5". This generalizes better to different sequence lengths.

## Putting It Together

### Multi-Head Attention in a Transformer Block

```python
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Feed-forward network
        self.ff1 = np.random.randn(d_model, d_ff) * np.sqrt(2 / d_model)
        self.ff2 = np.random.randn(d_ff, d_model) * np.sqrt(2 / d_ff)

        self.dropout_rate = dropout

    def forward(self, x, mask=None, training=True):
        # Self-attention with residual
        attn_out, attn_weights = self.attention.forward(x, x, x, mask)
        if training:
            attn_out = dropout(attn_out, self.dropout_rate)
        x = self.norm1.forward(x + attn_out)

        # Feed-forward with residual
        ff_out = np.maximum(0, x @ self.ff1) @ self.ff2  # ReLU activation
        if training:
            ff_out = dropout(ff_out, self.dropout_rate)
        x = self.norm2.forward(x + ff_out)

        return x, attn_weights

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta

def dropout(x, rate):
    if rate == 0:
        return x
    mask = np.random.binomial(1, 1 - rate, x.shape) / (1 - rate)
    return x * mask
```

## Summary

| Concept | Description |
|---------|-------------|
| Multiple heads | Parallel attention with different projections |
| Head specialization | Different heads learn different patterns |
| Concatenation | Combine head outputs |
| Output projection | Mix head information |
| Cross-attention | Q from one source, K/V from another |

**The essential insight:** Multi-head attention provides representational diversity. Instead of one attention pattern, the model learns multiple complementary patterns. This is analogous to CNN filters—each head is like a different "filter" for relationships. The combination of heads gives transformers their remarkable ability to capture complex linguistic patterns.

**Next:** [BERT](bert.md) covers masked language modeling and bidirectional transformers.
