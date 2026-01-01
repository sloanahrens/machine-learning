# The Transformer Architecture

```math
\boxed{\text{Transformer} = \text{Embedding} + \text{Positional Encoding} + N \times \text{Block} + \text{Output Head}}
```

The **transformer** is an architecture that processes sequences using attention instead of recurrence. It consists of stacked blocks, each containing self-attention followed by a feedforward network. This simple, parallelizable design has become the foundation of modern NLP.

Prerequisites: [self-attention](self-attention.md), [activation-functions](../neural-networks/activation-functions.md), [backpropagation](../neural-networks/backpropagation.md). Code: `numpy`.

---

## The Original Transformer

The 2017 paper "Attention Is All You Need" introduced a sequence-to-sequence model with two parts:

```
                     Encoder                      Decoder
                 ┌─────────────┐              ┌─────────────┐
                 │   Block N   │              │   Block N   │
                 │     ...     │              │     ...     │
                 │   Block 2   │   ───────→   │   Block 2   │
                 │   Block 1   │   (cross)    │   Block 1   │
                 ├─────────────┤              ├─────────────┤
                 │    + PE     │              │    + PE     │
                 │  Embedding  │              │  Embedding  │
                 └─────────────┘              └─────────────┘
                      ↑                            ↑
               Source tokens               Target tokens (shifted)
```

- **Encoder:** Processes the source sequence, producing contextualized representations
- **Decoder:** Generates the target sequence, attending to encoder outputs

For language modeling (GPT-style), we only need the decoder. For understanding tasks (BERT-style), we only need the encoder. Let's examine both.

## Encoder Block

Each encoder block has two sub-layers:

```
                    Input
                      ↓
           ┌──────────────────┐
           │  Self-Attention  │
           └────────┬─────────┘
                    │
              Add & Norm ←──── Residual connection
                    │
           ┌────────┴─────────┐
           │   Feed-Forward   │
           └────────┬─────────┘
                    │
              Add & Norm ←──── Residual connection
                    │
                  Output
```

### In Code

```python
import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def gelu(x):
    """GELU activation (used in modern transformers)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

class EncoderBlock:
    def __init__(self, d_model, n_heads, d_ff):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feedforward hidden dimension (typically 4 * d_model)
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Multi-head attention parameters
        self.W_Q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_K = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_V = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_O = np.random.randn(d_model, d_model) / np.sqrt(d_model)

        # Layer norm parameters
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

        # Feedforward parameters
        self.W1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros(d_model)

    def multi_head_attention(self, X, mask=None):
        """Multi-head self-attention."""
        seq_len = X.shape[0]

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        # Reshape for multi-head
        Q = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)

        # Scaled dot-product attention
        scores = np.einsum('hid,hjd->hij', Q, K) / np.sqrt(self.d_k)

        if mask is not None:
            scores = np.where(mask, scores, -np.inf)

        weights = softmax(scores, axis=-1)
        context = np.einsum('hij,hjd->hid', weights, V)

        # Concatenate heads
        context = context.transpose(1, 0, 2).reshape(seq_len, self.d_model)

        return context @ self.W_O

    def feedforward(self, X):
        """Position-wise feedforward network."""
        hidden = gelu(X @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2

    def forward(self, X, mask=None):
        """
        Forward pass through encoder block.

        Args:
            X: Input, shape (seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output, shape (seq_len, d_model)
        """
        # Self-attention with residual and layer norm
        attn_out = self.multi_head_attention(X, mask)
        X = layer_norm(X + attn_out, self.ln1_gamma, self.ln1_beta)

        # Feedforward with residual and layer norm
        ff_out = self.feedforward(X)
        X = layer_norm(X + ff_out, self.ln2_gamma, self.ln2_beta)

        return X
```

## Decoder Block

The decoder block adds **cross-attention** to attend to encoder outputs:

```
                    Input
                      ↓
           ┌──────────────────┐
           │ Masked Self-Attn │ ← Causal mask
           └────────┬─────────┘
              Add & Norm
                    │
           ┌────────┴─────────┐
           │  Cross-Attention │ ← Attends to encoder
           └────────┬─────────┘
              Add & Norm
                    │
           ┌────────┴─────────┐
           │   Feed-Forward   │
           └────────┬─────────┘
              Add & Norm
                    │
                  Output
```

### Cross-Attention

In cross-attention:
- **Queries** come from the decoder (what we're generating)
- **Keys and Values** come from the encoder (what we're reading)

```math
\text{CrossAttention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}})
```

**What this means:** Each decoder position can look at all encoder positions. This is how translation works—when generating "bonjour", the decoder attends to "hello" in the encoder's representation.

## Layer Normalization

Transformers use **layer normalization**, not batch normalization:

```math
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
```

where $\mu$ and $\sigma^2$ are computed over the feature dimension for each position independently.

### Why Layer Norm?

| Normalization | Computes stats over | Use case |
|---------------|---------------------|----------|
| Batch Norm | Batch dimension | CNNs, fixed batch size |
| Layer Norm | Feature dimension | Transformers, variable length |

Layer norm doesn't depend on batch size, making it suitable for:
- Variable-length sequences
- Small batches during inference
- Autoregressive generation

### Pre-Norm vs Post-Norm

The original transformer used **post-norm** (normalize after residual):
```python
X = layer_norm(X + sublayer(X))
```

Most modern transformers use **pre-norm** (normalize before sublayer):
```python
X = X + sublayer(layer_norm(X))
```

Pre-norm is more stable for deep networks and doesn't require careful learning rate warmup.

## The Feedforward Network

Each position goes through the same feedforward network independently:

```math
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
```

### Dimensions

If $d_{\text{model}} = 768$, typically $d_{\text{ff}} = 4 \times d_{\text{model}} = 3072$.

```python
def feedforward(x, W1, b1, W2, b2):
    """
    Position-wise feedforward network.

    x: (seq_len, d_model)
    W1: (d_model, d_ff)
    W2: (d_ff, d_model)
    """
    hidden = gelu(x @ W1 + b1)  # (seq_len, d_ff)
    return hidden @ W2 + b2      # (seq_len, d_model)
```

**What this means:** The FFN processes each position independently—it's "position-wise." All inter-position communication happens through attention. The FFN is where the model stores factual knowledge (who is the president, what year was X invented, etc.).

## Positional Encoding

Transformers add position information to embeddings:

```math
\text{input} = \text{Embedding}(x) + \text{PositionalEncoding}(\text{position})
```

### Sinusoidal (Original)

```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
```
```math
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
```

```python
def sinusoidal_positional_encoding(max_len, d_model):
    """Generate sinusoidal positional encodings."""
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe
```

### Learned (BERT, GPT)

Simply learn a lookup table:

```python
class LearnedPositionalEncoding:
    def __init__(self, max_len, d_model):
        self.pe = np.random.randn(max_len, d_model) * 0.02

    def forward(self, seq_len):
        return self.pe[:seq_len]
```

## Full Transformer Model

### Encoder-Only (BERT-style)

```python
class TransformerEncoder:
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len):
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.positional_encoding = sinusoidal_positional_encoding(max_len, d_model)

        self.layers = [
            EncoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]

        self.ln_f = (np.ones(d_model), np.zeros(d_model))  # Final layer norm

    def forward(self, token_ids, mask=None):
        """
        Args:
            token_ids: (seq_len,) integer token IDs
            mask: Optional attention mask

        Returns:
            Hidden states, shape (seq_len, d_model)
        """
        seq_len = len(token_ids)

        # Embed tokens and add position
        X = self.embedding[token_ids] + self.positional_encoding[:seq_len]

        # Pass through encoder blocks
        for layer in self.layers:
            X = layer.forward(X, mask)

        # Final layer norm
        X = layer_norm(X, *self.ln_f)

        return X
```

### Decoder-Only (GPT-style)

```python
class TransformerDecoder:
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len):
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.positional_encoding = sinusoidal_positional_encoding(max_len, d_model)

        self.layers = [
            EncoderBlock(d_model, n_heads, d_ff)  # Same as encoder, but with causal mask
            for _ in range(n_layers)
        ]

        self.ln_f = (np.ones(d_model), np.zeros(d_model))
        self.lm_head = np.random.randn(d_model, vocab_size) / np.sqrt(d_model)

    def forward(self, token_ids):
        """
        Args:
            token_ids: (seq_len,) integer token IDs

        Returns:
            Logits for next token, shape (seq_len, vocab_size)
        """
        seq_len = len(token_ids)

        # Create causal mask
        causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))

        # Embed tokens and add position
        X = self.embedding[token_ids] + self.positional_encoding[:seq_len]

        # Pass through decoder blocks
        for layer in self.layers:
            X = layer.forward(X, mask=causal_mask)

        # Final layer norm
        X = layer_norm(X, *self.ln_f)

        # Project to vocabulary
        logits = X @ self.lm_head

        return logits
```

## Model Sizes

Common configurations:

| Model | d_model | n_heads | n_layers | d_ff | Parameters |
|-------|---------|---------|----------|------|------------|
| GPT-2 Small | 768 | 12 | 12 | 3072 | 117M |
| GPT-2 Medium | 1024 | 16 | 24 | 4096 | 345M |
| GPT-2 Large | 1280 | 20 | 36 | 5120 | 762M |
| GPT-2 XL | 1600 | 25 | 48 | 6400 | 1.5B |
| GPT-3 | 12288 | 96 | 96 | 49152 | 175B |

### Parameter Counting

```python
def count_parameters(vocab_size, d_model, n_heads, d_ff, n_layers, max_len):
    """Count parameters in a GPT-style transformer."""

    # Embedding
    embed_params = vocab_size * d_model

    # Positional encoding (if learned)
    pos_params = max_len * d_model

    # Per-layer parameters
    attn_params = 4 * d_model * d_model  # W_Q, W_K, W_V, W_O
    ff_params = 2 * d_model * d_ff       # W1, W2
    ln_params = 4 * d_model               # 2 layer norms per block
    layer_params = attn_params + ff_params + ln_params

    # Final layer norm + output projection (often tied with embedding)
    final_params = 2 * d_model + vocab_size * d_model

    total = embed_params + pos_params + n_layers * layer_params + final_params

    return total

# GPT-2 Small
params = count_parameters(
    vocab_size=50257, d_model=768, n_heads=12,
    d_ff=3072, n_layers=12, max_len=1024
)
print(f"GPT-2 Small: {params / 1e6:.1f}M parameters")  # ~117M
```

## Residual Connections

Every sub-layer has a residual connection:

```math
\text{output} = x + \text{sublayer}(x)
```

### Why Residuals?

1. **Gradient flow:** Gradients can bypass transformations, preventing vanishing gradients
2. **Identity initialization:** Early in training, layers can approximate identity
3. **Depth scaling:** Enable training very deep networks (96+ layers)

```
Without residual:  x → sublayer → output
                   Gradient must flow through sublayer

With residual:     x ──────────────────┐
                   │                   │
                   └→ sublayer → (+) → output
                   Gradient can flow directly through (+)
```

## Dropout

Transformers apply dropout in three places:

1. **After attention weights:** `dropout(softmax(scores))`
2. **After each sub-layer:** Before residual addition
3. **On embeddings:** `dropout(embedding + position)`

```python
def dropout(x, p, training=True):
    """
    Dropout during training.

    Args:
        x: Input
        p: Dropout probability
        training: Whether in training mode
    """
    if not training or p == 0:
        return x

    mask = np.random.binomial(1, 1-p, size=x.shape) / (1-p)
    return x * mask
```

Typical values: p = 0.1 for base models, p = 0.3 for large models.

## Training Details

### Loss Function

For language modeling, cross-entropy loss over vocabulary:

```math
L = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{<t})
```

```python
def cross_entropy_loss(logits, targets):
    """
    Cross-entropy loss for language modeling.

    Args:
        logits: (seq_len, vocab_size)
        targets: (seq_len,) target token IDs
    """
    probs = softmax(logits)
    log_probs = np.log(probs[np.arange(len(targets)), targets] + 1e-10)
    return -log_probs.mean()
```

### Learning Rate Schedule

Transformers typically use warmup + decay:

```python
def learning_rate_schedule(step, d_model, warmup_steps=4000):
    """Original transformer learning rate schedule."""
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)
```

Modern practice: Linear warmup then cosine decay.

### Optimizer

Adam with specific hyperparameters:
- $\beta_1 = 0.9$, $\beta_2 = 0.98$ (or 0.999)
- $\epsilon = 10^{-9}$
- Weight decay (AdamW): 0.01 - 0.1

## Putting It All Together

```python
class GPT:
    """Minimal GPT implementation."""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len):
        self.model = TransformerDecoder(
            vocab_size, d_model, n_heads, d_ff, n_layers, max_len
        )
        self.vocab_size = vocab_size

    def forward(self, token_ids):
        """Get logits for next token prediction."""
        return self.model.forward(token_ids)

    def generate(self, prompt_ids, max_new_tokens, temperature=1.0):
        """Autoregressive generation."""
        token_ids = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = self.forward(np.array(token_ids))[-1]

            # Apply temperature
            logits = logits / temperature

            # Sample from distribution
            probs = softmax(logits)
            next_token = np.random.choice(self.vocab_size, p=probs)

            token_ids.append(next_token)

        return token_ids

# Create a small GPT
gpt = GPT(
    vocab_size=1000,
    d_model=256,
    n_heads=4,
    d_ff=1024,
    n_layers=4,
    max_len=128
)

# Generate some tokens
prompt = np.array([1, 2, 3, 4, 5])  # Random prompt
generated = gpt.generate(prompt, max_new_tokens=10)
print(f"Generated: {generated}")
```

## Common Variations

### Pre-Norm vs Post-Norm

```python
# Post-norm (original)
X = layer_norm(X + sublayer(X))

# Pre-norm (GPT-2, modern)
X = X + sublayer(layer_norm(X))
```

### Activation Functions

| Model | Activation |
|-------|------------|
| Original Transformer | ReLU |
| GPT-2, BERT | GELU |
| LLaMA | SwiGLU |

### Attention Variations

| Variant | Key Feature | Used In |
|---------|-------------|---------|
| Multi-Query | Shared K,V across heads | Inference optimization |
| Grouped-Query | K,V shared within groups | LLaMA 2 |
| Flash Attention | IO-aware algorithm | Most modern models |
| Sliding Window | Limited context per layer | Mistral |

## Summary

| Component | Formula | Purpose |
|-----------|---------|---------|
| Embedding | $E(x)$ | Convert tokens to vectors |
| Position | $E(x) + PE(pos)$ | Inject order information |
| Self-Attention | $\text{softmax}(QK^T/\sqrt{d_k})V$ | Token interaction |
| FFN | $W_2 \cdot \text{GELU}(W_1 x)$ | Per-position processing |
| Layer Norm | $\gamma \cdot \frac{x-\mu}{\sigma} + \beta$ | Stabilize training |
| Residual | $x + f(x)$ | Enable gradient flow |

**The essential insight:** A transformer is surprisingly simple: embeddings + position → stack of (attention + feedforward) blocks → output projection. The magic is in the attention mechanism, which lets every position communicate with every other position. Everything else (layer norm, residuals, feedforward) is infrastructure that makes training stable and effective.

The architecture scales remarkably well. The same basic design works from 100M to 100B+ parameters. Improvements mostly come from data, compute, and training techniques—the architecture itself has remained largely unchanged since 2017.

**Next:** [GPT](gpt.md) to see how decoder-only transformers enable language modeling at scale.

**Notebook:** [09-transformer-from-scratch.ipynb](../notebooks/09-transformer-from-scratch.ipynb) builds a complete transformer and trains it on a small task.
