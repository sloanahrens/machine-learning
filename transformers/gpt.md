# GPT: Generative Pre-trained Transformer

```math
\boxed{P(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} P(x_i | x_1, \ldots, x_{i-1})}
```

**GPT** (Generative Pre-trained Transformer) is a decoder-only transformer trained to predict the next token. This simple objective—predict what comes next—turns out to be extraordinarily powerful when scaled up. GPT-style models form the foundation of modern large language models.

Prerequisites: [transformer-architecture](transformer-architecture.md), [probability](../math-foundations/probability.md) (cross-entropy). Code: `numpy`.

---

## The Core Idea

GPT has one job: given a sequence of tokens, predict the next one.

```math
P(x_t | x_1, \ldots, x_{t-1})
```

Train on vast amounts of text. Scale up. Remarkable capabilities emerge.

**What this means:** GPT doesn't explicitly learn grammar, facts, or reasoning. It learns to predict text—and to predict text well, you need to implicitly learn all those things. The "G" in GPT is key: it *generates*, one token at a time.

## Autoregressive Language Modeling

### The Factorization

Any sequence probability can be factored autoregressively:

```math
P(x_1, x_2, \ldots, x_n) = P(x_1) \cdot P(x_2|x_1) \cdot P(x_3|x_1,x_2) \cdots P(x_n|x_1,\ldots,x_{n-1})
```

GPT models each conditional distribution with a transformer.

### Training

During training:
1. Take a sequence of tokens
2. Compute probability of each token given previous tokens
3. Maximize log-likelihood (minimize cross-entropy loss)

```python
import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def gpt_loss(logits, targets):
    """
    Cross-entropy loss for GPT.

    Args:
        logits: (seq_len, vocab_size) model predictions
        targets: (seq_len,) target token IDs

    The target at position i is the input at position i+1.
    """
    probs = softmax(logits)
    # Get probability assigned to correct next token
    correct_probs = probs[np.arange(len(targets)), targets]
    # Negative log likelihood
    return -np.log(correct_probs + 1e-10).mean()

# Example
vocab_size = 1000
seq_len = 10

logits = np.random.randn(seq_len, vocab_size)
targets = np.random.randint(0, vocab_size, seq_len)

loss = gpt_loss(logits, targets)
print(f"Loss: {loss:.3f}")  # Random: ~log(vocab_size) ≈ 6.9
```

### The Shift

An important detail: at position $i$, we predict token $i+1$.

```
Input:    [The] [cat] [sat] [on] [the]
Targets:  [cat] [sat] [on] [the] [mat]
          ↑     ↑     ↑    ↑     ↑
          Predict next token at each position
```

In code:
```python
def prepare_training_data(tokens):
    """Prepare input-target pairs for GPT training."""
    inputs = tokens[:-1]   # All but last
    targets = tokens[1:]   # All but first
    return inputs, targets

# "The cat sat on the mat" → tokens [1, 2, 3, 4, 1, 5]
tokens = np.array([1, 2, 3, 4, 1, 5])
inputs, targets = prepare_training_data(tokens)
# inputs:  [1, 2, 3, 4, 1]
# targets: [2, 3, 4, 1, 5]
```

## Architecture Details

GPT uses a decoder-only transformer with causal (masked) self-attention.

### The Causal Mask

Each position can only attend to previous positions:

```
Position:  0    1    2    3    4
         ┌─────────────────────────┐
       0 │ ✓    ✗    ✗    ✗    ✗   │
       1 │ ✓    ✓    ✗    ✗    ✗   │
       2 │ ✓    ✓    ✓    ✗    ✗   │
       3 │ ✓    ✓    ✓    ✓    ✗   │
       4 │ ✓    ✓    ✓    ✓    ✓   │
         └─────────────────────────┘
```

**Why?** During generation, we don't have future tokens. Training must match this constraint.

### Complete Forward Pass

```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

class GPTBlock:
    """Single GPT block (pre-norm variant)."""

    def __init__(self, d_model, n_heads, d_ff):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Attention
        self.W_Q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_K = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_V = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_O = np.random.randn(d_model, d_model) / np.sqrt(d_model)

        # FFN
        self.W1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros(d_model)

        # Layer norms
        self.ln1_g, self.ln1_b = np.ones(d_model), np.zeros(d_model)
        self.ln2_g, self.ln2_b = np.ones(d_model), np.zeros(d_model)

    def attention(self, X, causal_mask):
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
        scores = np.where(causal_mask, scores, -np.inf)
        weights = softmax(scores, axis=-1)

        # Apply attention
        context = np.einsum('hij,hjd->hid', weights, V)
        context = context.transpose(1, 0, 2).reshape(seq_len, self.d_model)

        return context @ self.W_O

    def ffn(self, X):
        return gelu(X @ self.W1 + self.b1) @ self.W2 + self.b2

    def forward(self, X, causal_mask):
        # Pre-norm: normalize before sublayer
        X = X + self.attention(layer_norm(X, self.ln1_g, self.ln1_b), causal_mask)
        X = X + self.ffn(layer_norm(X, self.ln2_g, self.ln2_b))
        return X


class GPT:
    """Complete GPT model."""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token and position embeddings
        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_emb = np.random.randn(max_len, d_model) * 0.02

        # Transformer blocks
        self.blocks = [GPTBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]

        # Final layer norm and output projection
        self.ln_f_g, self.ln_f_b = np.ones(d_model), np.zeros(d_model)
        self.lm_head = self.token_emb.T  # Weight tying

    def forward(self, token_ids):
        """
        Args:
            token_ids: (seq_len,) integer token IDs

        Returns:
            logits: (seq_len, vocab_size)
        """
        seq_len = len(token_ids)

        # Embeddings
        X = self.token_emb[token_ids] + self.pos_emb[:seq_len]

        # Causal mask
        causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))

        # Transformer blocks
        for block in self.blocks:
            X = block.forward(X, causal_mask)

        # Final norm and projection
        X = layer_norm(X, self.ln_f_g, self.ln_f_b)
        logits = X @ self.lm_head

        return logits
```

## Scaling Laws

GPT's power comes from scale. Empirically:

```math
L(N, D) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}
```

where $L$ is loss, $N$ is parameters, $D$ is data, and $\alpha \approx 0.076$.

### What This Means

- **10× more parameters** → predictable loss decrease
- **10× more data** → predictable loss decrease
- **Neither saturates** (at current scales)

| Model | Parameters | Training Tokens |
|-------|------------|-----------------|
| GPT-2 | 1.5B | 40B |
| GPT-3 | 175B | 300B |
| GPT-4 | ~1T (estimated) | ~10T (estimated) |

### Compute-Optimal Scaling

Chinchilla (2022) showed: **parameters and data should scale equally**.

For a given compute budget $C$:
- Too few parameters, too much data: Underfitting
- Too many parameters, too little data: Overfitting

Optimal: $N \propto D \propto \sqrt{C}$

## Generation

### Sampling Strategies

Once trained, GPT generates by repeatedly:
1. Get probability distribution over next token
2. Sample from it
3. Append sampled token
4. Repeat

```python
def generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    """
    Generate text autoregressively.

    Args:
        model: GPT model
        prompt_ids: Initial token IDs
        max_new_tokens: How many tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Only sample from top k tokens
        top_p: Nucleus sampling threshold
    """
    token_ids = list(prompt_ids)

    for _ in range(max_new_tokens):
        # Get logits for last position
        logits = model.forward(np.array(token_ids))[-1]

        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k is not None:
            indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
            logits[indices_to_remove] = -np.inf

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_indices = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            cumulative_probs = np.cumsum(softmax(sorted_logits))

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -np.inf

        # Sample
        probs = softmax(logits)
        next_token = np.random.choice(len(probs), p=probs)
        token_ids.append(next_token)

    return token_ids
```

### Temperature

Controls randomness:

| Temperature | Effect |
|-------------|--------|
| 0.0 | Greedy (always pick highest prob) |
| 0.7 | Focused but varied |
| 1.0 | Standard sampling |
| 1.5+ | Creative but potentially incoherent |

```math
P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
```

As $T \to 0$, distribution becomes one-hot (argmax).
As $T \to \infty$, distribution becomes uniform.

### Top-k Sampling

Only consider the $k$ most probable tokens:

```python
def top_k_sample(logits, k=50):
    """Sample from top k tokens only."""
    top_k_indices = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_indices]
    probs = softmax(top_k_logits)
    choice = np.random.choice(len(probs), p=probs)
    return top_k_indices[choice]
```

### Top-p (Nucleus) Sampling

Include tokens until cumulative probability reaches $p$:

```python
def top_p_sample(logits, p=0.9):
    """Nucleus sampling: include tokens until cumulative prob >= p."""
    sorted_indices = np.argsort(logits)[::-1]
    sorted_probs = softmax(logits[sorted_indices])
    cumsum = np.cumsum(sorted_probs)

    # Find cutoff
    cutoff_idx = np.searchsorted(cumsum, p) + 1

    # Sample from nucleus
    nucleus_indices = sorted_indices[:cutoff_idx]
    nucleus_probs = sorted_probs[:cutoff_idx]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    choice = np.random.choice(len(nucleus_probs), p=nucleus_probs)
    return nucleus_indices[choice]
```

**What this means:** Top-p adapts to the distribution. If the model is confident (one token has 95% probability), nucleus is small. If uncertain, nucleus is large.

## KV Cache

During generation, we recompute attention for all previous tokens every step. The **KV cache** avoids this:

```python
class GPTWithKVCache:
    """GPT with KV caching for efficient generation."""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len):
        # ... same initialization ...
        self.kv_cache = None

    def forward_with_cache(self, token_ids, use_cache=True):
        """
        Forward pass with KV caching.

        First call: process all tokens, populate cache
        Subsequent calls: only process new token, use cached K,V
        """
        if not use_cache or self.kv_cache is None:
            # Full forward pass, populate cache
            logits = self.forward(token_ids)
            # Cache K,V for each layer (not shown: store in forward)
            return logits

        # Incremental: only process last token
        new_token = token_ids[-1:]
        X = self.token_emb[new_token] + self.pos_emb[len(token_ids)-1:len(token_ids)]

        for i, block in enumerate(self.blocks):
            # Compute Q only for new token
            Q = X @ block.W_Q

            # Use cached K,V plus new K,V
            new_K = X @ block.W_K
            new_V = X @ block.W_V

            cached_K, cached_V = self.kv_cache[i]
            K = np.vstack([cached_K, new_K])
            V = np.vstack([cached_V, new_V])

            # Update cache
            self.kv_cache[i] = (K, V)

            # Attention (Q is 1 token, K,V are all tokens)
            scores = Q @ K.T / np.sqrt(block.d_k)
            weights = softmax(scores)
            attn_out = weights @ V

            # ... rest of block ...

        return logits[-1:]  # Only need last position
```

**Complexity reduction:**
- Without cache: $O(n^2)$ per new token → $O(n^3)$ total for $n$ tokens
- With cache: $O(n)$ per new token → $O(n^2)$ total

## Emergent Capabilities

As GPT scales, new capabilities emerge:

| Scale | Capability |
|-------|------------|
| ~100M | Basic grammar, simple patterns |
| ~1B | Some factual knowledge, basic reasoning |
| ~10B | In-context learning, few-shot tasks |
| ~100B | Complex reasoning, code generation |
| ~1T | Strong generalization, instruction following |

**In-context learning:** GPT can learn new tasks from examples in the prompt, without any gradient updates:

```
Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
cheese => ???
```

The model learns the pattern from examples and applies it.

## Training at Scale

### Data

Modern GPT training uses:
- Web crawls (Common Crawl, filtered)
- Books (BookCorpus, Books3)
- Wikipedia
- Code (GitHub)
- Scientific papers
- Curated high-quality sources

Data quality matters enormously. Filtering, deduplication, and curation are critical.

### Distributed Training

Training GPT-3 scale requires:

| Technique | Purpose |
|-----------|---------|
| Data parallelism | Split batches across devices |
| Tensor parallelism | Split individual operations |
| Pipeline parallelism | Split layers across devices |
| ZeRO | Optimizer state sharding |

### Mixed Precision

Train in FP16/BF16 with FP32 master weights:

```python
# Pseudocode
def mixed_precision_step():
    # Forward in FP16
    with autocast(dtype=float16):
        logits = model(inputs)
        loss = cross_entropy(logits, targets)

    # Scale loss to prevent underflow
    scaled_loss = loss * scale_factor

    # Backward in FP16
    scaled_loss.backward()

    # Update in FP32
    for param in model.parameters():
        param.fp32 -= lr * param.grad.fp32
```

## From GPT to ChatGPT

Base GPT is a text predictor. To make it useful:

### 1. Instruction Tuning

Fine-tune on instruction-following examples:

```
Instruction: Summarize the following article...
Article: [long text]
Summary: [short summary]
```

### 2. RLHF (Reinforcement Learning from Human Feedback)

1. Collect human comparisons of model outputs
2. Train a reward model to predict human preferences
3. Optimize GPT to maximize reward model scores

```
Prompt: "Explain quantum computing"
Response A: [technical jargon]
Response B: [clear explanation]
Human prefers: B

Reward model learns: clarity > jargon
GPT learns: produce clearer explanations
```

### 3. Constitutional AI

Train the model to follow rules:
- Be helpful, harmless, honest
- Refuse harmful requests
- Acknowledge uncertainty

## Practical Considerations

### Tokenization

GPT uses subword tokenization (BPE or SentencePiece):

```python
# Conceptual tokenization
"Hello world!" → ["Hello", " world", "!"]
"antidisestablishmentarianism" → ["anti", "dis", "establish", "ment", "arian", "ism"]
```

Vocabulary size is typically 32K-100K tokens.

### Context Length

| Model | Context Length |
|-------|---------------|
| GPT-2 | 1024 tokens |
| GPT-3 | 2048 tokens |
| GPT-4 | 8K-128K tokens |
| Claude 3 | 200K tokens |

Longer context = more memory and compute (quadratic in self-attention).

### Inference Optimization

| Technique | Speedup |
|-----------|---------|
| KV Cache | 10-100× |
| Flash Attention | 2-4× |
| Quantization (INT8) | 2-4× |
| Speculative decoding | 2-3× |
| Batching | Linear in batch size |

## Summary

| Concept | Key Idea |
|---------|----------|
| Autoregressive | Predict next token, one at a time |
| Causal mask | Can't see future during training |
| Scaling laws | More compute → predictable improvement |
| Temperature | Control randomness in sampling |
| KV cache | Avoid recomputing past tokens |
| Emergent | Capabilities appear with scale |

**The essential insight:** GPT's simplicity is deceptive. The training objective—predict the next token—seems almost trivially simple. But to predict text well, the model must learn grammar, facts, reasoning, style, and much more. Scale this up, and you get a general-purpose text generator that can be adapted to countless tasks through prompting or fine-tuning.

The recipe is: (1) transformer architecture, (2) autoregressive training, (3) massive data, (4) massive compute. The result is a model that, while "just" predicting text, exhibits remarkable capabilities that continue to surprise researchers.

**Next:** [Pretraining](../modern-llms/pretraining.md) covers how models like GPT are trained at scale on internet text.

**Notebook:** [10-gpt-from-scratch.ipynb](../notebooks/10-gpt-from-scratch.ipynb) trains a character-level GPT on Shakespeare.
