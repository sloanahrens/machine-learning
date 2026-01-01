# Inference Optimization

$$
\boxed{\text{Tokens/sec} = \frac{\text{Batch size} \times \text{Sequence length}}{\text{Latency}} \propto \frac{\text{Memory bandwidth}}{\text{Model size}}}
$$

**Inference optimization** makes LLMs fast and cheap to deploy. A 70B model generating one token at a time is slow and expensive. With KV caching, batching, quantization, and speculative decoding, the same model can serve thousands of requests efficiently.

Prerequisites: [attention](../transformers/attention.md), [GPT](../transformers/gpt.md). Code: `numpy`.

---

## The Inference Challenge

### Autoregressive Generation

LLMs generate one token at a time, each requiring a full forward pass:

```python
import numpy as np

def naive_generate(model, prompt_tokens, max_new_tokens):
    """
    Naive autoregressive generation.

    Problem: Each token requires full attention over ALL previous tokens.
    """
    tokens = list(prompt_tokens)

    for _ in range(max_new_tokens):
        # Full forward pass over entire sequence
        logits = model.forward(tokens)

        # Sample next token
        next_token = sample(logits[-1])
        tokens.append(next_token)

        if next_token == model.eos_token:
            break

    return tokens
```

### Why It's Slow

For sequence length $n$ and model dimension $d$:

| Operation | Time Complexity |
|-----------|-----------------|
| Each attention layer | $O(n^2 \cdot d)$ |
| Generating $n$ tokens | $O(n^3 \cdot d)$ |

With 32 layers and $n = 2048$, that's a lot of redundant computation.

**The key insight:** When generating token $t$, we've already computed attention for tokens $1$ through $t-1$. We're repeating work.

## KV Caching

### The Solution

Cache key and value projections from previous tokens:

```python
class KVCache:
    def __init__(self, num_layers, max_seq_len, num_heads, head_dim):
        """
        Cache for keys and values across layers.
        """
        self.num_layers = num_layers
        self.cache_k = [None] * num_layers
        self.cache_v = [None] * num_layers

    def get(self, layer_idx):
        """Get cached K, V for a layer."""
        return self.cache_k[layer_idx], self.cache_v[layer_idx]

    def update(self, layer_idx, new_k, new_v):
        """Append new K, V to cache."""
        if self.cache_k[layer_idx] is None:
            self.cache_k[layer_idx] = new_k
            self.cache_v[layer_idx] = new_v
        else:
            self.cache_k[layer_idx] = np.concatenate(
                [self.cache_k[layer_idx], new_k], axis=1
            )
            self.cache_v[layer_idx] = np.concatenate(
                [self.cache_v[layer_idx], new_v], axis=1
            )


def attention_with_kv_cache(x, W_q, W_k, W_v, W_o, cache, layer_idx):
    """
    Attention using KV cache.

    During generation:
    - x is just the NEW token embedding [batch, 1, d_model]
    - K, V come from cache + new token
    - Q is just for the new token
    """
    batch_size, seq_len, d_model = x.shape  # seq_len = 1 during generation

    # Compute Q, K, V for new tokens only
    Q_new = x @ W_q.T  # [batch, 1, d_model]
    K_new = x @ W_k.T
    V_new = x @ W_v.T

    # Update cache
    cache.update(layer_idx, K_new, V_new)

    # Get full K, V from cache
    K_full, V_full = cache.get(layer_idx)  # [batch, seq_so_far, d_model]

    # Attention: Q_new attends to all K, V
    scores = Q_new @ K_full.transpose(0, 2, 1) / np.sqrt(d_model)
    weights = softmax(scores, axis=-1)
    output = weights @ V_full

    return output @ W_o.T


def generate_with_kv_cache(model, prompt_tokens, max_new_tokens):
    """
    Efficient generation with KV caching.
    """
    cache = KVCache(model.num_layers, max_seq_len=4096,
                    num_heads=model.num_heads, head_dim=model.head_dim)

    # Prefill: process entire prompt
    x = model.embed(prompt_tokens)
    for layer_idx, layer in enumerate(model.layers):
        x = layer.forward_with_cache(x, cache, layer_idx)
    logits = model.lm_head(x)

    tokens = list(prompt_tokens)
    next_token = sample(logits[:, -1])
    tokens.append(next_token)

    # Decode: one token at a time
    for _ in range(max_new_tokens - 1):
        # Only process the NEW token
        x = model.embed([next_token])  # [batch, 1, d_model]

        for layer_idx, layer in enumerate(model.layers):
            x = layer.forward_with_cache(x, cache, layer_idx)

        logits = model.lm_head(x)
        next_token = sample(logits[:, -1])
        tokens.append(next_token)

        if next_token == model.eos_token:
            break

    return tokens
```

### Complexity Improvement

| Phase | Without Cache | With Cache |
|-------|---------------|------------|
| Prefill ($n$ tokens) | $O(n^2 \cdot d)$ | $O(n^2 \cdot d)$ (same) |
| Each decode step | $O(n^2 \cdot d)$ | $O(n \cdot d)$ |
| Generate $m$ new tokens | $O(m \cdot n^2 \cdot d)$ | $O(m \cdot n \cdot d)$ |

**What this means:** KV caching eliminates quadratic growth in decode time. For long sequences, this is orders of magnitude faster.

### Memory Trade-off

```python
def kv_cache_memory(batch_size, seq_len, num_layers, d_model, dtype_bytes=2):
    """
    Calculate KV cache memory.

    For each layer: 2 * seq_len * d_model (K and V)
    """
    per_layer = 2 * seq_len * d_model * dtype_bytes
    total = batch_size * num_layers * per_layer
    return total / 1e9  # GB

# 70B model, batch=1, seq=4096
print(f"KV cache: {kv_cache_memory(1, 4096, 80, 8192):.1f} GB")  # ~5 GB
```

## Batching Strategies

### Continuous Batching

Requests have different lengths; don't wait for all to finish:

```python
class ContinuousBatcher:
    def __init__(self, model, max_batch_size=32):
        """
        Continuous batching: add/remove requests dynamically.
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.active_requests = []
        self.pending_requests = []

    def step(self):
        """Process one generation step for all active requests."""
        if not self.active_requests:
            return []

        # Prepare batch
        inputs = []
        caches = []
        for req in self.active_requests:
            inputs.append(req['last_token'])
            caches.append(req['kv_cache'])

        # Batched forward pass
        outputs = self.model.batched_forward(inputs, caches)

        # Process outputs
        completed = []
        still_active = []

        for req, output in zip(self.active_requests, outputs):
            next_token = sample(output)
            req['tokens'].append(next_token)
            req['last_token'] = next_token

            if next_token == self.model.eos_token or len(req['tokens']) >= req['max_len']:
                completed.append(req)
            else:
                still_active.append(req)

        self.active_requests = still_active

        # Fill empty slots with pending requests
        while len(self.active_requests) < self.max_batch_size and self.pending_requests:
            new_req = self.pending_requests.pop(0)
            self._prefill(new_req)
            self.active_requests.append(new_req)

        return completed
```

### PagedAttention (vLLM)

Manage KV cache like virtual memory:

```python
class PagedKVCache:
    def __init__(self, num_blocks, block_size, num_layers, head_dim):
        """
        PagedAttention: manage KV cache in fixed-size blocks.

        Instead of pre-allocating max_seq_len per request,
        allocate blocks on demand.
        """
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Physical blocks: [num_blocks, 2, block_size, num_heads, head_dim]
        self.blocks = np.zeros((num_blocks, 2, block_size, num_layers, head_dim))

        # Free block list
        self.free_blocks = list(range(num_blocks))

        # Per-request block tables
        self.block_tables = {}  # request_id -> list of block indices

    def allocate(self, request_id):
        """Allocate first block for new request."""
        if not self.free_blocks:
            raise MemoryError("No free KV cache blocks")
        block_idx = self.free_blocks.pop()
        self.block_tables[request_id] = [block_idx]

    def append(self, request_id, k, v, position):
        """Append KV to request's cache."""
        block_idx = position // self.block_size
        block_offset = position % self.block_size

        # Allocate new block if needed
        while block_idx >= len(self.block_tables[request_id]):
            if not self.free_blocks:
                raise MemoryError("No free KV cache blocks")
            new_block = self.free_blocks.pop()
            self.block_tables[request_id].append(new_block)

        physical_block = self.block_tables[request_id][block_idx]
        self.blocks[physical_block, 0, block_offset] = k  # Key
        self.blocks[physical_block, 1, block_offset] = v  # Value

    def free(self, request_id):
        """Free all blocks for completed request."""
        blocks = self.block_tables.pop(request_id)
        self.free_blocks.extend(blocks)
```

**What this means:** PagedAttention eliminates memory fragmentation. Requests only use the memory they need, enabling higher throughput.

## Quantization

### Why Quantize?

```
Model size (fp16): 2 bytes per parameter
70B model: 140 GB just for weights

Model size (int8): 1 byte per parameter
70B model: 70 GB

Model size (int4): 0.5 bytes per parameter
70B model: 35 GB
```

### Weight Quantization

```python
def quantize_to_int8(weights):
    """
    Symmetric int8 quantization.

    Maps [-max, max] to [-127, 127]
    """
    max_val = np.max(np.abs(weights))
    scale = max_val / 127

    quantized = np.round(weights / scale).astype(np.int8)
    return quantized, scale


def dequantize_int8(quantized, scale):
    """Dequantize int8 back to float."""
    return quantized.astype(np.float32) * scale


def quantize_to_int4(weights, group_size=128):
    """
    Group-wise int4 quantization.

    Quantize in groups for better accuracy.
    """
    # Reshape into groups
    original_shape = weights.shape
    weights_flat = weights.flatten()
    num_groups = len(weights_flat) // group_size
    grouped = weights_flat[:num_groups * group_size].reshape(num_groups, group_size)

    # Quantize each group
    scales = np.max(np.abs(grouped), axis=1) / 7  # int4: -8 to 7
    quantized = np.round(grouped / scales[:, np.newaxis]).astype(np.int8)

    # Pack two int4 values into one int8
    # (Implementation detail omitted for clarity)

    return quantized, scales, original_shape
```

### Activation Quantization (W8A8)

Quantize both weights AND activations:

```python
def quantized_matmul(x, W_quantized, W_scale, x_scale=None):
    """
    Int8 matrix multiplication.

    If x is also quantized, this runs on int8 tensor cores.
    """
    if x_scale is None:
        # Quantize activation on the fly
        x_quantized, x_scale = quantize_to_int8(x)
    else:
        x_quantized = x

    # Int8 matmul (much faster on GPU)
    result_int32 = x_quantized.astype(np.int32) @ W_quantized.astype(np.int32)

    # Dequantize result
    result = result_int32.astype(np.float32) * (x_scale * W_scale)

    return result
```

### GPTQ and AWQ

Advanced quantization methods that minimize accuracy loss:

```python
def gptq_quantize_layer(W, H, bits=4):
    """
    GPTQ: Optimal Brain Quantization for Transformers.

    Uses Hessian information to quantize with minimal error.

    Args:
        W: Weight matrix
        H: Hessian approximation (X.T @ X from calibration data)
        bits: Target bit width
    """
    # Simplified GPTQ algorithm
    d_out, d_in = W.shape
    W_quantized = np.zeros_like(W)
    scales = np.zeros(d_in // 128)  # Group-wise scales

    # Process columns in order of Hessian diagonal (importance)
    order = np.argsort(np.diag(H))[::-1]

    for i in order:
        # Quantize column i
        col = W[:, i]
        scale = np.max(np.abs(col)) / (2 ** (bits - 1) - 1)
        q_col = np.round(col / scale) * scale
        W_quantized[:, i] = q_col

        # Update remaining columns to compensate for quantization error
        error = col - q_col
        # (Hessian-based error compensation)

    return W_quantized, scales
```

### Quantization Comparison

| Method | Bits | Accuracy | Speed | Memory |
|--------|------|----------|-------|--------|
| FP16 | 16 | Baseline | 1x | 1x |
| INT8 | 8 | ~99% | 1.5-2x | 0.5x |
| GPTQ 4-bit | 4 | ~97% | 2-3x | 0.25x |
| AWQ 4-bit | 4 | ~98% | 2-3x | 0.25x |

**What this means:** 4-bit quantization makes 70B models run on consumer GPUs with minimal quality loss.

## Speculative Decoding

### The Idea

Use a small "draft" model to propose tokens, verify with large model in parallel:

```python
def speculative_decode(large_model, small_model, prompt, gamma=4):
    """
    Speculative decoding: draft with small model, verify with large.

    Args:
        gamma: Number of tokens to draft before verification
    """
    tokens = list(prompt)

    while len(tokens) < max_length:
        # Draft: generate gamma tokens with small model
        draft_tokens = []
        draft_probs = []
        small_cache = init_cache(small_model)

        for _ in range(gamma):
            logits = small_model.forward(tokens + draft_tokens, small_cache)
            prob = softmax(logits[-1])
            token = sample(prob)
            draft_tokens.append(token)
            draft_probs.append(prob[token])

        # Verify: run large model on all draft tokens in parallel
        full_sequence = tokens + draft_tokens
        large_logits = large_model.forward(full_sequence)  # Batched!
        large_probs = softmax(large_logits[-gamma-1:-1], axis=-1)

        # Accept/reject each draft token
        accepted = 0
        for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
            large_prob = large_probs[i, draft_token]

            # Accept if large model agrees (or is more confident)
            if np.random.random() < large_prob / draft_prob:
                accepted += 1
            else:
                # Reject: sample from adjusted distribution
                adjusted = np.maximum(large_probs[i] - draft_probs, 0)
                adjusted /= adjusted.sum()
                new_token = sample(adjusted)
                tokens.append(new_token)
                break

        # Add accepted tokens
        tokens.extend(draft_tokens[:accepted])

        if accepted == gamma:
            # All accepted, sample one more from large model
            bonus_token = sample(softmax(large_logits[-1]))
            tokens.append(bonus_token)

    return tokens
```

### Why It Works

```
Without speculation:
Large model: [token1] [token2] [token3] [token4]
             100ms    100ms    100ms    100ms  = 400ms

With speculation (gamma=4, 75% accept rate):
Small model: [t1, t2, t3, t4] = 20ms (parallel)
Large model: [verify all 4]   = 110ms (single batch)
Average accepted: 3
Total: 130ms for 3 tokens = 43ms/token vs 100ms/token
```

### Acceptance Rate

The speedup depends on draft/target model agreement:

```python
def estimate_speedup(acceptance_rate, gamma, small_latency, large_latency):
    """
    Estimate speculative decoding speedup.

    Args:
        acceptance_rate: P(large accepts small's token)
        gamma: Draft length
        small_latency: Time for small model (all gamma tokens)
        large_latency: Time for large model (one batch)
    """
    # Expected accepted tokens per round
    expected_accepted = sum(acceptance_rate ** i for i in range(gamma))

    # Time per round
    time_per_round = small_latency + large_latency

    # Effective tokens per second
    tokens_per_round = expected_accepted + 1  # +1 for bonus token
    speedup = tokens_per_round / (time_per_round / large_latency)

    return speedup

# Example: 90% acceptance
print(f"Speedup: {estimate_speedup(0.9, 4, 20, 100):.2f}x")
```

## Flash Attention

### Memory-Efficient Attention

Standard attention materializes the full $n \times n$ attention matrix:

```python
def standard_attention(Q, K, V):
    """
    Standard attention: O(n²) memory.
    """
    # This matrix is n x n
    scores = Q @ K.T / np.sqrt(Q.shape[-1])  # [n, n]
    weights = softmax(scores, axis=-1)       # [n, n]
    output = weights @ V                      # [n, d]
    return output
```

Flash Attention computes in tiles without materializing full matrix:

```python
def flash_attention_simplified(Q, K, V, block_size=64):
    """
    Flash Attention: O(n) memory via tiling.

    Key insight: softmax can be computed incrementally.
    """
    n, d = Q.shape
    output = np.zeros((n, d))
    row_max = np.full(n, -np.inf)
    row_sum = np.zeros(n)

    # Process K, V in blocks
    for j in range(0, n, block_size):
        K_block = K[j:j+block_size]
        V_block = V[j:j+block_size]

        # Local scores
        scores = Q @ K_block.T / np.sqrt(d)  # [n, block_size]

        # Update running softmax
        block_max = np.max(scores, axis=-1)
        new_max = np.maximum(row_max, block_max)

        # Rescale previous sum
        exp_diff = np.exp(row_max - new_max)
        row_sum = row_sum * exp_diff

        # Add new block contribution
        exp_scores = np.exp(scores - new_max[:, np.newaxis])
        row_sum += exp_scores.sum(axis=-1)

        # Update output (rescale + add)
        output = output * exp_diff[:, np.newaxis]
        output += exp_scores @ V_block

        row_max = new_max

    # Normalize
    output = output / row_sum[:, np.newaxis]
    return output
```

### Benefits

| Aspect | Standard | Flash Attention |
|--------|----------|-----------------|
| Memory | $O(n^2)$ | $O(n)$ |
| IO operations | Many | Few (tiled) |
| GPU utilization | Lower | Higher |
| Speedup | Baseline | 2-4x |

**What this means:** Flash Attention enables longer sequences by reducing memory from quadratic to linear, while also being faster due to better hardware utilization.

## Putting It All Together

### Optimized Inference Pipeline

```python
class OptimizedLLMServer:
    def __init__(self, model_path, quantization='int4'):
        # Load quantized model
        self.model = load_quantized_model(model_path, quantization)

        # Initialize paged KV cache
        self.kv_cache = PagedKVCache(
            num_blocks=1000,
            block_size=16,
            num_layers=self.model.num_layers,
            head_dim=self.model.head_dim
        )

        # Continuous batcher
        self.batcher = ContinuousBatcher(self.model, max_batch_size=64)

        # Optional: draft model for speculation
        self.draft_model = load_draft_model(model_path)

    def generate(self, prompt, max_tokens=100, use_speculation=True):
        """Generate with all optimizations."""

        if use_speculation and self.draft_model:
            return speculative_decode(
                self.model,
                self.draft_model,
                prompt,
                gamma=4
            )
        else:
            return self.batcher.generate(prompt, max_tokens)
```

### Performance Comparison

For a 70B model generating 100 tokens:

| Configuration | Time | Memory |
|---------------|------|--------|
| Naive FP16 | 60s | 140 GB |
| + KV Cache | 15s | 145 GB |
| + INT4 Quantization | 15s | 40 GB |
| + Batching (8 requests) | 3s/request | 50 GB |
| + Speculative (2x speedup) | 1.5s/request | 55 GB |

## Summary

| Technique | What It Does | Benefit |
|-----------|--------------|---------|
| KV Caching | Cache keys/values from previous tokens | Linear decode instead of quadratic |
| Continuous Batching | Dynamic batch management | Higher throughput |
| PagedAttention | Virtual memory for KV cache | No fragmentation |
| Quantization | Reduce precision (INT8, INT4) | 2-4x less memory |
| Speculative Decoding | Draft/verify with small/large models | 2-3x faster generation |
| Flash Attention | Tiled attention computation | Linear memory, 2-4x faster |

**The essential insight:** LLM inference is memory-bandwidth bound, not compute bound. Optimizations that reduce memory (quantization), reuse computation (KV cache), or parallelize verification (speculation) yield massive speedups. The combination of these techniques makes serving 70B+ models practical and economical.

**Historical context:** These optimizations evolved rapidly from 2022-2024. KV caching was always used but PagedAttention (vLLM, 2023) solved fragmentation. Flash Attention (2022) enabled longer contexts. Speculative decoding (2023) and aggressive quantization (GPTQ, AWQ in 2023) made large models practical on consumer hardware.
