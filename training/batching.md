# Batching

$$
\boxed{\nabla_\theta L \approx \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta L_i}
$$

**Batching** processes multiple examples simultaneously. Instead of updating weights after each example (slow, noisy) or after the entire dataset (expensive, infrequent), mini-batch gradient descent hits a sweet spot: efficient GPU utilization, reasonable gradient estimates, and practical memory use.

Prerequisites: [optimizers](optimizers.md), [backpropagation](../neural-networks/backpropagation.md). Code: `numpy`.

---

## Why Batch?

### Single Example (SGD)

```python
def stochastic_gradient_descent(model, data, lr=0.01):
    """Update after each example. Very noisy."""
    for x, y in data:
        loss = model.forward(x)
        grads = model.backward(y)
        for param, grad in zip(model.params, grads):
            param -= lr * grad
```

**Problems:**
- Extremely noisy gradients (single example)
- Can't utilize GPU parallelism
- Slow (Python loop per example)

### Full Batch

```python
def batch_gradient_descent(model, data, lr=0.01):
    """Update after entire dataset. Expensive."""
    # Compute gradient over ALL data
    total_grad = [np.zeros_like(p) for p in model.params]

    for x, y in data:
        loss = model.forward(x)
        grads = model.backward(y)
        for i, grad in enumerate(grads):
            total_grad[i] += grad

    # Single update
    for param, grad in zip(model.params, total_grad):
        param -= lr * (grad / len(data))
```

**Problems:**
- Requires entire dataset in memory
- Updates too infrequent
- All examples equally weighted (no curriculum)

### Mini-Batch (The Sweet Spot)

```python
def mini_batch_gradient_descent(model, data, batch_size=32, lr=0.01):
    """Update after each mini-batch. Best of both worlds."""
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]

        # Forward and backward on batch
        batch_x = np.stack([x for x, y in batch])
        batch_y = np.stack([y for x, y in batch])

        loss = model.forward(batch_x)  # Batched forward
        grads = model.backward(batch_y)  # Batched backward

        for param, grad in zip(model.params, grads):
            param -= lr * grad
```

**What this means:** Mini-batches give gradient estimates that are good enough (variance decreases as $1/\sqrt{B}$) while fitting in memory and utilizing GPU parallelism.

## Batch Size Effects

### On Gradient Variance

```python
import numpy as np

def gradient_variance_analysis(true_gradient, per_example_grads, batch_sizes):
    """
    Show how batch size affects gradient estimate variance.
    """
    for batch_size in batch_sizes:
        variances = []

        for _ in range(1000):
            # Random batch
            indices = np.random.choice(len(per_example_grads), batch_size, replace=False)
            batch_grad = np.mean(per_example_grads[indices], axis=0)

            # Squared error from true gradient
            variance = np.sum((batch_grad - true_gradient) ** 2)
            variances.append(variance)

        print(f"Batch size {batch_size}: variance = {np.mean(variances):.4f}")

# Variance ∝ 1/batch_size
# Doubling batch size halves variance
```

### On Training Dynamics

| Batch Size | Gradient Quality | Updates/Epoch | Memory |
|------------|------------------|---------------|--------|
| 1 | Very noisy | N | Minimal |
| 32 | Moderate noise | N/32 | Low |
| 256 | Low noise | N/256 | Medium |
| 4096+ | Very stable | N/4096 | High |

### The Learning Rate Connection

**Linear scaling rule:** When batch size increases, scale learning rate proportionally:

```python
def scaled_learning_rate(base_lr, base_batch_size, new_batch_size):
    """
    Scale learning rate with batch size.

    Intuition: larger batches mean fewer updates,
    so each update should be larger.
    """
    return base_lr * (new_batch_size / base_batch_size)

# Example
base_lr = 0.001
base_batch = 32
new_batch = 256

new_lr = scaled_learning_rate(base_lr, base_batch, new_batch)
print(f"Scaled LR: {new_lr}")  # 0.008
```

**Warmup for large batches:**

```python
def learning_rate_with_warmup(step, base_lr, warmup_steps, total_steps):
    """
    Gradual warmup prevents instability with large batches.
    """
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step / warmup_steps)
    else:
        # Cosine decay after warmup
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
```

## Gradient Accumulation

### When Batch Doesn't Fit

Simulate large batches on limited memory:

```python
def train_with_gradient_accumulation(model, data, effective_batch_size=256,
                                      micro_batch_size=32, lr=0.01):
    """
    Accumulate gradients over multiple micro-batches.

    effective_batch_size = micro_batch_size × accumulation_steps
    """
    accumulation_steps = effective_batch_size // micro_batch_size
    optimizer = AdamW(model.params, lr=lr)

    accumulated_grads = [np.zeros_like(p) for p in model.params]
    step = 0

    for i in range(0, len(data), micro_batch_size):
        micro_batch = data[i:i+micro_batch_size]

        # Forward and backward
        loss = model.forward(micro_batch)
        grads = model.backward()

        # Accumulate (don't update yet)
        for j, grad in enumerate(grads):
            accumulated_grads[j] += grad / accumulation_steps

        step += 1

        # Update after accumulation_steps micro-batches
        if step % accumulation_steps == 0:
            optimizer.step(accumulated_grads)

            # Reset accumulators
            accumulated_grads = [np.zeros_like(p) for p in model.params]
```

### Memory vs Effective Batch Size

```
GPU Memory: 16GB
Model: 7B parameters

Full batch=256: Needs ~40GB (doesn't fit)

Gradient accumulation:
- micro_batch=32: Fits in 16GB
- accumulation_steps=8
- effective_batch=256 (same gradient quality)
```

**What this means:** Gradient accumulation decouples effective batch size from GPU memory. You can train with arbitrarily large effective batches on limited hardware.

## Dynamic Batching for Sequences

### The Problem

Sequences have different lengths:

```
Sentence 1: "Hello" (1 token)
Sentence 2: "The quick brown fox" (4 tokens)
Sentence 3: "Machine learning is fascinating" (5 tokens)
```

Naive batching wastes compute on padding:

```
Batch (padded to length 5):
["Hello", <pad>, <pad>, <pad>, <pad>]  ← 80% padding!
["The", "quick", "brown", "fox", <pad>]
["Machine", "learning", "is", "fascinating", "."]
```

### Padding and Masking

```python
def create_padded_batch(sequences, pad_token=0):
    """
    Pad sequences to same length with attention masking.
    """
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    padded = np.full((batch_size, max_len), pad_token)
    attention_mask = np.zeros((batch_size, max_len))

    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
        attention_mask[i, :len(seq)] = 1

    return padded, attention_mask


def masked_loss(logits, targets, attention_mask):
    """Compute loss only on non-padded tokens."""
    # Cross-entropy for each position
    log_probs = log_softmax(logits)
    token_losses = -log_probs[np.arange(logits.shape[0])[:, None],
                              np.arange(logits.shape[1]),
                              targets]

    # Mask out padding
    masked_losses = token_losses * attention_mask
    return masked_losses.sum() / attention_mask.sum()
```

### Length-Based Bucketing

Group similar-length sequences together:

```python
def bucket_by_length(sequences, bucket_boundaries=[32, 64, 128, 256, 512]):
    """
    Group sequences into buckets by length.

    Sequences in same bucket have similar lengths,
    reducing padding waste.
    """
    buckets = {b: [] for b in bucket_boundaries}
    buckets[float('inf')] = []  # Overflow bucket

    for seq in sequences:
        seq_len = len(seq)
        for boundary in bucket_boundaries:
            if seq_len <= boundary:
                buckets[boundary].append(seq)
                break
        else:
            buckets[float('inf')].append(seq)

    return buckets


def create_bucketed_batches(sequences, batch_size=32):
    """Create batches from length-bucketed sequences."""
    buckets = bucket_by_length(sequences)
    batches = []

    for boundary, seqs in buckets.items():
        np.random.shuffle(seqs)
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i+batch_size]
            if len(batch_seqs) > 0:
                batches.append(create_padded_batch(batch_seqs))

    np.random.shuffle(batches)  # Shuffle batch order
    return batches
```

### Packing (No Padding)

Concatenate sequences to fill context window:

```python
def pack_sequences(sequences, max_length=2048, sep_token=-100):
    """
    Pack multiple sequences into single examples.

    Instead of padding, concatenate sequences until max_length.
    """
    packed = []
    current_pack = []
    current_length = 0

    for seq in sequences:
        seq_len = len(seq)

        if current_length + seq_len + 1 <= max_length:
            # Add to current pack
            if current_pack:
                current_pack.append(sep_token)  # Separator
                current_length += 1
            current_pack.extend(seq)
            current_length += seq_len
        else:
            # Save current pack, start new
            if current_pack:
                # Pad to max_length
                current_pack.extend([0] * (max_length - len(current_pack)))
                packed.append(current_pack)

            current_pack = list(seq)
            current_length = seq_len

    # Don't forget last pack
    if current_pack:
        current_pack.extend([0] * (max_length - len(current_pack)))
        packed.append(current_pack)

    return np.array(packed)
```

**What this means:** Packing achieves near-100% GPU utilization by eliminating padding. Most modern LLM training uses packing for efficiency.

## Practical Considerations

### Batch Size Selection

```python
def find_max_batch_size(model, data_sample, start_batch=8):
    """
    Binary search for maximum batch size that fits in memory.
    """
    batch_size = start_batch

    while True:
        try:
            batch = data_sample[:batch_size]
            model.forward(batch)
            model.backward()

            batch_size *= 2
            print(f"Batch size {batch_size // 2} works, trying {batch_size}")

        except MemoryError:
            max_batch = batch_size // 2
            print(f"Max batch size: {max_batch}")
            return max_batch
```

### Shuffling

```python
def training_loop_with_shuffle(model, data, epochs=10, batch_size=32):
    """
    Shuffle data each epoch for better generalization.
    """
    for epoch in range(epochs):
        # Shuffle at epoch start
        indices = np.random.permutation(len(data))
        shuffled_data = [data[i] for i in indices]

        # Train on shuffled data
        for i in range(0, len(shuffled_data), batch_size):
            batch = shuffled_data[i:i+batch_size]
            train_step(model, batch)
```

### Drop Last Incomplete Batch

```python
def create_batches(data, batch_size, drop_last=True):
    """
    Optionally drop last incomplete batch.

    Useful for:
    - Consistent batch statistics (batch norm)
    - Avoiding small batch edge cases
    """
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        if drop_last and len(batch) < batch_size:
            continue
        batches.append(batch)
    return batches
```

## Batch Normalization Interaction

### Batch Size Affects Batch Norm

```python
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.eps = eps
        self.momentum = momentum

    def forward(self, x, training=True):
        """
        Batch norm statistics depend on batch size.

        Small batches → noisy statistics → unstable training
        """
        if training:
            # Compute batch statistics
            mean = x.mean(axis=0)
            var = x.var(axis=0)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

**Note:** For very small batch sizes, consider Layer Normalization or Group Normalization instead.

## Summary

| Concept | Description |
|---------|-------------|
| Mini-batch | Process B examples together |
| Gradient accumulation | Simulate large batches on limited memory |
| Bucketing | Group similar-length sequences |
| Packing | Concatenate sequences to avoid padding |
| Batch size scaling | Larger batch → scale learning rate |

**The essential insight:** Batching is about balancing gradient quality, memory usage, and hardware utilization. Mini-batches give good enough gradient estimates while maximizing GPU parallelism. Gradient accumulation lets you decouple effective batch size from memory constraints. For sequences, bucketing and packing minimize wasted computation on padding.

**Historical context:** The shift from SGD to mini-batch SGD enabled practical deep learning. Modern large-scale training uses massive effective batch sizes (32k+) with gradient accumulation and careful learning rate scaling. Packing became essential for LLM training efficiency.

**Next:** [Distributed Training](distributed-training.md) covers scaling beyond single-GPU with data, model, and pipeline parallelism.
