# Efficient Adaptation

```math
\boxed{W' = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k)}
```

**Efficient adaptation** modifies pretrained models with minimal parameter changes. Full fine-tuning updates billions of parameters; LoRA updates millions. This makes fine-tuning practical on consumer hardware and enables multiple task-specific adapters sharing one base model.

Prerequisites: [fine-tuning](fine-tuning.md), [attention](../transformers/attention.md). Code: `numpy`.

---

## The Problem: Fine-Tuning Is Expensive

### Full Fine-Tuning Costs

For a 7B parameter model:

| Resource | Full Fine-Tuning |
|----------|------------------|
| Trainable params | 7B |
| Optimizer states | 14B (Adam: m, v) |
| Gradients | 7B |
| Total memory | ~112 GB (fp16) |
| Storage per task | 14 GB |

Want 10 different fine-tuned versions? That's 140 GB of storage and 10 separate inference deployments.

### The Insight

Fine-tuning changes weights, but by how much?

```python
import numpy as np

def measure_weight_change(original, finetuned):
    """Analyze how much weights changed during fine-tuning."""
    diff = finetuned - original

    # Full rank of difference
    U, S, Vt = np.linalg.svd(diff, full_matrices=False)

    # How much variance in top-k singular values?
    total_variance = np.sum(S ** 2)
    for k in [1, 4, 8, 16, 64]:
        top_k_variance = np.sum(S[:k] ** 2)
        print(f"Top {k} singular values: {100 * top_k_variance / total_variance:.1f}% of variance")

# Typical finding: weight changes are low-rank
# Top 8-16 singular values capture 90%+ of the change
```

**What this means:** Fine-tuning doesn't need to update all parameters. The effective changes lie in a low-dimensional subspace. We can parameterize just that subspace.

## LoRA: Low-Rank Adaptation

### Core Idea

Instead of updating weight matrix $W$ directly, add a low-rank decomposition:

```math
W' = W + \Delta W = W + BA
```

Where:
- $W \in \mathbb{R}^{d \times k}$ is frozen
- $B \in \mathbb{R}^{d \times r}$ is trainable
- $A \in \mathbb{R}^{r \times k}$ is trainable
- $r \ll \min(d, k)$ is the rank (typically 4-64)

```python
class LoRALayer:
    def __init__(self, original_weight, rank=8, alpha=16):
        """
        LoRA adapter for a single weight matrix.

        Args:
            original_weight: [d_out, d_in] pretrained weights (frozen)
            rank: Low-rank dimension
            alpha: Scaling factor
        """
        self.W = original_weight  # Frozen
        d_out, d_in = original_weight.shape
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Initialize A with small random, B with zeros
        # This ensures ΔW = BA starts at zero
        self.A = np.random.randn(rank, d_in) * 0.01
        self.B = np.zeros((d_out, rank))

    def forward(self, x):
        """
        Args:
            x: [batch, seq, d_in]
        Returns:
            [batch, seq, d_out]
        """
        # Original path (frozen)
        base_output = x @ self.W.T

        # LoRA path (trainable)
        # x @ A.T @ B.T = x @ (BA).T
        lora_output = (x @ self.A.T) @ self.B.T

        return base_output + self.scaling * lora_output

    @property
    def trainable_params(self):
        return self.A.size + self.B.size

    @property
    def total_params(self):
        return self.W.size


# Compare parameter counts
d_out, d_in = 4096, 4096
rank = 8
original_params = d_out * d_in  # 16.7M
lora_params = d_out * rank + rank * d_in  # 65K (0.4%)
print(f"Original: {original_params:,}, LoRA: {lora_params:,}")
print(f"Reduction: {100 * (1 - lora_params / original_params):.1f}%")
```

### Where to Apply LoRA

Typically applied to attention projection matrices:

```python
class LoRATransformerLayer:
    def __init__(self, original_layer, rank=8, alpha=16):
        """Add LoRA to attention projections."""
        self.original = original_layer

        # LoRA on query and value projections (most common)
        self.lora_q = LoRALayer(original_layer.W_q, rank, alpha)
        self.lora_v = LoRALayer(original_layer.W_v, rank, alpha)

        # Optionally also key and output
        # self.lora_k = LoRALayer(original_layer.W_k, rank, alpha)
        # self.lora_o = LoRALayer(original_layer.W_o, rank, alpha)

    def attention(self, x):
        """Modified attention with LoRA."""
        # Use LoRA layers for Q and V
        Q = self.lora_q.forward(x)
        K = x @ self.original.W_k.T  # Original K
        V = self.lora_v.forward(x)

        # Standard attention computation
        scores = Q @ K.transpose(-2, -1) / np.sqrt(Q.shape[-1])
        weights = softmax(scores)
        output = weights @ V

        return output @ self.original.W_o.T


def count_lora_params(model, rank=8):
    """Count trainable parameters with LoRA."""
    total_trainable = 0
    total_frozen = 0

    for layer in model.layers:
        # Q, V projections get LoRA
        d = model.d_model
        total_trainable += 2 * (d * rank + rank * d)  # Q and V
        total_frozen += 4 * d * d  # Q, K, V, O original weights

        # FFN stays frozen
        total_frozen += 2 * d * 4 * d  # up and down projections

    return total_trainable, total_frozen
```

### Training with LoRA

```python
def lora_training_loop(model, lora_layers, data, epochs=3, lr=1e-4):
    """
    Train only LoRA parameters.

    Key differences from full fine-tuning:
    - Higher learning rate OK (fewer params)
    - Lower memory (no optimizer states for frozen params)
    - Faster training (fewer backward computations)
    """
    # Only optimize LoRA parameters
    lora_params = []
    for layer in lora_layers:
        lora_params.extend([layer.A, layer.B])

    optimizer = AdamW(lora_params, lr=lr)

    for epoch in range(epochs):
        for batch in data:
            # Forward through full model (frozen + LoRA)
            output = model.forward(batch['input'])
            loss = compute_loss(output, batch['target'])

            # Backward only computes gradients for LoRA params
            grads = backward(lora_params, loss)
            optimizer.step(grads)

        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

    return lora_layers


def merge_lora(original_weight, lora_layer):
    """Merge LoRA weights into original for inference."""
    return original_weight + lora_layer.scaling * (lora_layer.B @ lora_layer.A)
```

### LoRA Hyperparameters

| Parameter | Typical Values | Effect |
|-----------|----------------|--------|
| Rank (r) | 4, 8, 16, 32, 64 | Higher = more capacity, more params |
| Alpha (α) | 16, 32 | Scaling factor; α/r determines magnitude |
| Target modules | q, v (minimal) or q, k, v, o (more) | Which projections to adapt |
| Learning rate | 1e-4 to 3e-4 | Higher than full fine-tuning |

```python
def lora_config_comparison():
    """Different LoRA configurations and their parameter counts."""
    d = 4096  # Model dimension
    layers = 32  # Number of layers

    configs = {
        "rank=4, q+v": 4 * 2 * (d * 4 + 4 * d) * layers,
        "rank=8, q+v": 4 * 2 * (d * 8 + 8 * d) * layers,
        "rank=16, q+v": 4 * 2 * (d * 16 + 16 * d) * layers,
        "rank=8, q+k+v+o": 4 * 4 * (d * 8 + 8 * d) * layers,
        "full fine-tune": 4 * 4 * d * d * layers,  # Just attention
    }

    for name, params in configs.items():
        print(f"{name}: {params / 1e6:.1f}M params")
```

**What this means:** LoRA achieves 90-99% of full fine-tuning performance with <1% of trainable parameters. The rank controls the capacity/efficiency trade-off.

## Adapters

### Architecture

Insert small trainable modules between frozen layers:

```python
class Adapter:
    def __init__(self, d_model, bottleneck_dim=64):
        """
        Adapter module: down-project, nonlinearity, up-project.

        Args:
            d_model: Model hidden dimension
            bottleneck_dim: Adapter bottleneck size
        """
        self.down = np.random.randn(d_model, bottleneck_dim) * 0.01
        self.up = np.random.randn(bottleneck_dim, d_model) * 0.01
        self.bias_down = np.zeros(bottleneck_dim)
        self.bias_up = np.zeros(d_model)

    def forward(self, x):
        """
        Args:
            x: [batch, seq, d_model]
        Returns:
            [batch, seq, d_model]
        """
        # Down-project to bottleneck
        h = x @ self.down + self.bias_down

        # Nonlinearity
        h = np.maximum(0, h)  # ReLU

        # Up-project back + residual
        return x + (h @ self.up + self.bias_up)


class AdapterTransformerLayer:
    def __init__(self, original_layer, bottleneck_dim=64):
        """Add adapters to transformer layer."""
        self.original = original_layer

        # Adapter after attention
        self.adapter_attn = Adapter(original_layer.d_model, bottleneck_dim)

        # Adapter after FFN
        self.adapter_ffn = Adapter(original_layer.d_model, bottleneck_dim)

    def forward(self, x):
        # Original attention (frozen)
        attn_out = self.original.attention(x)
        attn_out = self.original.norm1(x + attn_out)

        # Adapter after attention
        attn_out = self.adapter_attn.forward(attn_out)

        # Original FFN (frozen)
        ffn_out = self.original.ffn(attn_out)
        ffn_out = self.original.norm2(attn_out + ffn_out)

        # Adapter after FFN
        ffn_out = self.adapter_ffn.forward(ffn_out)

        return ffn_out
```

### Adapter vs LoRA

| Aspect | Adapters | LoRA |
|--------|----------|------|
| Architecture | Sequential bottleneck | Parallel low-rank |
| Inference overhead | Adds latency | Can merge to zero overhead |
| Typical params | 0.5-3% | 0.1-1% |
| Where applied | After attention/FFN | Inside attention projections |

## Prefix Tuning

### Idea

Prepend learnable "soft prompts" to keys and values:

```python
class PrefixTuning:
    def __init__(self, num_layers, d_model, prefix_length=20, num_heads=32):
        """
        Prefix tuning: learn virtual tokens prepended to K and V.

        Args:
            num_layers: Number of transformer layers
            d_model: Model dimension
            prefix_length: Number of prefix tokens
            num_heads: Number of attention heads
        """
        self.prefix_length = prefix_length

        # Learnable prefix embeddings for each layer
        # Shape: [num_layers, 2, prefix_length, d_model]
        # The "2" is for keys and values
        self.prefix = np.random.randn(num_layers, 2, prefix_length, d_model) * 0.01

    def get_prefix_kv(self, layer_idx, batch_size):
        """Get prefix keys and values for a layer."""
        prefix_k = self.prefix[layer_idx, 0]  # [prefix_len, d_model]
        prefix_v = self.prefix[layer_idx, 1]

        # Expand for batch
        prefix_k = np.tile(prefix_k[np.newaxis], (batch_size, 1, 1))
        prefix_v = np.tile(prefix_v[np.newaxis], (batch_size, 1, 1))

        return prefix_k, prefix_v


class PrefixAttention:
    def __init__(self, original_attention, prefix_tuning, layer_idx):
        self.original = original_attention
        self.prefix_tuning = prefix_tuning
        self.layer_idx = layer_idx

    def forward(self, x):
        batch_size = x.shape[0]

        # Original Q, K, V
        Q = x @ self.original.W_q.T
        K = x @ self.original.W_k.T
        V = x @ self.original.W_v.T

        # Get prefix K, V
        prefix_k, prefix_v = self.prefix_tuning.get_prefix_kv(
            self.layer_idx, batch_size
        )

        # Prepend prefix to K and V
        K = np.concatenate([prefix_k, K], axis=1)
        V = np.concatenate([prefix_v, V], axis=1)

        # Attention with prefixed context
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(Q.shape[-1])
        weights = softmax(scores, axis=-1)
        output = weights @ V

        return output @ self.original.W_o.T
```

### Parameter Efficiency

```python
def prefix_tuning_params(d_model, num_layers, prefix_length):
    """Count prefix tuning parameters."""
    # 2 for K and V per layer
    return 2 * num_layers * prefix_length * d_model

# Example: 7B model
d_model = 4096
num_layers = 32
prefix_length = 20

params = prefix_tuning_params(d_model, num_layers, prefix_length)
print(f"Prefix tuning: {params / 1e6:.1f}M params")  # ~5.2M
```

## Prompt Tuning

### Simplest Approach

Learn soft prompt embeddings prepended to input:

```python
class PromptTuning:
    def __init__(self, embedding_dim, prompt_length=20):
        """
        Prompt tuning: learnable soft prompts in embedding space.

        Only modifies input, not internal representations.
        """
        self.soft_prompt = np.random.randn(prompt_length, embedding_dim) * 0.01

    def forward(self, input_embeddings):
        """
        Prepend soft prompt to input embeddings.

        Args:
            input_embeddings: [batch, seq, d_model]
        Returns:
            [batch, prompt_len + seq, d_model]
        """
        batch_size = input_embeddings.shape[0]

        # Expand prompt for batch
        prompt = np.tile(self.soft_prompt[np.newaxis], (batch_size, 1, 1))

        # Prepend to input
        return np.concatenate([prompt, input_embeddings], axis=1)
```

### Comparison

| Method | Where Applied | Inference Overhead | Params |
|--------|---------------|-------------------|--------|
| LoRA | Attention projections | Zero (can merge) | ~0.1-1% |
| Adapters | After attention/FFN | Adds computation | ~0.5-3% |
| Prefix Tuning | K, V at each layer | Minor | ~0.1% |
| Prompt Tuning | Input embeddings only | Minor | ~0.01% |

## QLoRA: Quantized LoRA

### Combine Quantization with LoRA

```python
class QLoRALayer:
    def __init__(self, quantized_weight, rank=8, alpha=16):
        """
        LoRA on top of quantized base weights.

        Base model: 4-bit quantized (frozen)
        LoRA adapters: full precision (trainable)
        """
        self.W_quantized = quantized_weight  # 4-bit, frozen
        self.rank = rank
        self.scaling = alpha / rank

        # LoRA in full precision
        d_out, d_in = quantized_weight.shape
        self.A = np.random.randn(rank, d_in).astype(np.float32) * 0.01
        self.B = np.zeros((d_out, rank), dtype=np.float32)

    def forward(self, x):
        # Dequantize for computation
        W = dequantize(self.W_quantized)

        # Base + LoRA
        base_output = x @ W.T
        lora_output = (x @ self.A.T) @ self.B.T

        return base_output + self.scaling * lora_output


def qlora_memory_estimate(model_params_billions, rank=8):
    """Estimate QLoRA memory usage."""
    # Base model: 4-bit = 0.5 bytes per param
    base_memory_gb = model_params_billions * 1e9 * 0.5 / 1e9

    # LoRA: assuming applied to attention Q, V
    # Each layer: 2 * (d * r + r * d) * 4 bytes (float32)
    d = 4096  # Approximate
    layers = int(model_params_billions * 3)  # Rough estimate
    lora_memory_gb = 2 * layers * 2 * d * rank * 4 / 1e9

    # Optimizer states for LoRA only
    optimizer_memory_gb = 2 * lora_memory_gb  # Adam m and v

    total = base_memory_gb + lora_memory_gb + optimizer_memory_gb
    print(f"Base model (4-bit): {base_memory_gb:.1f} GB")
    print(f"LoRA adapters: {lora_memory_gb:.2f} GB")
    print(f"Optimizer states: {optimizer_memory_gb:.2f} GB")
    print(f"Total: {total:.1f} GB")

    return total

# Fine-tune 70B model
qlora_memory_estimate(70)
# Base: ~35 GB, LoRA + optimizer: ~0.5 GB
# Total: ~36 GB (fits on single A100!)
```

**What this means:** QLoRA enables fine-tuning 65B+ models on a single GPU by quantizing frozen weights to 4-bit while keeping LoRA adapters in full precision.

## When to Use What

### Decision Tree

```
Need to fine-tune a large model?
├─ Have plenty of compute/memory?
│   └─ Full fine-tuning (best quality)
├─ Memory constrained?
│   ├─ Can fit fp16 model?
│   │   └─ LoRA (best quality/efficiency trade-off)
│   └─ Can't fit fp16?
│       └─ QLoRA (enables largest models)
├─ Need zero inference overhead?
│   └─ LoRA (merge adapters into weights)
├─ Need to switch tasks at runtime?
│   └─ LoRA or Adapters (swap adapters)
└─ Minimal changes, API-only access?
    └─ Prompt Tuning (soft prompts)
```

### Practical Guidelines

```python
def choose_method(model_size_b, available_memory_gb, requirements):
    """Suggest efficient adaptation method."""

    # Memory needed for full fine-tuning (fp16)
    full_ft_memory = model_size_b * 16  # params + grads + optimizer

    # Memory for LoRA (fp16 base + fp32 LoRA)
    lora_memory = model_size_b * 2 + 1  # Base + small LoRA overhead

    # Memory for QLoRA (4-bit base + fp32 LoRA)
    qlora_memory = model_size_b * 0.5 + 1

    if available_memory_gb >= full_ft_memory:
        return "Full fine-tuning"
    elif available_memory_gb >= lora_memory:
        return "LoRA"
    elif available_memory_gb >= qlora_memory:
        return "QLoRA"
    else:
        return "Need more memory or smaller model"

# Examples
print(choose_method(7, 80, {}))   # "Full fine-tuning" on A100-80GB
print(choose_method(7, 24, {}))   # "LoRA" on 3090
print(choose_method(70, 80, {}))  # "QLoRA" on A100-80GB
```

## Multi-Task with Adapters

### Adapter Switching

```python
class MultiTaskLoRA:
    def __init__(self, base_model, tasks, rank=8):
        """
        Multiple LoRA adapters sharing one base model.
        """
        self.base_model = base_model  # Frozen

        # One set of LoRA adapters per task
        self.adapters = {}
        for task in tasks:
            self.adapters[task] = create_lora_adapters(base_model, rank)

    def forward(self, x, task):
        """Forward with task-specific adapters."""
        # Apply appropriate LoRA adapters
        adapters = self.adapters[task]
        return forward_with_lora(self.base_model, adapters, x)

    def merge_for_inference(self, task):
        """Merge specific task's adapters for deployment."""
        merged_model = copy_model(self.base_model)
        for name, adapter in self.adapters[task].items():
            merged_weight = merge_lora(
                getattr(merged_model, name),
                adapter
            )
            setattr(merged_model, name, merged_weight)
        return merged_model


# Storage comparison
base_model_size = 14  # GB for 7B model
num_tasks = 10
adapter_size = 0.05  # GB per adapter set

full_ft_storage = base_model_size * num_tasks  # 140 GB
adapter_storage = base_model_size + adapter_size * num_tasks  # 14.5 GB
print(f"Full fine-tuning: {full_ft_storage} GB")
print(f"Shared base + adapters: {adapter_storage} GB")
```

## Summary

| Method | Trainable Params | Memory | Inference | Best For |
|--------|-----------------|--------|-----------|----------|
| Full fine-tuning | 100% | Very high | Native | Max quality |
| LoRA | 0.1-1% | Low | Zero overhead | General use |
| QLoRA | 0.1-1% | Very low | Minor overhead | Large models |
| Adapters | 0.5-3% | Low | Some overhead | Multi-task |
| Prefix tuning | ~0.1% | Very low | Minor overhead | Inference-sensitive |
| Prompt tuning | ~0.01% | Minimal | Minor overhead | API-only access |

**The essential insight:** Fine-tuning doesn't require updating all parameters. The task-specific adaptations live in a low-dimensional subspace that can be captured with far fewer parameters. LoRA is the current sweet spot—minimal parameters, zero inference overhead when merged, and near-full fine-tuning quality.

**Historical context:** LoRA (2021) revolutionized fine-tuning by showing that a rank-8 update captures most of what full fine-tuning achieves. QLoRA (2023) extended this to enable fine-tuning 65B models on consumer GPUs. These methods made LLM customization accessible beyond large companies.

**Next:** [Inference](inference.md) covers efficient generation with KV caching, speculative decoding, and quantization.
