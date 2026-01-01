# Distributed Training

```math
\boxed{\text{Training time} \propto \frac{\text{Dataset} \times \text{Model size}}{\text{Parallelism} \times \text{Hardware efficiency}}}
```

**Distributed training** scales beyond single GPUs. A 70B parameter model doesn't fit on one GPU. Training on trillions of tokens would take years on one machine. Distributed training combines data parallelism (split data), model parallelism (split model), and pipeline parallelism (split layers) to train massive models in practical time.

Prerequisites: [batching](batching.md), [optimizers](optimizers.md). Code: `numpy` (conceptual).

---

## Why Distributed?

### Scale Requirements

Modern LLMs require:

| Model | Parameters | Memory (fp16) | Training Compute |
|-------|------------|---------------|------------------|
| GPT-3 | 175B | 350 GB | ~3.6×10²³ FLOPS |
| LLaMA 2 70B | 70B | 140 GB | ~1.7×10²⁴ FLOPS |
| GPT-4 | ~1.8T (est.) | ~3.6 TB | ~2×10²⁵ FLOPS |

A single A100 has:
- 80 GB memory
- ~3×10¹⁵ FLOPS

**What this means:** Large models don't fit on one GPU, and training would take decades. We need to parallelize across hundreds or thousands of GPUs.

## Data Parallelism

### The Simplest Approach

Replicate model on each GPU, split data:

```python
import numpy as np

def data_parallel_training(model, data, num_gpus=4, batch_size=128):
    """
    Data parallelism: each GPU processes different data.

    1. Replicate model on all GPUs
    2. Split batch across GPUs
    3. Each GPU computes gradients on its portion
    4. All-reduce: average gradients across GPUs
    5. Each GPU updates with averaged gradient
    """
    local_batch_size = batch_size // num_gpus

    # Simulate GPUs
    models = [copy_model(model) for _ in range(num_gpus)]

    for batch in get_batches(data, batch_size):
        # Split batch across GPUs
        local_batches = [
            batch[i * local_batch_size:(i + 1) * local_batch_size]
            for i in range(num_gpus)
        ]

        # Each GPU computes gradients
        all_grads = []
        for gpu_id, (local_model, local_batch) in enumerate(zip(models, local_batches)):
            loss = local_model.forward(local_batch)
            grads = local_model.backward()
            all_grads.append(grads)

        # All-reduce: average gradients
        avg_grads = []
        for param_idx in range(len(all_grads[0])):
            grad_sum = sum(g[param_idx] for g in all_grads)
            avg_grads.append(grad_sum / num_gpus)

        # Update all replicas with same gradient
        for local_model in models:
            local_model.optimizer.step(avg_grads)
```

### All-Reduce Communication

```
GPU 0: grad_0 ──┐
GPU 1: grad_1 ──┼──→ All-Reduce ──→ avg_grad → All GPUs
GPU 2: grad_2 ──┤
GPU 3: grad_3 ──┘
```

Ring All-Reduce (efficient implementation):

```python
def ring_allreduce(grads_per_gpu, num_gpus):
    """
    Ring all-reduce: efficient gradient averaging.

    Each GPU sends to next, receives from previous.
    O(N) total data transferred (not O(N²)).
    """
    # Simplified: in practice, uses ring topology
    total = sum(grads_per_gpu)
    return [total / num_gpus] * num_gpus
```

### Scaling Efficiency

```python
def data_parallel_efficiency(num_gpus, compute_time, comm_time):
    """
    Efficiency decreases as communication overhead grows.

    Perfect scaling: N GPUs = N× speedup
    Reality: Communication overhead reduces speedup
    """
    total_time = compute_time + comm_time

    # Ideal time = single_gpu_time / num_gpus
    ideal_time = compute_time / num_gpus

    efficiency = ideal_time / total_time
    speedup = 1 / (total_time / compute_time) * num_gpus

    return efficiency, speedup

# Example: 100ms compute, 10ms communication
eff, speedup = data_parallel_efficiency(8, 100, 10)
print(f"Efficiency: {eff:.1%}, Speedup: {speedup:.1f}×")
```

**What this means:** Data parallelism scales well until communication dominates. With fast interconnects (NVLink, InfiniBand), hundreds of GPUs can be efficient.

## Model Parallelism

### When Model Doesn't Fit

A 70B model in fp16:
- Weights: 140 GB
- Optimizer states (Adam): 280 GB (m, v)
- Gradients: 140 GB
- Activations: Variable (depends on batch/seq)
- Total: ~560 GB minimum

Single A100: 80 GB

**Solution:** Split the model across GPUs.

### Tensor Parallelism

Split individual layers across GPUs:

```python
def tensor_parallel_linear(x, W_parts, num_gpus=4):
    """
    Split weight matrix column-wise across GPUs.

    W: [d_in, d_out] → W_parts: [d_in, d_out/num_gpus] per GPU
    """
    # Each GPU computes partial output
    partial_outputs = []
    for gpu_id, W_part in enumerate(W_parts):
        # GPU gpu_id computes: x @ W_part
        partial = x @ W_part  # [batch, d_out/num_gpus]
        partial_outputs.append(partial)

    # Concatenate outputs from all GPUs
    output = np.concatenate(partial_outputs, axis=-1)
    return output


def tensor_parallel_attention(x, W_q_parts, W_k_parts, W_v_parts, W_o_parts, num_gpus=4):
    """
    Attention with tensor parallelism.

    Split attention heads across GPUs.
    If 32 heads total and 4 GPUs → 8 heads per GPU.
    """
    head_outputs = []

    for gpu_id in range(num_gpus):
        # Each GPU handles its heads
        Q = x @ W_q_parts[gpu_id]  # [batch, seq, d_head * heads_per_gpu]
        K = x @ W_k_parts[gpu_id]
        V = x @ W_v_parts[gpu_id]

        # Local attention
        local_output = attention(Q, K, V)
        head_outputs.append(local_output)

    # All-reduce after output projection
    output = sum(head_outputs)  # Need proper reduction
    return output
```

### Communication Pattern

```
Tensor Parallelism requires:
- Forward: All-reduce after attention and FFN
- Backward: All-reduce for gradients

Communication is WITHIN each layer (frequent but small)
```

## Pipeline Parallelism

### Split Layers Across GPUs

```python
def pipeline_parallel_forward(x, layer_groups, num_stages=4):
    """
    Pipeline parallelism: different layers on different GPUs.

    GPU 0: layers 0-7
    GPU 1: layers 8-15
    GPU 2: layers 16-23
    GPU 3: layers 24-31
    """
    activations = x

    for stage_id, layers in enumerate(layer_groups):
        # Send to GPU stage_id (implicit)
        for layer in layers:
            activations = layer.forward(activations)
        # Send to next stage (implicit)

    return activations
```

### The Bubble Problem

Sequential dependency creates idle time:

```
Without micro-batches (inefficient):

GPU 0: [F0][F1][F2][F3]................[B0][B1][B2][B3]
GPU 1: ....[F0][F1][F2][F3]........[B0][B1][B2][B3]....
GPU 2: ........[F0][F1][F2][F3][B0][B1][B2][B3]........
GPU 3: ............[F0][F1][F2][F3][B3][B2][B1][B0]....

"Bubble" = idle time waiting for dependencies
```

### Micro-Batching (GPipe)

Split batch into micro-batches to fill the pipeline:

```python
def gpipe_forward_backward(micro_batches, layer_groups, num_stages=4):
    """
    GPipe: split batch into micro-batches to reduce bubble.

    Micro-batches proceed through pipeline,
    keeping all stages busy more of the time.
    """
    num_micro_batches = len(micro_batches)
    saved_activations = [[None] * num_stages for _ in range(num_micro_batches)]

    # Forward passes
    for mb_id, micro_batch in enumerate(micro_batches):
        x = micro_batch
        for stage_id, layers in enumerate(layer_groups):
            saved_activations[mb_id][stage_id] = x
            for layer in layers:
                x = layer.forward(x)

    # Backward passes (in reverse order)
    for mb_id in reversed(range(num_micro_batches)):
        grad = initial_grad
        for stage_id in reversed(range(num_stages)):
            x = saved_activations[mb_id][stage_id]
            grad = backward_through_stage(layer_groups[stage_id], x, grad)

    # Gradient accumulation across micro-batches
    # Then single optimizer step
```

### 1F1B Schedule

Interleave forward and backward to reduce memory:

```
1F1B (One Forward, One Backward):

GPU 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 B4 B5 B6
GPU 1:    F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 B5 B6
GPU 2:       F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 B6
GPU 3:          F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6

Less memory: only need to store 1-2 micro-batch activations per stage
```

**What this means:** Pipeline parallelism trades communication for compute efficiency. With enough micro-batches, the bubble overhead becomes small.

## ZeRO: Zero Redundancy Optimizer

### The Redundancy Problem

In data parallelism, EVERY GPU stores:
- Full model weights
- Full optimizer states (m, v for Adam)
- Full gradients

For a 7B model with Adam:
- Weights: 14 GB (fp16)
- Optimizer states: 28 GB (fp32)
- Gradients: 14 GB (fp16)
- Total: 56 GB × N GPUs (all redundant!)

### ZeRO Stages

Progressively partition optimizer state, gradients, and parameters:

```python
def zero_stage_1(model, data, num_gpus=4):
    """
    ZeRO Stage 1: Partition optimizer states.

    Each GPU only stores 1/N of optimizer states (m, v).
    Gradients: all-reduce then scatter.
    """
    # Each GPU owns 1/N of parameters for optimizer
    my_gpu = 0
    my_params = get_params_for_gpu(model, my_gpu, num_gpus)

    # Forward/backward as normal (all GPUs have full model)
    loss = model.forward(data)
    grads = model.backward()

    # All-reduce gradients
    avg_grads = all_reduce(grads)

    # Update only MY parameters
    for param_id in my_params:
        # Only store optimizer state for my params
        model.optimizer.step(avg_grads[param_id], param_id)

    # All-gather updated parameters
    all_gather_params(model)


def zero_stage_2(model, data, num_gpus=4):
    """
    ZeRO Stage 2: Partition gradients too.

    Each GPU only computes/stores gradients for its parameters.
    """
    # Forward (all GPUs have full model)
    loss = model.forward(data)

    # Backward: reduce-scatter (each GPU gets its gradient portion)
    grads = model.backward()
    my_grads = reduce_scatter(grads)  # Only my portion, already reduced

    # Update my parameters
    model.optimizer.step(my_grads)

    # All-gather updated parameters
    all_gather_params(model)


def zero_stage_3(model, data, num_gpus=4):
    """
    ZeRO Stage 3: Partition parameters too.

    Each GPU only stores 1/N of everything.
    All-gather parameters just-in-time for forward/backward.
    """
    # All-gather parameters needed for forward
    full_params = all_gather_params(model.my_params)

    # Forward
    loss = model.forward_with_params(data, full_params)

    # Backward with parameter gathering
    grads = model.backward_with_params(full_params)

    # Reduce-scatter gradients
    my_grads = reduce_scatter(grads)

    # Update my parameters
    model.optimizer.step(my_grads)
```

### Memory Savings

| Setup | Weights | Optimizer | Gradients | Total |
|-------|---------|-----------|-----------|-------|
| No ZeRO | N×14GB | N×28GB | N×14GB | N×56GB |
| ZeRO-1 | N×14GB | 28GB | N×14GB | N×28GB + 28GB |
| ZeRO-2 | N×14GB | 28GB | 14GB | N×14GB + 42GB |
| ZeRO-3 | 14GB | 28GB | 14GB | 56GB total |

**What this means:** ZeRO-3 enables training models that don't fit on any single GPU by distributing all state across GPUs.

## Combining Parallelism Strategies

### 3D Parallelism

Modern large-model training uses all three:

```
            ┌─────────────────────────────────────────┐
            │           Data Parallel Replicas        │
            │  ┌───────────┐          ┌───────────┐  │
            │  │  Replica 1│          │  Replica 2│  │
            │  │           │          │           │  │
            │  │ Pipeline  │          │ Pipeline  │  │
            │  │ Stage 0-3 │          │ Stage 0-3 │  │
            │  │           │          │           │  │
            │  │ Tensor    │          │ Tensor    │  │
            │  │ Parallel  │          │ Parallel  │  │
            │  │ within    │          │ within    │  │
            │  │ each stage│          │ each stage│  │
            │  └───────────┘          └───────────┘  │
            └─────────────────────────────────────────┘
```

```python
def three_d_parallelism_config(model_size_b, num_gpus):
    """
    Configure 3D parallelism for given model and cluster.

    Example: 175B model on 1024 GPUs
    - Tensor parallel: 8 GPUs (within node, fast NVLink)
    - Pipeline parallel: 16 stages (across nodes)
    - Data parallel: 8 replicas (for throughput)
    - Total: 8 × 16 × 8 = 1024 GPUs
    """
    # Tensor parallel: limited by single-node GPUs
    tensor_parallel = min(8, num_gpus)

    # Pipeline parallel: limited by model depth
    num_layers = estimate_layers(model_size_b)
    pipeline_parallel = min(num_layers // 2, num_gpus // tensor_parallel)

    # Data parallel: remaining
    data_parallel = num_gpus // (tensor_parallel * pipeline_parallel)

    return {
        'tensor_parallel': tensor_parallel,
        'pipeline_parallel': pipeline_parallel,
        'data_parallel': data_parallel
    }
```

### Communication Hierarchy

Different parallelism types have different communication patterns:

| Type | Communication | Frequency | Best Interconnect |
|------|---------------|-----------|-------------------|
| Tensor | All-reduce per layer | Very high | NVLink (within node) |
| Pipeline | Point-to-point | Per micro-batch | InfiniBand |
| Data | All-reduce per step | Per batch | InfiniBand |

## Practical Considerations

### Checkpointing

Save and restore distributed training state:

```python
def save_distributed_checkpoint(model, optimizer, step, path):
    """
    Save checkpoint in distributed setting.

    Each rank saves its portion; consolidate for full checkpoint.
    """
    checkpoint = {
        'step': step,
        'model_state': get_local_model_state(model),  # My params
        'optimizer_state': get_local_optimizer_state(optimizer),  # My states
    }

    # Each rank saves to different file
    rank = get_rank()
    torch.save(checkpoint, f"{path}/checkpoint_{rank}.pt")


def load_distributed_checkpoint(model, optimizer, path):
    """Load checkpoint across distributed ranks."""
    rank = get_rank()
    checkpoint = torch.load(f"{path}/checkpoint_{rank}.pt")

    load_local_model_state(model, checkpoint['model_state'])
    load_local_optimizer_state(optimizer, checkpoint['optimizer_state'])

    return checkpoint['step']
```

### Fault Tolerance

Handle GPU failures in large clusters:

```python
def training_with_fault_tolerance(model, data, checkpoint_freq=100):
    """
    Training loop with automatic recovery.
    """
    step = load_latest_checkpoint(model)

    while step < total_steps:
        try:
            # Training step
            loss = train_step(model, data[step])
            step += 1

            # Checkpoint periodically
            if step % checkpoint_freq == 0:
                save_checkpoint(model, step)

        except (GPUError, NetworkError) as e:
            # Log error, restore from checkpoint
            print(f"Error at step {step}: {e}")
            step = load_latest_checkpoint(model)
            # Possibly reduce to working GPUs
            reconfigure_distributed(exclude_failed_gpus())
```

### Gradient Synchronization

```python
def synchronized_training_step(model, batch, num_gpus):
    """
    Ensure all GPUs stay synchronized.
    """
    # All GPUs must reach this point
    barrier()

    # Forward
    loss = model.forward(batch)

    # Backward
    grads = model.backward()

    # Synchronized gradient reduction
    avg_grads = all_reduce(grads)

    # All GPUs apply same update
    model.optimizer.step(avg_grads)

    # Verify synchronization (debugging)
    assert verify_model_sync(model, num_gpus)
```

## Summary

| Strategy | What It Splits | Communication | Use Case |
|----------|---------------|---------------|----------|
| Data Parallel | Data batches | All-reduce per batch | Standard scaling |
| Tensor Parallel | Layers (width) | All-reduce per layer | Within node |
| Pipeline Parallel | Layers (depth) | Point-to-point | Across nodes |
| ZeRO | Optimizer state | All-gather, reduce-scatter | Memory efficiency |

**The essential insight:** Distributed training is about managing trade-offs between computation, communication, and memory. Data parallelism scales well but is memory-limited. Model parallelism enables larger models but increases communication. ZeRO eliminates redundancy. Modern training combines all approaches based on model size, cluster topology, and interconnect speed.

**Historical context:** GPT-3 (2020) required 3D parallelism on thousands of GPUs. ZeRO (2020) from DeepSpeed democratized large model training. Megatron-LM and DeepSpeed became standard frameworks. Today's frontier models train on 10,000+ GPUs with sophisticated parallelism strategies.

**Infrastructure reality:** Training GPT-4 scale models requires not just GPUs but also:
- High-bandwidth interconnects (NVLink, InfiniBand)
- Massive storage systems (petabytes)
- Cooling infrastructure
- Power (megawatts)
- Reliability engineering (failures happen at scale)
