# Notebook Implementation Plan

10 remaining notebooks to build, progressing from NumPy fundamentals to PyTorch transformer implementations.

## Completed

- **01-numpy-neural-net** - 2-layer network from scratch
- **02-backprop-from-scratch** - Manual gradient computation

## Phase 1: NumPy Fundamentals

### 03-activations-visualized

**Goal:** Visualize activation functions and understand their behaviors

**Content:**
- Plot sigmoid, tanh, ReLU, GELU, Swish
- Demonstrate vanishing gradients with deep sigmoid networks
- Show dead ReLU problem with negative inputs
- Compare gradient flow through different activations
- Interactive: Train same network with different activations

**Prerequisites:** activation-functions.md

**Dependencies:** matplotlib, numpy

---

### 04-optimization-landscape

**Goal:** Build intuition for optimization algorithms

**Content:**
- Visualize loss landscape on 2D functions (Rosenbrock, saddle points)
- Implement SGD, momentum, RMSprop, Adam from scratch
- Animate optimizer paths on contour plots
- Show learning rate effects (too high, too low, just right)
- Compare convergence speed and stability

**Prerequisites:** optimization.md, optimizers.md

**Dependencies:** matplotlib, numpy

---

## Phase 2: Sequence Models (NumPy)

### 05-rnn-from-scratch

**Goal:** Build a character-level RNN from scratch

**Content:**
- Implement RNN cell: `h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t)`
- Forward pass with hidden state propagation
- Backpropagation through time (BPTT)
- Train on tiny text (names, short sequences)
- Demonstrate vanishing gradients with long sequences

**Prerequisites:** rnns.md, backpropagation.md

**Dependencies:** numpy

---

### 06-lstm-from-scratch

**Goal:** Implement LSTM cell and compare to vanilla RNN

**Content:**
- Implement forget, input, output gates
- Cell state update mechanism
- Compare gradient flow: LSTM vs RNN on long sequences
- Train on same task as notebook 05, show better long-range learning

**Prerequisites:** lstms.md, notebook 05

**Dependencies:** numpy

---

## Phase 3: Attention (NumPy → PyTorch transition)

### 07-attention-from-scratch

**Goal:** Implement attention mechanism from first principles

**Content:**
- Scaled dot-product attention in NumPy
- Query, key, value intuition with visualizations
- Attention weights heatmap on example sequences
- Masked attention for autoregressive models
- Compare attention to RNN hidden state bottleneck

**Prerequisites:** attention.md, self-attention.md

**Dependencies:** numpy, matplotlib

---

### 08-minimal-transformer

**Goal:** Build a minimal single-layer transformer

**Content:**
- Multi-head attention implementation
- Position-wise feedforward network
- Positional encoding (sinusoidal)
- Layer normalization and residual connections
- Complete encoder block
- Train on simple sequence task (copy, reverse)

**Prerequisites:** multi-head-attention.md, transformer-architecture.md

**Dependencies:** PyTorch (transition point)

---

## Phase 4: GPT and Training (PyTorch)

### 09-minimal-gpt

**Goal:** Train a tiny GPT model from scratch

**Content:**
- Decoder-only architecture
- Causal masking
- Train on Shakespeare (or similar small corpus)
- Generate text samples during training
- Show loss curves, perplexity

**Prerequisites:** gpt.md, loss-functions.md

**Dependencies:** PyTorch

**Reference:** Karpathy's minGPT/nanoGPT

---

### 10-fine-tuning-basics

**Goal:** Fine-tune a pretrained model on downstream task

**Content:**
- Load small pretrained model (e.g., DistilBERT or GPT-2 small)
- Add classification head
- Demonstrate catastrophic forgetting
- Learning rate schedules for fine-tuning
- Compare frozen vs unfrozen approaches

**Prerequisites:** fine-tuning.md, pretraining.md

**Dependencies:** PyTorch, transformers (HuggingFace)

---

## Phase 5: Modern Techniques (PyTorch)

### 11-lora-from-scratch

**Goal:** Implement LoRA adaptation

**Content:**
- Low-rank decomposition concept
- Implement LoRA layer wrapping linear layers
- Compare parameter counts: full fine-tuning vs LoRA
- Train LoRA on same task as notebook 10
- Show comparable performance with fewer parameters

**Prerequisites:** efficient-adaptation.md

**Dependencies:** PyTorch

---

### 12-kv-cache-demo

**Goal:** Demonstrate KV caching speedup

**Content:**
- Naive autoregressive generation (recompute everything)
- KV cache implementation
- Benchmark: time per token with and without caching
- Memory usage tradeoffs
- Visualize what's being cached

**Prerequisites:** inference.md

**Dependencies:** PyTorch

---

## Implementation Notes

### Framework Progression

| Notebooks | Framework | Rationale |
|-----------|-----------|-----------|
| 03-07 | NumPy | Build understanding from scratch |
| 08-12 | PyTorch | Practical implementation, GPU training |

### Notebook Style

Each notebook should:
1. State learning objectives at the top
2. Link to relevant markdown documents
3. Build implementations incrementally
4. Include visualizations where helpful
5. End with exercises or extensions

### Estimated Complexity

| Notebook | Lines of Code | Training Time |
|----------|---------------|---------------|
| 03 | ~150 | None (visualization only) |
| 04 | ~200 | None (visualization only) |
| 05 | ~250 | Minutes (CPU) |
| 06 | ~300 | Minutes (CPU) |
| 07 | ~200 | None |
| 08 | ~400 | Minutes (GPU helpful) |
| 09 | ~500 | 10-30 min (GPU) |
| 10 | ~300 | 5-10 min (GPU) |
| 11 | ~350 | 5-10 min (GPU) |
| 12 | ~200 | None (benchmark only) |

## Suggested Order

Build in order (03 → 12) for natural progression, or by phase:

1. **Phase 1** (03-04): Foundation visualizations, can be done in any order
2. **Phase 2** (05-06): Sequence models, must be in order
3. **Phase 3** (07-08): Attention → Transformer, must be in order
4. **Phase 4-5** (09-12): Can be somewhat parallelized after 08
