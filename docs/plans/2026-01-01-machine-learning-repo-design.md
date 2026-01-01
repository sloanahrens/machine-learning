# Machine Learning Repository Design

This document outlines the design for a machine learning educational repository following the structure and spirit of physics-stuff.

## Overview

**Goal:** Explain modern LLM architecture by building prerequisite understanding from foundational concepts through transformers and modern training techniques.

**Approach:** Interlinked markdown documents with LaTeX equations and inline code, paired with runnable Jupyter notebooks.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Audience | All levels | Layered depth - beginners through practitioners |
| Math foundations | Self-contained, ML-focused | Not dependent on physics-stuff; ML-specific examples |
| Code integration | Inline snippets + notebooks | Snippets for immediacy, notebooks for experimentation |
| Endpoint | Modern LLM techniques | RLHF, LoRA, inference optimization - enough to understand ChatGPT |
| Frameworks | NumPy → PyTorch | NumPy for fundamentals, PyTorch when complexity demands |

## Repository Structure

```
machine-learning/
├── README.md                    # Overview, document table, dependency graph, reading paths
├── ROADMAP.md                   # Future additions
├── math-foundations/            # Self-contained ML math
│   ├── linear-algebra.md        # Vectors, matrices, eigenvalues → PCA, embeddings
│   ├── calculus.md              # Derivatives, chain rule, gradients
│   ├── probability.md           # Distributions, Bayes, entropy → softmax, cross-entropy
│   ├── optimization.md          # Gradient descent, convexity, saddle points
│   └── information-theory.md    # KL divergence, mutual information → loss functions
├── neural-networks/             # Core NN concepts
│   ├── perceptron.md            # Single neuron, linear classifiers
│   ├── multilayer-networks.md   # Universal approximation, depth vs width
│   ├── backpropagation.md       # Chain rule, computational graphs
│   ├── activation-functions.md  # Sigmoid, ReLU, GELU, why they matter
│   └── regularization.md        # Dropout, weight decay, batch norm
├── architectures/               # Network designs
│   ├── cnns.md                  # Convolution, pooling (brief, for context)
│   ├── rnns.md                  # Sequential processing, vanishing gradients
│   ├── lstms.md                 # Gates, memory cells
│   └── sequence-to-sequence.md  # Encoder-decoder, the road to attention
├── transformers/                # The main event
│   ├── attention.md             # Dot-product attention, why it works
│   ├── self-attention.md        # Query-key-value, positional encoding
│   ├── multi-head-attention.md  # Parallel attention heads
│   ├── transformer-architecture.md  # Full stack: embeddings → output
│   ├── bert.md                  # Encoder-only, MLM, fine-tuning
│   └── gpt.md                   # Decoder-only, autoregressive LMs
├── training/                    # How models learn
│   ├── loss-functions.md        # Cross-entropy, perplexity
│   ├── optimizers.md            # SGD, Adam, learning rate schedules
│   ├── batching.md              # Mini-batch, gradient accumulation
│   └── distributed-training.md  # Data parallel, model parallel
├── modern-llms/                 # Post-GPT techniques
│   ├── pretraining.md           # Next-token prediction at scale
│   ├── fine-tuning.md           # Task adaptation, catastrophic forgetting
│   ├── rlhf.md                  # Reward models, PPO, alignment
│   ├── instruction-tuning.md    # SFT, chat formatting
│   ├── efficient-adaptation.md  # LoRA, adapters, prompt tuning
│   └── inference.md             # KV caching, speculative decoding, quantization
└── notebooks/                   # Runnable implementations
    ├── 01-numpy-neural-net.ipynb
    ├── 02-backprop-from-scratch.ipynb
    ├── 03-activations-visualized.ipynb
    ├── 04-optimization-landscape.ipynb
    ├── 05-rnn-from-scratch.ipynb
    ├── 06-lstm-from-scratch.ipynb
    ├── 07-attention-from-scratch.ipynb
    ├── 08-minimal-transformer.ipynb
    ├── 09-minimal-gpt.ipynb
    ├── 10-fine-tuning-basics.ipynb
    ├── 11-lora-from-scratch.ipynb
    └── 12-kv-cache-demo.ipynb
```

## Document Format

Each document follows this template:

```markdown
# Topic Name

Opening paragraph with the key equation/concept boxed, plus one-sentence significance.

Prerequisites: [link1](path), [link2](path). For code: `numpy`, `torch` (if needed).

---

## Section 1: Core Concept

### Mathematical Foundation

$$
\boxed{\text{key equation}}
$$

**What this means:** Plain-English intuition, 2-3 sentences.

### In Code

\```python
# Inline snippet showing the math as NumPy/PyTorch
\```

**What the code shows:** Connect implementation to math.

## Section 2: Deeper Treatment

For readers wanting more depth.

## Section 3: Practical Considerations

Numerical stability, common bugs, hyperparameters.

## Summary

| Concept | Key Formula | Code Pattern | Where It's Used |
|---------|-------------|--------------|-----------------|

**The essential insight:** One paragraph synthesis.

**Notebook:** [link](path) for hands-on implementation.
```

## Dependency Graph

```
MATH FOUNDATIONS                           NEURAL NETWORKS                    TRANSFORMERS → LLMs
──────────────────                         ───────────────                    ──────────────────

Linear Algebra ──┬──► Perceptron ──────────► Multilayer Networks
                 │         │                        │
Calculus ────────┼─────────┴──► Backpropagation ◄───┘
                 │                    │
Probability ─────┼────────────────────┼──► Loss Functions
                 │                    │          │
Optimization ────┴────────────────────┴──► Optimizers
                                             │
Information Theory ──────────────────────────┤
                                             ▼
                 CNNs ◄─────────── Activation Functions
                   │                         │
                   │              Regularization
                   │                         │
                   └────► RNNs ──► LSTMs ────┤
                                    │        │
                         Seq2Seq ◄──┘        │
                            │                │
                            ▼                ▼
                      ┌─────────────────────────┐
                      │       ATTENTION         │
                      └─────────────────────────┘
                                 │
                      Self-Attention + Positional Encoding
                                 │
                      Multi-Head Attention
                                 │
                      Transformer Architecture
                            ┌────┴────┐
                            ▼         ▼
                          BERT       GPT
                            │         │
                            └────┬────┘
                                 ▼
                           Pretraining
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              Fine-tuning      RLHF    Instruction Tuning
                    │            │            │
                    └────────────┼────────────┘
                                 ▼
                      Efficient Adaptation (LoRA)
                                 │
                              Inference
```

## Reading Paths

| Track | Path |
|-------|------|
| **Quickstart** | Linear Algebra → Perceptron → Backprop → Attention → Transformer Architecture → GPT |
| **Foundations-first** | All math-foundations → Neural networks in order → Architectures → Transformers |
| **Practitioner deep-dive** | Attention → Transformer Architecture → GPT → All modern-llms |
| **Historical** | Perceptron → CNNs → RNNs → LSTMs → Seq2Seq → Attention (the research arc) |

## Notebooks

| Notebook | Focus | Framework | Pairs With |
|----------|-------|-----------|------------|
| `01-numpy-neural-net.ipynb` | 2-layer net from scratch | NumPy | perceptron.md, multilayer-networks.md |
| `02-backprop-from-scratch.ipynb` | Manual gradient computation | NumPy | backpropagation.md |
| `03-activations-visualized.ipynb` | Plot activations, dead ReLU | NumPy + matplotlib | activation-functions.md |
| `04-optimization-landscape.ipynb` | Visualize SGD, Adam | NumPy + matplotlib | optimization.md, optimizers.md |
| `05-rnn-from-scratch.ipynb` | Character-level RNN | NumPy | rnns.md |
| `06-lstm-from-scratch.ipynb` | LSTM cell implementation | NumPy | lstms.md |
| `07-attention-from-scratch.ipynb` | Dot-product attention | NumPy | attention.md, self-attention.md |
| `08-minimal-transformer.ipynb` | Single-layer transformer | PyTorch | transformer-architecture.md |
| `09-minimal-gpt.ipynb` | Train tiny GPT | PyTorch | gpt.md |
| `10-fine-tuning-basics.ipynb` | Fine-tune on classification | PyTorch + HF | fine-tuning.md |
| `11-lora-from-scratch.ipynb` | Implement LoRA | PyTorch | efficient-adaptation.md |
| `12-kv-cache-demo.ipynb` | KV caching speedup | PyTorch | inference.md |

## Implementation Phases

### Phase 1: Foundation (Core Path)
Minimum viable path from zero to transformers:

- math-foundations: linear-algebra.md, calculus.md, probability.md
- neural-networks: perceptron.md, multilayer-networks.md, backpropagation.md, activation-functions.md
- transformers: attention.md, self-attention.md, transformer-architecture.md, gpt.md
- notebooks: 01, 02, 07, 08, 09

### Phase 2: Depth
Supporting material:

- math-foundations: optimization.md, information-theory.md
- neural-networks: regularization.md
- training: loss-functions.md, optimizers.md
- architectures: rnns.md, lstms.md, sequence-to-sequence.md
- transformers: multi-head-attention.md, bert.md
- notebooks: 03, 04, 05, 06

### Phase 3: Modern LLMs
The "why ChatGPT works" layer:

- modern-llms: all 6 documents
- notebooks: 10, 11, 12

### Phase 4: Polish
- architectures: cnns.md
- training: batching.md, distributed-training.md
- Cross-reference audit
- README finalization

## Totals

- **~30 documents** across 6 folders
- **12 notebooks** progressing NumPy → PyTorch
- **4 reading paths** for different audiences
