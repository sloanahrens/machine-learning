# Machine Learning Repository Roadmap

This document outlines the implementation phases and future additions for the machine-learning repository.

## Overview

| Phase | Focus | Documents | Status |
|-------|-------|-----------|--------|
| 1 | Foundation (Core Path) | 11 docs, 5 notebooks | Complete |
| 2 | Depth | 9 docs, 4 notebooks | Complete |
| 3 | Modern LLMs | 6 docs, 3 notebooks | Complete |
| 4 | Polish | 3 docs, cross-refs | Complete |

---

## Phase 1: Foundation (Core Path)

Minimum viable path from zero to understanding transformers.

### Math Foundations (3 documents)

#### 1.1 Linear Algebra
**File:** `math-foundations/linear-algebra.md`

**Outline:**
- Vectors and vector spaces
- Matrix multiplication and its meaning
- Eigenvalues and eigenvectors → PCA intuition
- Dot products → similarity and attention
- Matrix decompositions (SVD overview)

**ML connections:** Embeddings, weight matrices, attention scores

---

#### 1.2 Calculus
**File:** `math-foundations/calculus.md`

**Outline:**
- Derivatives and the chain rule
- Partial derivatives and gradients
- Jacobians (for understanding backprop)
- Taylor series (for optimization intuition)

**ML connections:** Backpropagation, gradient descent

---

#### 1.3 Probability
**File:** `math-foundations/probability.md`

**Outline:**
- Probability distributions (discrete and continuous)
- Conditional probability and Bayes' theorem
- Expectation and variance
- Entropy and cross-entropy
- Softmax as probability distribution

**ML connections:** Loss functions, output layers, sampling

---

### Neural Networks (4 documents)

#### 1.4 Perceptron
**File:** `neural-networks/perceptron.md`

**Outline:**
- The single neuron model
- Linear classifiers and decision boundaries
- Weights, bias, activation
- Learning rule (gradient descent preview)
- Limitations: XOR problem

**Notebook:** `01-numpy-neural-net.ipynb`

---

#### 1.5 Multilayer Networks
**File:** `neural-networks/multilayer-networks.md`

**Outline:**
- Hidden layers and depth
- Universal approximation theorem
- Width vs depth trade-offs
- Forward pass computation
- Representation learning

**Notebook:** `01-numpy-neural-net.ipynb`

---

#### 1.6 Backpropagation
**File:** `neural-networks/backpropagation.md`

**Outline:**
- The chain rule in action
- Computational graphs
- Forward and backward passes
- Gradient flow and accumulation
- Automatic differentiation

**Notebook:** `02-backprop-from-scratch.ipynb`

---

#### 1.7 Activation Functions
**File:** `neural-networks/activation-functions.md`

**Outline:**
- Why non-linearity matters
- Sigmoid and tanh (and their problems)
- ReLU and variants (Leaky, PReLU)
- GELU (used in transformers)
- Vanishing/exploding gradients

---

### Transformers (4 documents)

#### 1.8 Attention
**File:** `transformers/attention.md`

**Outline:**
- The problem: fixed representations in RNNs
- Attention as weighted retrieval
- Query, key, value decomposition
- Scaled dot-product attention
- Attention weights visualization

**Notebook:** `07-attention-from-scratch.ipynb`

---

#### 1.9 Self-Attention
**File:** `transformers/self-attention.md`

**Outline:**
- Attending to yourself (not encoder-decoder)
- Positional encoding (sinusoidal, learned)
- Why position information is needed
- Masking for autoregressive models

**Notebook:** `07-attention-from-scratch.ipynb`

---

#### 1.10 Transformer Architecture
**File:** `transformers/transformer-architecture.md`

**Outline:**
- Full architecture diagram
- Embeddings → attention → FFN → output
- Layer normalization and residual connections
- Encoder vs decoder stacks
- "Attention Is All You Need" context

**Notebook:** `08-minimal-transformer.ipynb`

---

#### 1.11 GPT
**File:** `transformers/gpt.md`

**Outline:**
- Decoder-only architecture
- Autoregressive language modeling
- Causal masking
- Scaling properties
- From GPT-1 to GPT-4 (conceptual)

**Notebook:** `09-minimal-gpt.ipynb`

---

## Phase 2: Depth

Supporting material for deeper understanding.

### Math Foundations (2 documents)

#### 2.1 Optimization
**File:** `math-foundations/optimization.md`

**Outline:**
- Gradient descent variants
- Convexity and local minima
- Saddle points in high dimensions
- Momentum and acceleration
- Learning rate intuition

**Notebook:** `04-optimization-landscape.ipynb`

---

#### 2.2 Information Theory
**File:** `math-foundations/information-theory.md`

**Outline:**
- Shannon entropy
- Cross-entropy and KL divergence
- Mutual information
- Connection to loss functions
- Bits and nats

---

### Neural Networks (1 document)

#### 2.3 Regularization
**File:** `neural-networks/regularization.md`

**Outline:**
- Overfitting and generalization
- L1/L2 weight decay
- Dropout
- Batch normalization
- Layer normalization (for transformers)

**Notebook:** `03-activations-visualized.ipynb` (includes dropout effects)

---

### Training (2 documents)

#### 2.4 Loss Functions
**File:** `training/loss-functions.md`

**Outline:**
- Mean squared error
- Cross-entropy loss
- Perplexity for language models
- Contrastive losses (brief)

---

#### 2.5 Optimizers
**File:** `training/optimizers.md`

**Outline:**
- SGD with momentum
- Adam and AdamW
- Learning rate schedules (warmup, decay)
- Gradient clipping

**Notebook:** `04-optimization-landscape.ipynb`

---

### Architectures (3 documents)

#### 2.6 RNNs
**File:** `architectures/rnns.md`

**Outline:**
- Sequential processing
- Hidden state as memory
- Vanishing gradient problem
- Backpropagation through time

**Notebook:** `05-rnn-from-scratch.ipynb`

---

#### 2.7 LSTMs
**File:** `architectures/lstms.md`

**Outline:**
- Gates: forget, input, output
- Cell state and hidden state
- Gradient highways
- GRU as simplification

**Notebook:** `06-lstm-from-scratch.ipynb`

---

#### 2.8 Sequence-to-Sequence
**File:** `architectures/sequence-to-sequence.md`

**Outline:**
- Encoder-decoder architecture
- The bottleneck problem
- Teacher forcing
- The motivation for attention

---

### Transformers (2 documents)

#### 2.9 Multi-Head Attention
**File:** `transformers/multi-head-attention.md`

**Outline:**
- Multiple attention heads in parallel
- Different representation subspaces
- Concatenation and projection
- Interpretation of heads

---

#### 2.10 BERT
**File:** `transformers/bert.md`

**Outline:**
- Encoder-only architecture
- Masked language modeling (MLM)
- Next sentence prediction
- Fine-tuning paradigm
- BERT variants

---

## Phase 3: Modern LLMs

The "why ChatGPT works" layer.

### 3.1 Pretraining
**File:** `modern-llms/pretraining.md`

**Outline:**
- Next-token prediction objective
- Web-scale data collection
- Tokenization (BPE, SentencePiece)
- Compute requirements and scaling laws

---

### 3.2 Fine-tuning
**File:** `modern-llms/fine-tuning.md`

**Outline:**
- Task-specific adaptation
- Catastrophic forgetting
- Few-shot vs full fine-tuning
- Evaluation metrics

**Notebook:** `10-fine-tuning-basics.ipynb`

---

### 3.3 RLHF
**File:** `modern-llms/rlhf.md`

**Outline:**
- Why supervised learning isn't enough
- Reward modeling from human preferences
- PPO for policy optimization
- Constitutional AI and alternatives
- Alignment as a goal

---

### 3.4 Instruction Tuning
**File:** `modern-llms/instruction-tuning.md`

**Outline:**
- Supervised fine-tuning (SFT)
- Instruction-response formatting
- Chat templates and system prompts
- FLAN, InstructGPT, ChatGPT

---

### 3.5 Efficient Adaptation
**File:** `modern-llms/efficient-adaptation.md`

**Outline:**
- The problem: full fine-tuning is expensive
- LoRA: low-rank adaptation
- Adapters and prefix tuning
- Prompt tuning
- When to use what

**Notebook:** `11-lora-from-scratch.ipynb`

---

### 3.6 Inference
**File:** `modern-llms/inference.md`

**Outline:**
- KV caching for autoregressive generation
- Speculative decoding
- Quantization (INT8, INT4)
- Batching strategies
- Memory vs compute trade-offs

**Notebook:** `12-kv-cache-demo.ipynb`

---

## Phase 4: Polish

### 4.1 CNNs (brief)
**File:** `architectures/cnns.md`

**Outline:**
- Convolution operation
- Pooling and stride
- Translation invariance
- Historical context (AlexNet, ResNet)

---

### 4.2 Batching
**File:** `training/batching.md`

**Outline:**
- Mini-batch gradient descent
- Batch size effects
- Gradient accumulation
- Dynamic batching for sequences

---

### 4.3 Distributed Training
**File:** `training/distributed-training.md`

**Outline:**
- Data parallelism
- Model parallelism
- Pipeline parallelism
- ZeRO optimizer states

---

### 4.4 Cross-Reference Audit

Review all documents and ensure:
- Prerequisites linked at top
- Related documents cross-referenced in context
- Notebook links present where applicable
- Dependency graph in README matches actual links

---

## Progress Tracking

| Document | Phase | Status | Dependencies Met |
|----------|-------|--------|------------------|
| linear-algebra.md | 1 | ✅ Complete | Yes |
| calculus.md | 1 | ✅ Complete | Yes |
| probability.md | 1 | ✅ Complete | Yes |
| perceptron.md | 1 | ✅ Complete | Yes |
| multilayer-networks.md | 1 | ✅ Complete | Yes |
| backpropagation.md | 1 | ✅ Complete | Yes |
| activation-functions.md | 1 | ✅ Complete | Yes |
| attention.md | 1 | ✅ Complete | Yes |
| self-attention.md | 1 | ✅ Complete | Yes |
| transformer-architecture.md | 1 | ✅ Complete | Yes |
| gpt.md | 1 | ✅ Complete | Yes |
| optimization.md | 2 | ✅ Complete | Yes |
| information-theory.md | 2 | ✅ Complete | Yes |
| regularization.md | 2 | ✅ Complete | Yes |
| loss-functions.md | 2 | ✅ Complete | Yes |
| optimizers.md | 2 | ✅ Complete | Yes |
| rnns.md | 2 | ✅ Complete | Yes |
| lstms.md | 2 | ✅ Complete | Yes |
| sequence-to-sequence.md | 2 | ✅ Complete | Yes |
| multi-head-attention.md | 2 | ✅ Complete | Yes |
| bert.md | 2 | ✅ Complete | Yes |
| pretraining.md | 3 | ✅ Complete | Yes |
| fine-tuning.md | 3 | ✅ Complete | Yes |
| rlhf.md | 3 | ✅ Complete | Yes |
| instruction-tuning.md | 3 | ✅ Complete | Yes |
| efficient-adaptation.md | 3 | ✅ Complete | Yes |
| inference.md | 3 | ✅ Complete | Yes |
| cnns.md | 4 | ✅ Complete | Yes |
| batching.md | 4 | ✅ Complete | Yes |
| distributed-training.md | 4 | ✅ Complete | Yes |

---

## Final State

Upon completion:
- **~30 documents** across 6 folders
- **12 notebooks** with runnable implementations
- **Complete path** from linear algebra to modern LLM training
- **All major concepts** interlinked with prerequisites and related topics
