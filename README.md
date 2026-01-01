# Machine Learning

Understanding modern LLMs from first principles. ~30 interlinked documents covering mathematical foundations through transformers and modern training techniques.

## Documents

### Mathematical Foundations
| Document | Topic |
|----------|-------|
| [Linear Algebra](math-foundations/linear-algebra.md) | Vectors, matrices, eigenvalues, PCA, embeddings |
| [Calculus](math-foundations/calculus.md) | Derivatives, chain rule, gradients, Jacobians |
| [Probability](math-foundations/probability.md) | Distributions, Bayes, entropy, softmax, cross-entropy |
| [Optimization](math-foundations/optimization.md) | Gradient descent, convexity, saddle points, momentum |
| [Information Theory](math-foundations/information-theory.md) | KL divergence, mutual information, loss functions |

### Neural Networks
| Document | Topic |
|----------|-------|
| [Perceptron](neural-networks/perceptron.md) | Single neuron, linear classifiers, decision boundaries |
| [Multilayer Networks](neural-networks/multilayer-networks.md) | Universal approximation, depth vs width |
| [Backpropagation](neural-networks/backpropagation.md) | Chain rule, computational graphs, automatic differentiation |
| [Activation Functions](neural-networks/activation-functions.md) | Sigmoid, ReLU, GELU, vanishing gradients |
| [Regularization](neural-networks/regularization.md) | Dropout, weight decay, batch normalization |

### Architectures
| Document | Topic |
|----------|-------|
| [CNNs](architectures/cnns.md) | Convolution, pooling, translation invariance |
| [RNNs](architectures/rnns.md) | Sequential processing, hidden states, vanishing gradients |
| [LSTMs](architectures/lstms.md) | Gates, memory cells, gradient flow |
| [Sequence-to-Sequence](architectures/sequence-to-sequence.md) | Encoder-decoder, the road to attention |

### Transformers
| Document | Topic |
|----------|-------|
| [Attention](transformers/attention.md) | Dot-product attention, query-key-value |
| [Self-Attention](transformers/self-attention.md) | Attending to yourself, positional encoding |
| [Multi-Head Attention](transformers/multi-head-attention.md) | Parallel attention heads, different subspaces |
| [Transformer Architecture](transformers/transformer-architecture.md) | Full stack: embeddings to output |
| [BERT](transformers/bert.md) | Encoder-only, masked language modeling, fine-tuning |
| [GPT](transformers/gpt.md) | Decoder-only, autoregressive language models |

### Training
| Document | Topic |
|----------|-------|
| [Loss Functions](training/loss-functions.md) | Cross-entropy, perplexity, contrastive losses |
| [Optimizers](training/optimizers.md) | SGD, Adam, learning rate schedules |
| [Batching](training/batching.md) | Mini-batch, gradient accumulation |
| [Distributed Training](training/distributed-training.md) | Data parallel, model parallel, ZeRO |

### Modern LLMs
| Document | Topic |
|----------|-------|
| [Pretraining](modern-llms/pretraining.md) | Next-token prediction at scale |
| [Fine-tuning](modern-llms/fine-tuning.md) | Task adaptation, catastrophic forgetting |
| [RLHF](modern-llms/rlhf.md) | Reward models, PPO, alignment |
| [Instruction Tuning](modern-llms/instruction-tuning.md) | Supervised fine-tuning, chat formatting |
| [Efficient Adaptation](modern-llms/efficient-adaptation.md) | LoRA, adapters, prompt tuning |
| [Inference](modern-llms/inference.md) | KV caching, speculative decoding, quantization |

## Notebooks

Runnable implementations in NumPy:

| Notebook | Focus |
|----------|-------|
| [01-numpy-neural-net](notebooks/01-numpy-neural-net.ipynb) | 2-layer net from scratch |
| [02-backprop-from-scratch](notebooks/02-backprop-from-scratch.ipynb) | Manual gradient computation |

## Folder Structure

```
machine-learning/
├── math-foundations/       # Linear algebra, calculus, probability, optimization
├── neural-networks/        # Perceptrons, backprop, activations, regularization
├── architectures/          # CNNs, RNNs, LSTMs (historical context)
├── transformers/           # Attention, self-attention, BERT, GPT
├── training/               # Loss functions, optimizers, distributed
├── modern-llms/            # Pretraining, RLHF, LoRA, inference
└── notebooks/              # Runnable implementations
```

## How Documents Connect

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

## Reading Order

**New to ML math?** Start with foundations:
- [Linear Algebra](math-foundations/linear-algebra.md) → [Calculus](math-foundations/calculus.md) → [Probability](math-foundations/probability.md) for the core math
- [Optimization](math-foundations/optimization.md) for gradient descent intuition

**Ready for neural networks?** Choose a track:

| Track | Path |
|-------|------|
| **Quickstart** | Linear Algebra → Perceptron → Backprop → Attention → Transformer Architecture → GPT |
| **Foundations-first** | All math-foundations → Neural networks in order → Architectures → Transformers |
| **Practitioner deep-dive** | Attention → Transformer Architecture → GPT → All modern-llms |
| **Historical** | Perceptron → CNNs → RNNs → LSTMs → Seq2Seq → Attention (the research arc) |

## Code Examples

Documents include inline code snippets showing math translated to NumPy/PyTorch:

```python
def attention(Q, K, V):
    """The core attention mechanism in 4 lines."""
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # similarity
    weights = softmax(scores)         # normalize
    return weights @ V                # weighted sum
```

Notebooks provide complete, runnable implementations for hands-on learning.

## Rendering

GitHub renders LaTeX natively. For local viewing, use VS Code with a math-preview extension.

Example: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
