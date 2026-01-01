# Pretraining

$$
\boxed{L = -\sum_{t} \log p(x_t | x_{<t})}
$$

**Pretraining** is how large language models learn from the internet. By predicting the next token billions of times across trillions of tokens, models develop remarkably general capabilities—reasoning, coding, following instructions—all from one simple objective. Understanding pretraining explains why LLMs know what they know and why they fail how they fail.

Prerequisites: [GPT](../transformers/gpt.md), [information theory](../math-foundations/information-theory.md). Code: `numpy`.

---

## The Next-Token Prediction Objective

### The Core Idea

Given a sequence of tokens, predict the next one:

```
Input:  "The quick brown fox"
Target: "jumps"
```

Repeat this across all of the internet's text. That's pretraining.

$$
L = -\frac{1}{T} \sum_{t=1}^{T} \log p(x_t | x_1, \ldots, x_{t-1})
$$

```python
import numpy as np

def pretraining_loss(model, tokens):
    """
    Next-token prediction loss.

    Args:
        model: Transformer LM
        tokens: [batch_size, seq_len] token indices
    """
    # Input: all tokens except last
    # Target: all tokens except first
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    # Forward pass
    logits = model.forward(inputs)  # [batch, seq-1, vocab]

    # Cross-entropy
    log_probs = log_softmax(logits)
    loss = -log_probs[
        np.arange(logits.shape[0])[:, None],
        np.arange(logits.shape[1]),
        targets
    ]
    return loss.mean()

def log_softmax(x):
    max_x = np.max(x, axis=-1, keepdims=True)
    return x - max_x - np.log(np.sum(np.exp(x - max_x), axis=-1, keepdims=True))
```

### Why This Works

Next-token prediction seems simple, but it requires:

**Syntax:** To predict "jumps", know that verbs follow noun phrases
**Semantics:** To predict "Paris" after "The capital of France is", know facts
**Reasoning:** To predict "4" after "2 + 2 =", do arithmetic
**Style:** To predict "thy" in Shakespeare, recognize archaic English

```
What must the model learn to predict well?

"The Eiffel Tower is in ___"
→ Geography: Paris

"def fibonacci(n):\n    if n <= 1:\n        return ___"
→ Programming: n

"The patient presents with fever and ___"
→ Medical knowledge: (symptoms)

"If it's raining, then the ground is ___"
→ Causal reasoning: wet
```

**What this means:** The objective is simple, but the data contains everything. Models learn whatever patterns help predict text—including factual knowledge, logical reasoning, and stylistic conventions.

## Tokenization

### Why Tokenize?

Raw text is characters, but character-level models are inefficient:
- Vocabulary too small (just ~100 characters)
- Sequences too long (each word = many characters)
- Hard to capture word-level patterns

Word-level is too large:
- Every word form is separate ("run", "runs", "running")
- Can't handle new words (OOV = out-of-vocabulary)

### Subword Tokenization

Split words into meaningful pieces:

```
"unhappiness" → ["un", "happiness"]
"tokenization" → ["token", "ization"]
"transformers" → ["transform", "ers"]
"ChatGPT" → ["Chat", "G", "PT"]
```

### Byte Pair Encoding (BPE)

Start with characters, iteratively merge frequent pairs:

```python
def learn_bpe(texts, num_merges):
    """
    Learn BPE vocabulary from texts.

    1. Start with character vocabulary
    2. Find most frequent adjacent pair
    3. Merge into new token
    4. Repeat
    """
    # Initialize with characters
    vocab = set()
    for text in texts:
        for char in text:
            vocab.add(char)

    word_freqs = get_word_frequencies(texts)
    splits = {word: list(word) for word in word_freqs}

    merges = []
    for _ in range(num_merges):
        # Count pairs
        pair_freqs = {}
        for word, freq in word_freqs.items():
            symbols = splits[word]
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq

        if not pair_freqs:
            break

        # Find most frequent pair
        best_pair = max(pair_freqs, key=pair_freqs.get)
        merges.append(best_pair)

        # Merge in all words
        new_token = best_pair[0] + best_pair[1]
        vocab.add(new_token)
        for word in splits:
            symbols = splits[word]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (i < len(symbols) - 1 and
                    symbols[i] == best_pair[0] and
                    symbols[i + 1] == best_pair[1]):
                    new_symbols.append(new_token)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            splits[word] = new_symbols

    return vocab, merges


def apply_bpe(word, merges):
    """Apply learned BPE merges to tokenize a word."""
    symbols = list(word)

    for merge in merges:
        new_symbols = []
        i = 0
        while i < len(symbols):
            if (i < len(symbols) - 1 and
                symbols[i] == merge[0] and
                symbols[i + 1] == merge[1]):
                new_symbols.append(merge[0] + merge[1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols

    return symbols
```

### Common Tokenizers

| Tokenizer | Used By | Vocabulary Size |
|-----------|---------|-----------------|
| BPE | GPT-2, GPT-3 | 50,257 |
| SentencePiece | LLaMA, T5 | 32,000 |
| Tiktoken | GPT-4 | ~100,000 |
| WordPiece | BERT | 30,522 |

**What this means:** Tokenization determines what "words" the model sees. A 50k vocabulary with BPE handles most text efficiently—common words are single tokens, rare words split into recognizable pieces.

## Training Data

### Scale

Modern LLMs train on trillions of tokens:

| Model | Training Tokens | Dataset Size |
|-------|----------------|--------------|
| GPT-2 | 40B | ~40GB |
| GPT-3 | 300B | ~570GB |
| LLaMA | 1.4T | ~1.4TB |
| GPT-4 | ~13T (est.) | ~13TB (est.) |

### Data Sources

Typical training mix:

| Source | Proportion | Why |
|--------|------------|-----|
| Web crawl (CommonCrawl) | 60% | Scale, diversity |
| Books | 15% | Long-form, quality |
| Wikipedia | 5% | Factual, structured |
| Code (GitHub) | 15% | Programming ability |
| Academic papers | 5% | Technical knowledge |

### Data Quality

Raw web text is messy. Processing pipeline:

```python
def process_document(text):
    """Basic document processing."""
    # 1. Language detection
    if not is_english(text):
        return None

    # 2. Quality filtering
    if len(text) < 100:  # Too short
        return None
    if len(text.split()) / len(text) < 0.3:  # Not enough words
        return None

    # 3. Deduplication (hashing)
    doc_hash = hash_document(text)
    if doc_hash in seen_hashes:
        return None
    seen_hashes.add(doc_hash)

    # 4. Content filtering
    if contains_toxic_content(text):
        return None

    # 5. Personal information removal
    text = remove_pii(text)

    return text
```

### Data Repetition

Models see each token approximately once—or a few times for high-quality data:

```
Epochs by data quality:
- Web crawl: 1-2 epochs
- Books: 2-4 epochs
- Code: 2-4 epochs
- Wikipedia: 4-6 epochs
```

**What this means:** Unlike human learning, LLMs don't memorize through repetition. They see most text once. What matters is having vast, diverse data covering many domains.

## Scaling Laws

### The Discovery

Model performance follows predictable power laws:

$$
L \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty
$$

Where:
- $L$ = loss
- $N$ = model parameters
- $D$ = dataset tokens
- $L_\infty$ = irreducible loss

### Chinchilla Scaling

Optimal compute allocation:

$$
N \propto C^{0.5}, \quad D \propto C^{0.5}
$$

**What this means:** Double compute → double parameters AND double data.

Before Chinchilla, models were undertrained. GPT-3 (175B params on 300B tokens) could have achieved similar performance with 70B params on 1.4T tokens.

### Scaling Example

```python
def estimate_performance(params_billions, tokens_trillions):
    """
    Rough estimate of model loss from scaling laws.
    (Simplified; real scaling is more complex)
    """
    # Empirical constants (approximate)
    A = 400  # Parameter scaling constant
    B = 0.25  # Token scaling constant
    alpha = 0.34  # Parameter exponent
    beta = 0.28  # Token exponent
    L_inf = 1.69  # Irreducible loss (nats)

    N = params_billions * 1e9
    D = tokens_trillions * 1e12

    loss = (A / N) ** alpha + (B * 1e12 / D) ** beta + L_inf
    return loss

# Compare models
print(f"7B on 1T: {estimate_performance(7, 1):.3f}")
print(f"70B on 1T: {estimate_performance(70, 1):.3f}")
print(f"70B on 10T: {estimate_performance(70, 10):.3f}")
```

### Emergent Abilities

Some capabilities appear suddenly at scale:

```
Model Size:  1B    10B    100B   175B+
              |      |       |      |
Arithmetic:  poor   poor    OK     good
Code:        poor   OK      good   good
Reasoning:   poor   poor    OK     good
```

These "emergent abilities" may be:
- Real phase transitions in model capability
- Artifacts of how we measure (threshold effects)
- Gradual improvements that cross usefulness thresholds

## Compute Requirements

### Training Cost

$$
\text{FLOPs} \approx 6 \times N \times D
$$

For a 70B model on 2T tokens:
$$
6 \times 70 \times 10^9 \times 2 \times 10^{12} = 8.4 \times 10^{23} \text{ FLOPs}
$$

On A100 GPUs (~3×10¹⁵ FLOPS):
$$
\frac{8.4 \times 10^{23}}{3 \times 10^{15}} \approx 280 \text{ million GPU-seconds} \approx 3,200 \text{ GPU-years}
$$

### Training Infrastructure

| Model | GPUs | Training Time | Estimated Cost |
|-------|------|---------------|----------------|
| GPT-3 175B | 10,000 V100 | ~1 month | ~$4.6M |
| LLaMA 70B | 2,048 A100 | ~3 months | ~$2M |
| GPT-4 | ~25,000 A100 | ~3 months | ~$100M (est.) |

### Parallelism Strategies

Training requires multiple forms of parallelism:

```
Data Parallelism:
GPU 1: batch[0:8]    → gradients → average → update
GPU 2: batch[8:16]   → gradients →
...

Model Parallelism:
GPU 1: layers 0-11
GPU 2: layers 12-23
...

Pipeline Parallelism:
GPU 1: micro-batch 1 → GPU 2 → GPU 3 → GPU 4
        micro-batch 2 → GPU 2 → GPU 3
                       micro-batch 3 → GPU 3
```

## Training Dynamics

### Loss Curve

```
Loss
  |
  |  \
  |   \
  |    \___
  |        \___
  |            \_____ plateau
  +-----------------> Tokens seen
```

### Training Instabilities

Large models are prone to:

**Loss spikes:**
```python
def detect_loss_spike(losses, window=100, threshold=3.0):
    """Detect sudden loss increases."""
    if len(losses) < window:
        return False
    recent_mean = np.mean(losses[-window:-1])
    recent_std = np.std(losses[-window:-1])
    return losses[-1] > recent_mean + threshold * recent_std
```

**Solutions:**
- Gradient clipping
- Learning rate warmup
- Smaller learning rates
- Checkpointing for rollback

### Curriculum and Data Mixing

Some training schedules vary data mixture over time:

```python
def get_data_mixture(step, total_steps):
    """Vary data sources during training."""
    progress = step / total_steps

    if progress < 0.3:
        # Early: more diverse, noisy data is OK
        return {"web": 0.7, "books": 0.15, "code": 0.15}
    elif progress < 0.7:
        # Middle: balanced
        return {"web": 0.5, "books": 0.25, "code": 0.25}
    else:
        # Late: higher quality, focused
        return {"web": 0.3, "books": 0.35, "code": 0.35}
```

## Summary

| Concept | Description |
|---------|-------------|
| Next-token prediction | Core objective—predict what comes next |
| Tokenization | BPE/SentencePiece for subword units |
| Training data | Trillions of tokens from web, books, code |
| Scaling laws | Performance predictable from size + data |
| Compute | Thousands of GPUs, months of training |

**The essential insight:** Pretraining works because next-token prediction is a universal task that requires learning everything—language, facts, reasoning, style. Scale both model and data, and capabilities emerge. The simplicity of the objective belies the complexity of what must be learned to minimize it.

**Next:** [Fine-tuning](fine-tuning.md) covers adapting pretrained models to specific tasks.
