# Sequence-to-Sequence

$$
\boxed{p(y_1, \ldots, y_T | x_1, \ldots, x_S) = \prod_{t=1}^{T} p(y_t | y_{<t}, c)}
$$

**Sequence-to-sequence** (seq2seq) models transform one sequence into another. An encoder reads the input; a decoder generates the output. This architecture enabled breakthroughs in machine translation (2014) and introduced concepts—encoder-decoder structure, attention mechanisms—that evolved into modern transformers.

Prerequisites: [LSTMs](lstms.md), [activation functions](../neural-networks/activation-functions.md). Code: `numpy`.

---

## The Sequence-to-Sequence Problem

### Variable-Length In, Variable-Length Out

Many tasks need to map sequences of different lengths:

| Task | Input | Output |
|------|-------|--------|
| Translation | "How are you?" | "Comment allez-vous?" |
| Summarization | Long article | Short summary |
| Dialogue | User utterance | Bot response |
| Code generation | Natural language | Code |

Standard RNNs output one element per input element—not suitable when lengths differ.

### The Encoder-Decoder Solution

```
Input: The cat sat
        ↓   ↓   ↓
      [Encoder LSTM]
              ↓
           context
              ↓
      [Decoder LSTM]
        ↓   ↓   ↓   ↓
Output: Le chat assis <EOS>
```

1. **Encoder:** Read entire input, compress into fixed-size context
2. **Context:** Summary of input meaning
3. **Decoder:** Generate output token by token, conditioned on context

## Basic Seq2Seq

### Encoder

The encoder reads the input sequence and produces a context vector:

```python
import numpy as np

class Encoder:
    def __init__(self, vocab_size, embed_dim, hidden_size):
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.01
        self.lstm = LSTMCell(embed_dim, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, src_tokens):
        """
        Args:
            src_tokens: [batch_size, src_len] source token indices

        Returns:
            context: Final hidden state (h, c)
        """
        batch_size, src_len = src_tokens.shape

        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))

        for t in range(src_len):
            x = self.embedding[src_tokens[:, t]]
            h, c = self.lstm.forward(x, h, c)

        return (h, c)  # Context = final state
```

### Decoder

The decoder generates output tokens one at a time:

```python
class Decoder:
    def __init__(self, vocab_size, embed_dim, hidden_size):
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.01
        self.lstm = LSTMCell(embed_dim, hidden_size)
        self.output_proj = np.random.randn(hidden_size, vocab_size) * 0.01
        self.vocab_size = vocab_size

    def forward(self, tgt_tokens, initial_state):
        """
        Args:
            tgt_tokens: [batch_size, tgt_len] target token indices
            initial_state: (h, c) from encoder

        Returns:
            logits: [batch_size, tgt_len, vocab_size]
        """
        batch_size, tgt_len = tgt_tokens.shape
        h, c = initial_state

        logits = []
        for t in range(tgt_len):
            x = self.embedding[tgt_tokens[:, t]]
            h, c = self.lstm.forward(x, h, c)
            logit = h @ self.output_proj
            logits.append(logit)

        return np.stack(logits, axis=1)

    def generate(self, initial_state, max_len, start_token, end_token):
        """Generate sequence autoregressively."""
        batch_size = initial_state[0].shape[0]
        h, c = initial_state

        generated = [np.full(batch_size, start_token)]
        current_token = generated[0]

        for _ in range(max_len):
            x = self.embedding[current_token]
            h, c = self.lstm.forward(x, h, c)
            logit = h @ self.output_proj

            # Greedy decoding
            current_token = np.argmax(logit, axis=-1)
            generated.append(current_token)

            # Stop if all sequences have ended
            if np.all(current_token == end_token):
                break

        return np.stack(generated, axis=1)
```

### Complete Seq2Seq Model

```python
class Seq2Seq:
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_size):
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_size)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_size)

    def forward(self, src_tokens, tgt_tokens):
        """
        Training forward pass (teacher forcing).

        Args:
            src_tokens: [batch, src_len] source sequence
            tgt_tokens: [batch, tgt_len] target sequence (shifted right)

        Returns:
            logits: [batch, tgt_len, vocab_size]
        """
        context = self.encoder.forward(src_tokens)
        logits = self.decoder.forward(tgt_tokens, context)
        return logits

    def translate(self, src_tokens, max_len=50, start_token=1, end_token=2):
        """Inference: generate translation."""
        context = self.encoder.forward(src_tokens)
        return self.decoder.generate(context, max_len, start_token, end_token)
```

**What this means:** The encoder compresses variable-length input into a fixed context vector. The decoder then generates output conditioned on this context. During training, we use "teacher forcing"—feeding the correct previous token. During inference, we feed the model's own predictions.

## The Information Bottleneck

### The Problem

All input information must flow through a single context vector:

```
Long input sequence...........................
                     ↓
              [single vector]  ← Bottleneck!
                     ↓
Output sequence...
```

For long sequences, this fixed-size vector can't retain all relevant information.

### Evidence of the Problem

```python
def translation_quality_by_length():
    """BLEU score typically drops for longer sentences."""
    # Empirical observation (conceptual):
    lengths = [5, 10, 20, 30, 40, 50]
    bleu_basic_seq2seq = [35, 30, 25, 18, 12, 8]
    bleu_with_attention = [35, 33, 32, 30, 28, 26]

    print("Sentence Length | Basic Seq2Seq | With Attention")
    for l, b1, b2 in zip(lengths, bleu_basic_seq2seq, bleu_with_attention):
        print(f"      {l:2d}        |     {b1:2d}        |     {b2:2d}")
```

## Attention Mechanism

### The Key Insight

Instead of one context vector, let the decoder **look at all encoder states** and focus on relevant parts for each output token:

```
Encoder states:  h₁   h₂   h₃   h₄   h₅
                  ↘   ↓   ↙ ↙   ↙
                   [attention]  ← Different weights for each decoder step
                       ↓
                   context_t
                       ↓
                  [decoder]
                       ↓
                     y_t
```

### Attention Equations

**Attention scores** — how relevant is encoder state $j$ for decoder step $t$:
$$
e_{t,j} = \text{score}(s_{t-1}, h_j)
$$

**Attention weights** — normalized scores:
$$
\alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_k \exp(e_{t,k})}
$$

**Context vector** — weighted sum of encoder states:
$$
c_t = \sum_j \alpha_{t,j} h_j
$$

### Scoring Functions

| Name | Formula | Notes |
|------|---------|-------|
| Dot product | $s^T h$ | Simple, fast |
| Bilinear | $s^T W h$ | Learnable |
| Additive (Bahdanau) | $v^T \tanh(W_s s + W_h h)$ | Original, expressive |

### Implementation

```python
class Attention:
    def __init__(self, hidden_size):
        # Additive attention (Bahdanau style)
        self.Ws = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.v = np.random.randn(hidden_size) * 0.01

    def forward(self, decoder_state, encoder_outputs):
        """
        Args:
            decoder_state: [batch_size, hidden_size]
            encoder_outputs: [batch_size, src_len, hidden_size]

        Returns:
            context: [batch_size, hidden_size]
            attention_weights: [batch_size, src_len]
        """
        batch_size, src_len, hidden_size = encoder_outputs.shape

        # Project decoder state
        s_proj = decoder_state @ self.Ws  # [batch, hidden]
        s_proj = s_proj[:, np.newaxis, :]  # [batch, 1, hidden]

        # Project encoder outputs
        h_proj = encoder_outputs @ self.Wh  # [batch, src_len, hidden]

        # Compute scores
        energy = np.tanh(s_proj + h_proj)  # [batch, src_len, hidden]
        scores = energy @ self.v  # [batch, src_len]

        # Softmax to get weights
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

        # Weighted sum of encoder outputs
        context = np.sum(attention_weights[:, :, np.newaxis] * encoder_outputs, axis=1)

        return context, attention_weights


class AttentionDecoder:
    def __init__(self, vocab_size, embed_dim, hidden_size):
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.01
        self.attention = Attention(hidden_size)

        # LSTM takes concatenated [embedding, context]
        self.lstm = LSTMCell(embed_dim + hidden_size, hidden_size)

        self.output_proj = np.random.randn(hidden_size, vocab_size) * 0.01
        self.hidden_size = hidden_size

    def forward(self, tgt_tokens, initial_state, encoder_outputs):
        """
        Decode with attention over encoder outputs.
        """
        batch_size, tgt_len = tgt_tokens.shape
        h, c = initial_state

        logits = []
        attention_history = []

        for t in range(tgt_len):
            # Get attention context
            context, attn_weights = self.attention.forward(h, encoder_outputs)
            attention_history.append(attn_weights)

            # Embed current token and concatenate with context
            x = self.embedding[tgt_tokens[:, t]]
            x_combined = np.concatenate([x, context], axis=-1)

            # LSTM step
            h, c = self.lstm.forward(x_combined, h, c)

            # Output projection
            logit = h @ self.output_proj
            logits.append(logit)

        return np.stack(logits, axis=1), attention_history
```

**What this means:** Instead of forcing all information through one vector, attention lets the decoder dynamically query the encoder. For "Le chat", it might focus on "The cat". For "assis", it focuses on "sat". This is the precursor to self-attention in transformers.

## Training

### Teacher Forcing

During training, feed correct previous tokens (not predictions):

```python
def train_step(model, src, tgt, optimizer):
    """
    Args:
        src: Source tokens [batch, src_len]
        tgt: Target tokens [batch, tgt_len]
    """
    # Shift target: input is tgt[:-1], label is tgt[1:]
    tgt_input = tgt[:, :-1]
    tgt_label = tgt[:, 1:]

    # Forward
    logits = model.forward(src, tgt_input)

    # Cross-entropy loss
    loss = cross_entropy_loss(logits, tgt_label)

    # Backward and update
    grads = model.backward(tgt_label)
    optimizer.step(grads)

    return loss
```

### Scheduled Sampling

Mix teacher forcing with model predictions to reduce train/inference gap:

```python
def train_step_scheduled_sampling(model, src, tgt, teacher_forcing_ratio=0.5):
    """Randomly use model predictions instead of ground truth."""
    batch_size, tgt_len = tgt.shape

    h, c = model.encoder.forward(src)
    enc_outputs = model.encoder.get_outputs()

    logits = []
    current_token = tgt[:, 0]  # Start token

    for t in range(1, tgt_len):
        # Decide whether to use teacher forcing
        use_teacher_forcing = np.random.random() < teacher_forcing_ratio

        x = model.decoder.embedding[current_token]
        context, _ = model.decoder.attention.forward(h, enc_outputs)
        x_combined = np.concatenate([x, context], axis=-1)
        h, c = model.decoder.lstm.forward(x_combined, h, c)
        logit = h @ model.decoder.output_proj
        logits.append(logit)

        if use_teacher_forcing:
            current_token = tgt[:, t]
        else:
            current_token = np.argmax(logit, axis=-1)

    return np.stack(logits, axis=1)
```

## Decoding Strategies

### Greedy Decoding

Always pick the highest probability token:

```python
def greedy_decode(model, src, max_len, start_token, end_token):
    context = model.encoder.forward(src)
    enc_outputs = model.encoder.get_outputs()
    h, c = context

    tokens = [np.array([start_token])]

    for _ in range(max_len):
        x = model.decoder.embedding[tokens[-1]]
        ctx, _ = model.decoder.attention.forward(h, enc_outputs)
        x_combined = np.concatenate([x, ctx], axis=-1)
        h, c = model.decoder.lstm.forward(x_combined, h, c)
        logit = h @ model.decoder.output_proj

        next_token = np.argmax(logit, axis=-1)
        tokens.append(next_token)

        if next_token[0] == end_token:
            break

    return np.concatenate(tokens)
```

### Beam Search

Maintain top-k candidates at each step:

```python
def beam_search(model, src, max_len, beam_size=5, start_token=1, end_token=2):
    """
    Beam search decoding.

    Returns:
        best_sequence: Token sequence with highest score
    """
    context = model.encoder.forward(src)
    enc_outputs = model.encoder.get_outputs()

    # Each beam: (score, tokens, hidden_state)
    beams = [(0.0, [start_token], context)]
    completed = []

    for _ in range(max_len):
        candidates = []

        for score, tokens, (h, c) in beams:
            if tokens[-1] == end_token:
                completed.append((score, tokens))
                continue

            # Get next token probabilities
            x = model.decoder.embedding[np.array([tokens[-1]])]
            ctx, _ = model.decoder.attention.forward(h, enc_outputs)
            x_combined = np.concatenate([x, ctx], axis=-1)
            h_new, c_new = model.decoder.lstm.forward(x_combined, h, c)
            logit = h_new @ model.decoder.output_proj
            log_probs = log_softmax(logit[0])

            # Expand beam with top-k tokens
            top_k = np.argsort(log_probs)[-beam_size:]
            for token in top_k:
                new_score = score + log_probs[token]
                new_tokens = tokens + [token]
                candidates.append((new_score, new_tokens, (h_new, c_new)))

        # Keep top beam_size candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

        if not beams:
            break

    # Return best completed sequence
    completed.extend([(s, t) for s, t, _ in beams])
    completed.sort(key=lambda x: x[0] / len(x[1]), reverse=True)  # Length normalize
    return completed[0][1] if completed else [start_token]

def log_softmax(x):
    x_max = np.max(x)
    return x - x_max - np.log(np.sum(np.exp(x - x_max)))
```

**What this means:** Greedy is fast but can miss better sequences. Beam search explores multiple paths, typically finding better translations at the cost of more computation.

## Visualizing Attention

```python
def plot_attention(attention_weights, src_tokens, tgt_tokens, src_vocab, tgt_vocab):
    """
    Visualize attention alignment between source and target.

    Args:
        attention_weights: [tgt_len, src_len] attention matrix
        src_tokens, tgt_tokens: Token indices
        src_vocab, tgt_vocab: Index to word mappings
    """
    import matplotlib.pyplot as plt

    src_words = [src_vocab[t] for t in src_tokens]
    tgt_words = [tgt_vocab[t] for t in tgt_tokens]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(attention_weights, cmap='Blues')

    ax.set_xticks(range(len(src_words)))
    ax.set_yticks(range(len(tgt_words)))
    ax.set_xticklabels(src_words, rotation=45)
    ax.set_yticklabels(tgt_words)

    ax.set_xlabel('Source')
    ax.set_ylabel('Target')
    ax.set_title('Attention Alignment')

    plt.tight_layout()
    plt.show()
```

Example attention pattern for translation:

```
         The   cat   sat   on   the   mat
Le       0.8   0.1   0.0   0.0  0.1   0.0
chat     0.1   0.8   0.1   0.0  0.0   0.0
était    0.0   0.0   0.6   0.0  0.0   0.0
assis    0.0   0.2   0.7   0.1  0.0   0.0
sur      0.0   0.0   0.0   0.9  0.1   0.0
le       0.0   0.0   0.0   0.1  0.8   0.1
tapis    0.0   0.0   0.0   0.0  0.0   0.9
```

## Complete Example

```python
class TranslationModel:
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256, hidden_size=512):
        # Encoder
        self.src_embedding = np.random.randn(src_vocab_size, embed_dim) * 0.01
        self.encoder = LSTMCell(embed_dim, hidden_size)

        # Decoder with attention
        self.tgt_embedding = np.random.randn(tgt_vocab_size, embed_dim) * 0.01
        self.attention = Attention(hidden_size)
        self.decoder = LSTMCell(embed_dim + hidden_size, hidden_size)
        self.output = np.random.randn(hidden_size, tgt_vocab_size) * 0.01

        self.hidden_size = hidden_size

    def encode(self, src):
        batch_size, src_len = src.shape
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))

        encoder_outputs = []
        for t in range(src_len):
            x = self.src_embedding[src[:, t]]
            h, c = self.encoder.forward(x, h, c)
            encoder_outputs.append(h)

        self.encoder_outputs = np.stack(encoder_outputs, axis=1)
        return h, c

    def decode_step(self, token, h, c):
        x = self.tgt_embedding[token]
        context, attn = self.attention.forward(h, self.encoder_outputs)
        x_combined = np.concatenate([x, context], axis=-1)
        h, c = self.decoder.forward(x_combined, h, c)
        logits = h @ self.output
        return logits, h, c, attn

    def translate(self, src, max_len=50, start=1, end=2):
        h, c = self.encode(src)
        token = np.array([start])
        result = [start]

        for _ in range(max_len):
            logits, h, c, _ = self.decode_step(token, h, c)
            token = np.argmax(logits, axis=-1)
            result.append(token[0])
            if token[0] == end:
                break

        return result
```

## Summary

| Component | Role |
|-----------|------|
| Encoder | Compress input to context |
| Decoder | Generate output from context |
| Context vector | Information bottleneck (basic seq2seq) |
| Attention | Dynamic focus on relevant encoder states |
| Teacher forcing | Train with correct previous tokens |
| Beam search | Find high-probability sequences |

**The essential insight:** Seq2seq with attention was the architecture that demonstrated neural networks could match (and exceed) traditional machine translation. The attention mechanism—letting the decoder look back at encoder states—was the key innovation. This same idea, generalized to **self-attention** (where a sequence attends to itself), became the foundation of transformers.

**Historical context:** The 2014 seq2seq paper showed neural translation was possible. The 2015 attention paper made it practical. By 2017, transformers replaced LSTMs entirely with attention, achieving better parallelization and longer-range dependencies.

**Next:** [Multi-Head Attention](../transformers/multi-head-attention.md) extends attention to the transformer architecture.
