# Recurrent Neural Networks (RNNs)

```math
\boxed{h_t = \tanh(W_h h_{t-1} + W_x x_t + b)}
```

**Recurrent Neural Networks** process sequences by maintaining hidden state that passes information from one step to the next. Before transformers, RNNs were the foundation of sequence modeling—language models, machine translation, speech recognition. Understanding RNNs illuminates why transformers were designed the way they were.

Prerequisites: [multilayer networks](../neural-networks/multilayer-networks.md), [backpropagation](../neural-networks/backpropagation.md). Code: `numpy`.

---

## The Sequence Modeling Problem

### Why Sequences Are Different

Standard feedforward networks:
- Fixed input size
- No notion of order
- Each input independent

Sequences require:
- Variable length handling
- Order matters (a b c ≠ c b a)
- Dependencies across positions

```
Feedforward:  x → [NN] → y   (one shot)

Recurrent:    x₁ → x₂ → x₃ → ...
               ↓     ↓     ↓
              h₁ → h₂ → h₃ → ...
               ↓     ↓     ↓
              y₁    y₂    y₃
```

### Applications

| Task | Input | Output |
|------|-------|--------|
| Language modeling | Words 1..t | Word t+1 |
| Sentiment | Review text | Positive/negative |
| Translation | Source sentence | Target sentence |
| Speech recognition | Audio frames | Text |
| Time series | Observations | Forecast |

## Basic RNN

### The Core Idea

Maintain a hidden state $h_t$ that summarizes the sequence so far:

```math
h_t = f(h_{t-1}, x_t)
```

The same function $f$ (same weights) applies at every time step.

### Equations

```math
h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h)
```
```math
y_t = W_y h_t + b_y
```

```python
import numpy as np

class RNNCell:
    def __init__(self, input_size, hidden_size):
        # Xavier initialization
        self.Wx = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
        self.Wh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / (2 * hidden_size))
        self.b = np.zeros(hidden_size)

    def forward(self, x, h_prev):
        """
        Args:
            x: Input at current step [batch_size, input_size]
            h_prev: Previous hidden state [batch_size, hidden_size]

        Returns:
            h_next: New hidden state [batch_size, hidden_size]
        """
        self.x = x
        self.h_prev = h_prev

        # Compute new hidden state
        self.z = x @ self.Wx + h_prev @ self.Wh + self.b
        h_next = np.tanh(self.z)

        self.h_next = h_next
        return h_next

    def backward(self, dh_next):
        """
        Args:
            dh_next: Gradient from future [batch_size, hidden_size]

        Returns:
            dx: Gradient for input
            dh_prev: Gradient for previous hidden state
        """
        # Through tanh
        dz = dh_next * (1 - self.h_next ** 2)

        # Gradients for weights
        self.dWx = self.x.T @ dz
        self.dWh = self.h_prev.T @ dz
        self.db = np.sum(dz, axis=0)

        # Gradients for inputs
        dx = dz @ self.Wx.T
        dh_prev = dz @ self.Wh.T

        return dx, dh_prev
```

### Unrolling Through Time

To process a sequence, we "unroll" the RNN:

```
x₁      x₂      x₃      x₄
 ↓       ↓       ↓       ↓
[RNN] → [RNN] → [RNN] → [RNN]
 ↓       ↓       ↓       ↓
h₁      h₂      h₃      h₄

Same weights at each step!
```

```python
class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.cell = RNNCell(input_size, hidden_size)
        self.Wy = np.random.randn(hidden_size, output_size) * 0.01
        self.by = np.zeros(output_size)
        self.hidden_size = hidden_size

    def forward(self, x_seq, h0=None):
        """
        Args:
            x_seq: Input sequence [batch_size, seq_len, input_size]
            h0: Initial hidden state (optional)

        Returns:
            outputs: Output at each step [batch_size, seq_len, output_size]
            h_final: Final hidden state
        """
        batch_size, seq_len, _ = x_seq.shape

        if h0 is None:
            h = np.zeros((batch_size, self.hidden_size))
        else:
            h = h0

        self.hidden_states = [h]
        self.inputs = []
        outputs = []

        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            self.inputs.append(x_t)

            h = self.cell.forward(x_t, h)
            self.hidden_states.append(h)

            y_t = h @ self.Wy + self.by
            outputs.append(y_t)

        outputs = np.stack(outputs, axis=1)
        return outputs, h
```

**What this means:** An RNN is a feedforward network that shares weights across time steps. The hidden state acts as memory, carrying information forward through the sequence.

## Backpropagation Through Time (BPTT)

### The Algorithm

Gradients flow backwards through the unrolled network:

```
         ← ← ← ← ← ← ← gradients
x₁      x₂      x₃      x₄
 ↓       ↓       ↓       ↓
[RNN] → [RNN] → [RNN] → [RNN]
 ↓       ↓       ↓       ↓
loss₁   loss₂   loss₃   loss₄
```

```python
def bptt(self, x_seq, targets):
    """
    Backpropagation through time.

    Args:
        x_seq: Input sequence
        targets: Target at each step
    """
    batch_size, seq_len, _ = x_seq.shape

    # Forward pass
    outputs, _ = self.forward(x_seq)

    # Initialize gradients
    dWx = np.zeros_like(self.cell.Wx)
    dWh = np.zeros_like(self.cell.Wh)
    db = np.zeros_like(self.cell.b)
    dWy = np.zeros_like(self.Wy)
    dby = np.zeros_like(self.by)

    # Backward pass
    dh_next = np.zeros((batch_size, self.hidden_size))

    for t in reversed(range(seq_len)):
        # Gradient from output
        dy = outputs[:, t, :] - targets[:, t, :]  # Assuming softmax CE loss
        dWy += self.hidden_states[t + 1].T @ dy
        dby += np.sum(dy, axis=0)

        # Gradient into hidden state
        dh = dy @ self.Wy.T + dh_next

        # Through RNN cell
        self.cell.h_next = self.hidden_states[t + 1]
        self.cell.h_prev = self.hidden_states[t]
        self.cell.x = self.inputs[t]
        self.cell.z = np.arctanh(np.clip(self.hidden_states[t + 1], -0.999, 0.999))

        dx, dh_next = self.cell.backward(dh)

        # Accumulate gradients (same weights at each step)
        dWx += self.cell.dWx
        dWh += self.cell.dWh
        db += self.cell.db

    return dWx, dWh, db, dWy, dby
```

### Truncated BPTT

For long sequences, full BPTT is expensive. Truncate to fixed windows:

```python
def truncated_bptt(self, x_seq, targets, bptt_len=20):
    """
    Truncated backpropagation through time.

    Only backpropagate through bptt_len steps at a time.
    """
    seq_len = x_seq.shape[1]
    h = np.zeros((x_seq.shape[0], self.hidden_size))

    total_loss = 0
    for start in range(0, seq_len, bptt_len):
        end = min(start + bptt_len, seq_len)

        # Forward for this chunk (keeping hidden state from previous chunk)
        chunk_outputs = []
        for t in range(start, end):
            h = self.cell.forward(x_seq[:, t, :], h)
            y = h @ self.Wy + self.by
            chunk_outputs.append(y)

        # Backward for this chunk only
        # (gradients don't flow back beyond 'start')
        self._backward_chunk(chunk_outputs, targets[:, start:end, :])

        # Detach hidden state (no gradient flows to previous chunk)
        h = h.copy()  # Stop gradient

    return total_loss
```

**What this means:** Truncated BPTT trades off memory/compute for gradient quality. The hidden state still carries information forward, but gradients only flow back a fixed number of steps.

## The Vanishing Gradient Problem

### The Problem

Gradients multiply through time:

```math
\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=2}^{T} W_h^T \cdot \text{diag}(1 - h_t^2)
```

If $|W_h| < 1$ or $|\tanh'| < 1$: gradients shrink exponentially.
If $|W_h| > 1$: gradients explode.

```python
def demonstrate_vanishing_gradient():
    """Show gradient decay over time steps."""
    hidden_size = 100
    seq_len = 50

    # Typical weight magnitude
    W = np.random.randn(hidden_size, hidden_size) * 0.1

    # Simulate gradient propagation
    gradient = np.ones(hidden_size)
    gradient_norms = [np.linalg.norm(gradient)]

    for t in range(seq_len):
        # Multiply by weight and tanh derivative (average ~0.5)
        gradient = (W.T @ gradient) * 0.5
        gradient_norms.append(np.linalg.norm(gradient))

    print(f"Initial gradient norm: {gradient_norms[0]:.4f}")
    print(f"After 50 steps: {gradient_norms[-1]:.10f}")
    # Typically shrinks to near zero!
```

### Consequences

```
Gradient magnitude over time:

|grad|
  |    ***
  |       ***
  |          ***
  |             ***
  |                *** → vanishing
  +------------------→ t
       early    late
       steps    steps
```

- Early parts of sequence get tiny gradients
- Long-range dependencies are hard to learn
- RNN "forgets" early information

### Partial Solutions

1. **Careful initialization:** Orthogonal weight matrices
2. **Gradient clipping:** Prevent explosion
3. **Better architectures:** LSTMs, GRUs (see [LSTMs](lstms.md))

```python
def orthogonal_init(shape):
    """Initialize with orthogonal matrix (helps with gradients)."""
    a = np.random.randn(*shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    return u if u.shape == shape else v

def clip_gradients(grads, max_norm):
    """Clip gradients to prevent explosion."""
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        grads = [g * scale for g in grads]
    return grads
```

## RNN Variants

### Bidirectional RNN

Process sequence in both directions:

```
Forward:   → → → →
           h₁ h₂ h₃ h₄

Backward:  ← ← ← ←
           h₁ h₂ h₃ h₄

Output:    [h→; h←] at each position
```

```python
class BidirectionalRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.forward_rnn = RNN(input_size, hidden_size, hidden_size)
        self.backward_rnn = RNN(input_size, hidden_size, hidden_size)
        self.Wy = np.random.randn(2 * hidden_size, output_size) * 0.01
        self.by = np.zeros(output_size)

    def forward(self, x_seq):
        # Forward direction
        h_forward, _ = self.forward_rnn.forward(x_seq)

        # Backward direction (reverse input)
        x_reversed = x_seq[:, ::-1, :]
        h_backward, _ = self.backward_rnn.forward(x_reversed)
        h_backward = h_backward[:, ::-1, :]  # Reverse output

        # Concatenate
        h_combined = np.concatenate([h_forward, h_backward], axis=-1)

        # Output
        outputs = h_combined @ self.Wy + self.by
        return outputs
```

**What this means:** Bidirectional RNNs see context from both past and future, useful for tasks where the whole sequence is available (classification, NER) but not for generation.

### Deep RNN (Stacked)

Stack multiple RNN layers:

```
x → [RNN₁] → [RNN₂] → [RNN₃] → y
```

```python
class DeepRNN:
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.layers = []

        # First layer: input_size → hidden_size
        self.layers.append(RNNCell(input_size, hidden_size))

        # Subsequent layers: hidden_size → hidden_size
        for _ in range(num_layers - 1):
            self.layers.append(RNNCell(hidden_size, hidden_size))

        self.Wy = np.random.randn(hidden_size, output_size) * 0.01
        self.by = np.zeros(output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x_seq, h0=None):
        batch_size, seq_len, _ = x_seq.shape

        if h0 is None:
            h = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        else:
            h = h0

        outputs = []
        for t in range(seq_len):
            layer_input = x_seq[:, t, :]

            for l, layer in enumerate(self.layers):
                h[l] = layer.forward(layer_input, h[l])
                layer_input = h[l]

            y_t = h[-1] @ self.Wy + self.by
            outputs.append(y_t)

        return np.stack(outputs, axis=1), h
```

## Language Modeling Example

### Character-Level RNN

```python
class CharRNN:
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Embedding (one-hot to dense)
        self.embedding = np.random.randn(vocab_size, hidden_size) * 0.01

        # RNN
        self.rnn = RNNCell(hidden_size, hidden_size)

        # Output projection
        self.Wy = np.random.randn(hidden_size, vocab_size) * 0.01
        self.by = np.zeros(vocab_size)

    def forward(self, char_indices, h=None):
        """
        Args:
            char_indices: [batch_size, seq_len] character indices
            h: Optional initial hidden state
        """
        batch_size, seq_len = char_indices.shape

        if h is None:
            h = np.zeros((batch_size, self.hidden_size))

        outputs = []
        for t in range(seq_len):
            # Embed
            x = self.embedding[char_indices[:, t]]

            # RNN step
            h = self.rnn.forward(x, h)

            # Project to vocabulary
            logits = h @ self.Wy + self.by
            outputs.append(logits)

        return np.stack(outputs, axis=1), h

    def sample(self, seed_char, length, temperature=1.0):
        """Generate text starting from seed character."""
        h = np.zeros((1, self.hidden_size))
        current_char = seed_char
        generated = [current_char]

        for _ in range(length):
            x = self.embedding[current_char:current_char+1]
            h = self.rnn.forward(x, h)
            logits = h @ self.Wy + self.by

            # Sample from distribution
            probs = softmax(logits[0] / temperature)
            current_char = np.random.choice(self.vocab_size, p=probs)
            generated.append(current_char)

        return generated

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
```

### Training Loop

```python
def train_char_rnn(model, text, epochs=100, seq_len=50, lr=0.001):
    """Train character-level language model."""
    # Build vocabulary
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}

    # Convert text to indices
    data = np.array([char_to_idx[c] for c in text])

    # Training
    for epoch in range(epochs):
        total_loss = 0
        h = None  # Carry hidden state across chunks

        for start in range(0, len(data) - seq_len - 1, seq_len):
            # Get batch
            x = data[start:start + seq_len].reshape(1, -1)
            y = data[start + 1:start + seq_len + 1].reshape(1, -1)

            # Forward
            logits, h = model.forward(x, h)
            h = h.copy()  # Detach for truncated BPTT

            # Cross-entropy loss
            probs = softmax_batch(logits)
            loss = -np.mean(np.log(probs[0, range(seq_len), y[0]] + 1e-8))
            total_loss += loss

            # Backward and update (simplified)
            # ... gradient computation and update ...

        if epoch % 10 == 0:
            sample = model.sample(0, 100)
            text_sample = ''.join(idx_to_char[i] for i in sample)
            print(f"Epoch {epoch}: {text_sample[:50]}...")

def softmax_batch(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

## Summary

| Concept | Description |
|---------|-------------|
| Hidden state | Memory that carries information across time |
| Weight sharing | Same parameters at every time step |
| BPTT | Backpropagation through the unrolled network |
| Vanishing gradient | Long-range dependencies hard to learn |
| Bidirectional | See context from both directions |
| Deep RNN | Stack layers for more capacity |

**The essential insight:** RNNs process sequences by maintaining state that evolves through time. The same weights at each step make them efficient and enable variable-length processing. However, the vanishing gradient problem limits their ability to learn long-range dependencies—a problem addressed by LSTMs and ultimately solved by transformers.

**Next:** [LSTMs](lstms.md) add gating mechanisms to control information flow and solve the vanishing gradient problem.
