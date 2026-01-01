# Long Short-Term Memory (LSTM)

```math
\boxed{c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t}
```

**LSTMs** solve the vanishing gradient problem that plagues basic RNNs. The key insight: instead of transforming state through nonlinearities at each step, let information flow through a **cell state** with minimal modification. Gates learn when to add, remove, or output information from this memory.

Prerequisites: [RNNs](rnns.md), [activation functions](../neural-networks/activation-functions.md). Code: `numpy`.

---

## The Problem LSTMs Solve

### Vanishing Gradients Revisited

In basic RNNs:
```math
\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} W_h^T \cdot \text{diag}(\tanh'(z_t))
```

Each multiplication shrinks gradients because $|\tanh'| \leq 1$ and $|W_h|$ is typically small.

### The LSTM Solution

Create a **highway** for gradients:

```
Basic RNN:  h₁ →[×W]→ h₂ →[×W]→ h₃ →[×W]→ h₄
            Gradients multiply through W each step

LSTM:       c₁ ----→ c₂ ----→ c₃ ----→ c₄
              ↑+       ↑+       ↑+
            Gates control what's added
            Gradient flows straight through!
```

The cell state $c_t$ acts as a conveyor belt—information can flow unchanged, and gates control modifications.

## LSTM Architecture

### The Three Gates

| Gate | Symbol | Purpose |
|------|--------|---------|
| Forget | $f_t$ | What to discard from cell state |
| Input | $i_t$ | What new information to store |
| Output | $o_t$ | What to output from cell state |

### Equations

**Forget gate** — decide what to forget:
```math
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
```

**Input gate** — decide what to add:
```math
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
```

**Candidate values** — propose new information:
```math
\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)
```

**Cell state update** — the key equation:
```math
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
```

**Output gate** — decide what to output:
```math
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
```

**Hidden state** — filtered cell state:
```math
h_t = o_t \odot \tanh(c_t)
```

### Implementation

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size

        # Combined weights for efficiency: [input, forget, candidate, output]
        combined_size = input_size + hidden_size

        # Xavier initialization
        scale = np.sqrt(2 / (combined_size + hidden_size))
        self.W = np.random.randn(combined_size, 4 * hidden_size) * scale
        self.b = np.zeros(4 * hidden_size)

        # Initialize forget gate bias to 1 (remember by default)
        self.b[hidden_size:2*hidden_size] = 1.0

    def forward(self, x, h_prev, c_prev):
        """
        Args:
            x: Input [batch_size, input_size]
            h_prev: Previous hidden state [batch_size, hidden_size]
            c_prev: Previous cell state [batch_size, hidden_size]

        Returns:
            h_next: New hidden state
            c_next: New cell state
        """
        batch_size = x.shape[0]
        H = self.hidden_size

        # Concatenate input and previous hidden state
        combined = np.concatenate([x, h_prev], axis=1)

        # Compute all gates at once
        gates = combined @ self.W + self.b

        # Split into individual gates
        i = sigmoid(gates[:, 0:H])           # Input gate
        f = sigmoid(gates[:, H:2*H])         # Forget gate
        c_tilde = np.tanh(gates[:, 2*H:3*H]) # Candidate
        o = sigmoid(gates[:, 3*H:4*H])       # Output gate

        # Cell state update
        c_next = f * c_prev + i * c_tilde

        # Hidden state
        h_next = o * np.tanh(c_next)

        # Cache for backward pass
        self.cache = (x, h_prev, c_prev, combined, i, f, c_tilde, o, c_next)

        return h_next, c_next

    def backward(self, dh_next, dc_next):
        """
        Backpropagate through LSTM cell.

        Args:
            dh_next: Gradient of hidden state [batch_size, hidden_size]
            dc_next: Gradient of cell state [batch_size, hidden_size]

        Returns:
            dx: Gradient for input
            dh_prev: Gradient for previous hidden state
            dc_prev: Gradient for previous cell state
        """
        x, h_prev, c_prev, combined, i, f, c_tilde, o, c_next = self.cache
        H = self.hidden_size

        # Gradient through h = o * tanh(c)
        tanh_c = np.tanh(c_next)
        do = dh_next * tanh_c
        dc = dh_next * o * (1 - tanh_c ** 2) + dc_next

        # Gradients for gates
        di = dc * c_tilde
        df = dc * c_prev
        dc_tilde = dc * i

        # Through activations
        di_raw = di * i * (1 - i)  # sigmoid derivative
        df_raw = df * f * (1 - f)
        dc_tilde_raw = dc_tilde * (1 - c_tilde ** 2)  # tanh derivative
        do_raw = do * o * (1 - o)

        # Combine gate gradients
        dgates = np.concatenate([di_raw, df_raw, dc_tilde_raw, do_raw], axis=1)

        # Gradients for weights and biases
        self.dW = combined.T @ dgates
        self.db = np.sum(dgates, axis=0)

        # Gradients for inputs
        dcombined = dgates @ self.W.T
        dx = dcombined[:, :x.shape[1]]
        dh_prev = dcombined[:, x.shape[1]:]

        # Cell state gradient flows straight back
        dc_prev = dc * f

        return dx, dh_prev, dc_prev
```

**What this means:** The cell state update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ is the key. When $f_t \approx 1$ and $i_t \approx 0$, information passes through unchanged. The gradient $\frac{\partial c_t}{\partial c_{t-1}} = f_t$ can stay near 1, preventing vanishing.

## Understanding the Gates

### Forget Gate

Learns when to discard information:

```python
def forget_gate_examples():
    """Examples of forget gate behavior."""
    # Sentence boundary: forget previous context
    # "The cat sat. The dog ran."
    #              ↑ f ≈ 0 (forget cat context)

    # Pronoun resolution: keep subject
    # "John said he would..."
    #      ↑ f ≈ 1 (keep "John" for "he")
```

### Input Gate

Learns when to write new information:

```python
def input_gate_examples():
    """Examples of input gate behavior."""
    # New subject: store it
    # "Mary went to..."
    #  ↑ i ≈ 1 (store "Mary")

    # Filler words: don't store
    # "The very tall..."
    #      ↑ i ≈ 0 (skip "very")
```

### Output Gate

Learns when to expose cell state:

```python
def output_gate_examples():
    """Examples of output gate behavior."""
    # Time to use stored info
    # "John... he said"
    #         ↑ o ≈ 1 (need "John" for "he")

    # Information not needed yet
    # "In the beginning... [long passage]"
    #                ↑ o ≈ 0 (save "beginning" for later)
```

### Visualizing Gate Activations

```python
def visualize_gates(lstm, sequence):
    """Track gate activations through a sequence."""
    h = np.zeros((1, lstm.hidden_size))
    c = np.zeros((1, lstm.hidden_size))

    forget_gates = []
    input_gates = []
    output_gates = []

    for x in sequence:
        h, c = lstm.forward(x.reshape(1, -1), h, c)
        _, _, _, _, i, f, _, o, _ = lstm.cache

        forget_gates.append(f.mean())
        input_gates.append(i.mean())
        output_gates.append(o.mean())

    return forget_gates, input_gates, output_gates
```

## Complete LSTM Network

```python
class LSTM:
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stack LSTM cells
        self.cells = []
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCell(input_dim, hidden_size))

        # Output projection
        self.Wy = np.random.randn(hidden_size, output_size) * 0.01
        self.by = np.zeros(output_size)

    def forward(self, x_seq, initial_state=None):
        """
        Args:
            x_seq: [batch_size, seq_len, input_size]
            initial_state: Tuple of (h, c) for each layer

        Returns:
            outputs: [batch_size, seq_len, output_size]
            final_state: Tuple of (h, c) for each layer
        """
        batch_size, seq_len, _ = x_seq.shape

        # Initialize states
        if initial_state is None:
            h = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
            c = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        else:
            h, c = initial_state

        outputs = []
        self.all_states = []

        for t in range(seq_len):
            layer_input = x_seq[:, t, :]

            for l, cell in enumerate(self.cells):
                h[l], c[l] = cell.forward(layer_input, h[l], c[l])
                layer_input = h[l]

            self.all_states.append((
                [h_l.copy() for h_l in h],
                [c_l.copy() for c_l in c]
            ))

            # Output at each step
            y = h[-1] @ self.Wy + self.by
            outputs.append(y)

        outputs = np.stack(outputs, axis=1)
        return outputs, (h, c)

    def get_params(self):
        """Get all parameters for optimization."""
        params = []
        for cell in self.cells:
            params.extend([cell.W, cell.b])
        params.extend([self.Wy, self.by])
        return params
```

## GRU: A Simpler Alternative

### Gated Recurrent Unit

GRU combines forget and input gates, and merges cell and hidden state:

```math
z_t = \sigma(W_z [h_{t-1}, x_t])  \quad \text{(update gate)}
```
```math
r_t = \sigma(W_r [h_{t-1}, x_t])  \quad \text{(reset gate)}
```
```math
\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t])  \quad \text{(candidate)}
```
```math
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t  \quad \text{(update)}
```

```python
class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size

        scale = np.sqrt(2 / (combined_size + hidden_size))

        # Weights for z and r gates
        self.Wz = np.random.randn(combined_size, hidden_size) * scale
        self.Wr = np.random.randn(combined_size, hidden_size) * scale
        self.Wh = np.random.randn(combined_size, hidden_size) * scale

        self.bz = np.zeros(hidden_size)
        self.br = np.zeros(hidden_size)
        self.bh = np.zeros(hidden_size)

    def forward(self, x, h_prev):
        combined = np.concatenate([x, h_prev], axis=1)

        # Gates
        z = sigmoid(combined @ self.Wz + self.bz)
        r = sigmoid(combined @ self.Wr + self.br)

        # Candidate
        combined_reset = np.concatenate([x, r * h_prev], axis=1)
        h_tilde = np.tanh(combined_reset @ self.Wh + self.bh)

        # Update
        h_next = (1 - z) * h_prev + z * h_tilde

        self.cache = (x, h_prev, combined, z, r, h_tilde, combined_reset)
        return h_next
```

### LSTM vs GRU

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| States | 2 (cell, hidden) | 1 (hidden) |
| Parameters | More | ~25% fewer |
| Performance | Slightly better on long sequences | Comparable |
| Training | Slower | Faster |

**What this means:** GRU is simpler and trains faster. LSTM has more capacity for complex long-term dependencies. In practice, both work well—try GRU first for efficiency, LSTM if you need more power.

## Training LSTMs

### Gradient Clipping

Essential for stable training:

```python
def train_lstm(model, data, lr=0.001, clip_norm=5.0):
    """Training loop with gradient clipping."""
    optimizer = Adam(model.get_params(), lr=lr)

    for x_batch, y_batch in data:
        # Forward
        outputs, _ = model.forward(x_batch)
        loss = cross_entropy(outputs, y_batch)

        # Backward
        grads = model.backward(y_batch)

        # Clip gradients
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if total_norm > clip_norm:
            grads = [g * clip_norm / total_norm for g in grads]

        # Update
        optimizer.step(grads)
```

### Forget Gate Bias

Initialize forget gate bias to 1-2:

```python
# In LSTMCell.__init__:
self.b[hidden_size:2*hidden_size] = 1.0  # Forget gate bias

# Why? Start by remembering everything, let model learn what to forget.
```

### Dropout for LSTMs

Apply dropout to non-recurrent connections:

```python
class LSTMWithDropout:
    def __init__(self, input_size, hidden_size, dropout=0.5):
        self.cell = LSTMCell(input_size, hidden_size)
        self.dropout = dropout

    def forward(self, x_seq, h0, c0, training=True):
        outputs = []
        h, c = h0, c0

        for t in range(x_seq.shape[1]):
            x = x_seq[:, t, :]

            # Dropout on input (not recurrent)
            if training and self.dropout > 0:
                mask = np.random.binomial(1, 1-self.dropout, x.shape) / (1-self.dropout)
                x = x * mask

            h, c = self.cell.forward(x, h, c)
            outputs.append(h)

        return np.stack(outputs, axis=1), h, c
```

## Practical Example: Sequence Classification

```python
class SentimentLSTM:
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        # Embedding
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.01

        # LSTM
        self.lstm = LSTM(embed_dim, hidden_size, hidden_size)

        # Classifier (on final hidden state)
        self.classifier = np.random.randn(hidden_size, num_classes) * 0.01
        self.bias = np.zeros(num_classes)

    def forward(self, token_ids):
        """
        Args:
            token_ids: [batch_size, seq_len] integer token indices

        Returns:
            logits: [batch_size, num_classes]
        """
        # Embed tokens
        embedded = self.embedding[token_ids]  # [batch, seq, embed]

        # LSTM
        _, (final_h, _) = self.lstm.forward(embedded)

        # Classify from final hidden state
        logits = final_h[-1] @ self.classifier + self.bias

        return logits

    def predict(self, token_ids):
        logits = self.forward(token_ids)
        return np.argmax(logits, axis=-1)
```

## Summary

| Component | Purpose |
|-----------|---------|
| Cell state ($c$) | Long-term memory highway |
| Forget gate ($f$) | Erase old information |
| Input gate ($i$) | Write new information |
| Output gate ($o$) | Read from memory |
| Hidden state ($h$) | Short-term output |

**The essential insight:** LSTMs create a separate memory pathway (cell state) where information can flow with minimal transformation. Gates learned via gradient descent control what's remembered, what's added, and what's used. This architecture lets gradients flow back through many time steps, enabling learning of long-range dependencies that basic RNNs cannot capture.

**Historical note:** LSTMs dominated sequence modeling from ~2014-2017. Attention mechanisms (first used with LSTMs in seq2seq) eventually led to transformers, which now handle most sequence tasks. Understanding LSTMs illuminates why attention was needed and how transformers improve upon recurrent architectures.

**Next:** [Sequence-to-Sequence](sequence-to-sequence.md) covers encoder-decoder architectures for translation and generation.
