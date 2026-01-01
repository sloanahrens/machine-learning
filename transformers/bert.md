# BERT

```math
\boxed{p(x_{\text{mask}} | x_{\text{context}}) = \text{softmax}(h_{\text{mask}} W^T)}
```

**BERT** (Bidirectional Encoder Representations from Transformers) revolutionized NLP by showing that deep bidirectional pretraining produces powerful representations. Unlike GPT's left-to-right generation, BERT sees context from both directions simultaneously, making it ideal for understanding tasks like classification, question answering, and information extraction.

Prerequisites: [transformer architecture](transformer-architecture.md), [multi-head attention](multi-head-attention.md). Code: `numpy`.

---

## The Key Insight

### The Problem with Left-to-Right

GPT-style models generate left-to-right:

```
"The cat sat on the ___"
      ↓   ↓   ↓   ↓
   [Transformer]
          ↓
        "mat"
```

Each position can only attend to previous positions. But for understanding tasks:

```
"The bank by the river was steep."
     ↑
What kind of bank? Need right context ("river") to know!
```

### BERT's Solution: Masking

Mask random tokens, predict them using full bidirectional context:

```
Input:  "The [MASK] sat on the mat"
              ↓
      [Bidirectional Transformer]
              ↓
Output: "cat"
```

Each position attends to all other positions—true bidirectionality.

**What this means:** BERT trades generation capability for better understanding. It can't generate text autoregressively, but its representations capture context from both directions.

## Architecture

### Model Structure

BERT uses only the **encoder** part of the transformer:

```
[CLS] token1 token2 ... [SEP]
  ↓     ↓      ↓    ↓    ↓
    [Transformer Encoder]
           (N layers)
  ↓     ↓      ↓    ↓    ↓
 h_cls  h1    h2   ... h_sep
```

| Component | BERT-Base | BERT-Large |
|-----------|-----------|------------|
| Layers | 12 | 24 |
| Hidden size | 768 | 1024 |
| Attention heads | 12 | 16 |
| Parameters | 110M | 340M |

### Implementation

```python
import numpy as np

class BERT:
    def __init__(self, vocab_size, d_model=768, num_layers=12,
                 num_heads=12, d_ff=3072, max_len=512):
        self.d_model = d_model

        # Embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.position_embedding = np.random.randn(max_len, d_model) * 0.02
        self.segment_embedding = np.random.randn(2, d_model) * 0.02  # Sentence A/B

        # Transformer layers
        self.layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]

        # Layer norm
        self.norm = LayerNorm(d_model)

    def forward(self, token_ids, segment_ids=None, attention_mask=None):
        """
        Args:
            token_ids: [batch_size, seq_len]
            segment_ids: [batch_size, seq_len] - 0 for sentence A, 1 for B
            attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding

        Returns:
            hidden_states: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = token_ids.shape

        # Combine embeddings
        positions = np.arange(seq_len)
        x = self.token_embedding[token_ids]
        x = x + self.position_embedding[positions]

        if segment_ids is not None:
            x = x + self.segment_embedding[segment_ids]

        # Create attention mask
        if attention_mask is not None:
            # [batch, 1, 1, seq_len] for broadcasting
            mask = attention_mask[:, np.newaxis, np.newaxis, :]
        else:
            mask = None

        # Pass through transformer layers
        for layer in self.layers:
            x = layer.forward(x, mask)

        return self.norm.forward(x)


class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Feed-forward
        self.ff1 = np.random.randn(d_model, d_ff) * np.sqrt(2 / d_model)
        self.ff2 = np.random.randn(d_ff, d_model) * np.sqrt(2 / d_ff)
        self.ff_bias1 = np.zeros(d_ff)
        self.ff_bias2 = np.zeros(d_model)

    def forward(self, x, mask=None):
        # Self-attention (bidirectional - no causal mask!)
        attn_out, _ = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_out)

        # Feed-forward with GELU activation
        ff_out = gelu(x @ self.ff1 + self.ff_bias1) @ self.ff2 + self.ff_bias2
        x = self.norm2.forward(x + ff_out)

        return x


def gelu(x):
    """Gaussian Error Linear Unit (BERT's activation)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
```

## Pretraining

### Task 1: Masked Language Modeling (MLM)

Randomly mask 15% of tokens, predict the originals:

```python
def prepare_mlm_batch(token_ids, vocab_size, mask_token_id,
                       mask_prob=0.15):
    """
    Prepare batch for masked language modeling.

    The 15% selected tokens are:
    - 80%: replaced with [MASK]
    - 10%: replaced with random token
    - 10%: kept unchanged
    """
    batch_size, seq_len = token_ids.shape
    masked_ids = token_ids.copy()
    labels = np.full_like(token_ids, -100)  # -100 = ignore in loss

    # Select positions to mask
    mask_positions = np.random.random((batch_size, seq_len)) < mask_prob

    # Don't mask special tokens (assume 0, 1, 2 are [PAD], [CLS], [SEP])
    mask_positions[:, 0] = False  # [CLS]
    for i in range(batch_size):
        sep_pos = np.where(token_ids[i] == 2)[0]
        if len(sep_pos) > 0:
            mask_positions[i, sep_pos[0]] = False

    # Set labels for masked positions
    labels[mask_positions] = token_ids[mask_positions]

    # Apply masking strategy
    for i in range(batch_size):
        for j in range(seq_len):
            if mask_positions[i, j]:
                rand = np.random.random()
                if rand < 0.8:
                    masked_ids[i, j] = mask_token_id  # [MASK]
                elif rand < 0.9:
                    masked_ids[i, j] = np.random.randint(3, vocab_size)  # Random
                # else: keep original (10%)

    return masked_ids, labels


class MLMHead:
    """Predict original tokens from BERT output."""

    def __init__(self, d_model, vocab_size):
        self.dense = np.random.randn(d_model, d_model) * 0.02
        self.bias = np.zeros(d_model)
        self.norm = LayerNorm(d_model)
        self.decoder = np.random.randn(d_model, vocab_size) * 0.02
        self.decoder_bias = np.zeros(vocab_size)

    def forward(self, hidden_states):
        x = gelu(hidden_states @ self.dense + self.bias)
        x = self.norm.forward(x)
        logits = x @ self.decoder + self.decoder_bias
        return logits
```

**What this means:** MLM forces BERT to build rich contextual representations. To predict "cat" in "The [MASK] sat on the mat", it must understand syntax, semantics, and world knowledge.

### Task 2: Next Sentence Prediction (NSP)

Predict whether sentence B follows sentence A:

```python
def prepare_nsp_batch(sentences, is_next_labels):
    """
    Prepare batch for next sentence prediction.

    Input: "[CLS] sentence A [SEP] sentence B [SEP]"
    Label: 1 if B follows A, 0 if random

    Note: NSP was later found to be less important than MLM.
    """
    # Format: [CLS] tokens_A [SEP] tokens_B [SEP]
    # segment_ids: 0 for A tokens, 1 for B tokens
    pass


class NSPHead:
    """Predict if sentence B follows sentence A."""

    def __init__(self, d_model):
        self.dense = np.random.randn(d_model, 2) * 0.02
        self.bias = np.zeros(2)

    def forward(self, cls_output):
        """
        Args:
            cls_output: [batch_size, d_model] - hidden state of [CLS] token
        """
        logits = cls_output @ self.dense + self.bias
        return logits  # Binary classification
```

### Pretraining Loss

```python
def bert_pretraining_loss(model, mlm_head, nsp_head,
                           token_ids, segment_ids, mlm_labels, nsp_labels):
    """Combined MLM and NSP loss."""
    # Forward pass
    hidden_states = model.forward(token_ids, segment_ids)

    # MLM loss
    mlm_logits = mlm_head.forward(hidden_states)
    mlm_loss = masked_cross_entropy(mlm_logits, mlm_labels)

    # NSP loss
    cls_output = hidden_states[:, 0]  # [CLS] token
    nsp_logits = nsp_head.forward(cls_output)
    nsp_loss = cross_entropy(nsp_logits, nsp_labels)

    return mlm_loss + nsp_loss


def masked_cross_entropy(logits, labels, ignore_index=-100):
    """Cross-entropy ignoring positions with ignore_index."""
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # Mask valid positions
    valid = labels_flat != ignore_index
    if not np.any(valid):
        return 0.0

    # Softmax and cross-entropy
    probs = softmax(logits_flat[valid])
    labels_valid = labels_flat[valid]
    loss = -np.log(probs[np.arange(len(labels_valid)), labels_valid] + 1e-10)

    return np.mean(loss)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

## Fine-Tuning

### The Fine-Tuning Recipe

1. Take pretrained BERT
2. Add task-specific head
3. Fine-tune all parameters on downstream task

```
Pretrained BERT (frozen or fine-tuned)
              ↓
[Task-specific head] ← New parameters
              ↓
          Output
```

### Classification (Single Sentence)

```python
class BertForClassification:
    def __init__(self, bert, num_classes, dropout=0.1):
        self.bert = bert
        self.dropout_rate = dropout
        self.classifier = np.random.randn(bert.d_model, num_classes) * 0.02
        self.classifier_bias = np.zeros(num_classes)

    def forward(self, token_ids, attention_mask=None, training=True):
        hidden_states = self.bert.forward(token_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        cls_output = hidden_states[:, 0]

        if training:
            cls_output = dropout(cls_output, self.dropout_rate)

        logits = cls_output @ self.classifier + self.classifier_bias
        return logits

    def predict(self, token_ids, attention_mask=None):
        logits = self.forward(token_ids, attention_mask, training=False)
        return np.argmax(logits, axis=-1)


def dropout(x, rate):
    if rate == 0:
        return x
    mask = np.random.binomial(1, 1 - rate, x.shape) / (1 - rate)
    return x * mask
```

### Token Classification (NER, POS)

```python
class BertForTokenClassification:
    """Predict a label for each token (e.g., named entity recognition)."""

    def __init__(self, bert, num_labels):
        self.bert = bert
        self.classifier = np.random.randn(bert.d_model, num_labels) * 0.02

    def forward(self, token_ids, attention_mask=None):
        hidden_states = self.bert.forward(token_ids, attention_mask=attention_mask)

        # Classify each token
        logits = hidden_states @ self.classifier
        return logits  # [batch, seq_len, num_labels]
```

### Question Answering

```python
class BertForQuestionAnswering:
    """Extract answer span from context."""

    def __init__(self, bert):
        self.bert = bert
        # Predict start and end positions
        self.qa_outputs = np.random.randn(bert.d_model, 2) * 0.02

    def forward(self, token_ids, attention_mask=None):
        """
        Input: "[CLS] question [SEP] context [SEP]"
        Output: start_logits, end_logits for each position
        """
        hidden_states = self.bert.forward(token_ids, attention_mask=attention_mask)

        logits = hidden_states @ self.qa_outputs
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]

        return start_logits, end_logits

    def predict(self, token_ids, attention_mask=None):
        start_logits, end_logits = self.forward(token_ids, attention_mask)

        start_idx = np.argmax(start_logits, axis=-1)
        end_idx = np.argmax(end_logits, axis=-1)

        # Ensure end >= start
        for i in range(len(start_idx)):
            if end_idx[i] < start_idx[i]:
                end_idx[i] = start_idx[i]

        return start_idx, end_idx
```

## Tokenization

### WordPiece

BERT uses WordPiece tokenization—splitting rare words into subwords:

```
"unhappiness" → ["un", "##happy", "##ness"]
"transformers" → ["transform", "##ers"]
```

```python
class SimpleWordPieceTokenizer:
    """Simplified WordPiece tokenizer (illustration only)."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.unk_token = "[UNK]"

    def tokenize(self, text):
        words = text.lower().split()
        tokens = []

        for word in words:
            if word in self.vocab:
                tokens.append(word)
            else:
                # Try to break into subwords
                subtokens = self._wordpiece(word)
                tokens.extend(subtokens)

        return tokens

    def _wordpiece(self, word):
        """Greedy longest-match-first."""
        tokens = []
        start = 0

        while start < len(word):
            end = len(word)
            found = False

            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr

                if substr in self.vocab:
                    tokens.append(substr)
                    found = True
                    break
                end -= 1

            if not found:
                tokens.append(self.unk_token)
                break
            start = end

        return tokens
```

### Special Tokens

| Token | Purpose |
|-------|---------|
| [CLS] | Classification token (start of sequence) |
| [SEP] | Separator (between sentences, end of sequence) |
| [MASK] | Masked token (for pretraining) |
| [PAD] | Padding token |
| [UNK] | Unknown token |

## BERT Variants

### RoBERTa

"Robustly optimized BERT":
- No NSP task
- Dynamic masking (different masks each epoch)
- Larger batches, more data
- Better hyperparameters

### ALBERT

"A Lite BERT":
- Factorized embedding (reduce parameters)
- Cross-layer parameter sharing
- Sentence order prediction instead of NSP

### DistilBERT

Smaller, faster:
- Knowledge distillation from BERT
- 40% smaller, 60% faster
- 97% of BERT's performance

### Comparison

| Model | Layers | Hidden | Params | Speed |
|-------|--------|--------|--------|-------|
| BERT-Base | 12 | 768 | 110M | 1x |
| BERT-Large | 24 | 1024 | 340M | 0.3x |
| DistilBERT | 6 | 768 | 66M | 1.6x |
| ALBERT-Base | 12 | 768 | 12M | 0.8x |
| RoBERTa-Base | 12 | 768 | 125M | 1x |

## BERT vs GPT

| Aspect | BERT | GPT |
|--------|------|-----|
| Direction | Bidirectional | Left-to-right |
| Pretraining | Masked LM | Autoregressive LM |
| Architecture | Encoder only | Decoder only |
| Best for | Understanding tasks | Generation tasks |
| Fine-tuning | Add task head | Prompt-based |

```
BERT: Sees full context
"The cat sat on the [MASK]"
  ↕   ↕   ↕  ↕   ↕     ↕
[All tokens attend to all others]

GPT: Sees only left context
"The cat sat on the ___"
  →   →   →  →   →
[Each token attends only to previous]
```

## Practical Usage Example

```python
def sentiment_classification_example():
    """End-to-end sentiment classification with BERT."""

    # 1. Load pretrained BERT
    bert = load_pretrained_bert("bert-base-uncased")

    # 2. Add classification head
    classifier = BertForClassification(bert, num_classes=2)

    # 3. Prepare data
    texts = ["This movie was amazing!", "Terrible waste of time."]
    labels = [1, 0]  # positive, negative

    # Tokenize
    tokenizer = load_tokenizer("bert-base-uncased")
    encoded = tokenizer.batch_encode(texts)
    token_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # 4. Fine-tune
    optimizer = AdamW(classifier.get_params(), lr=2e-5)

    for epoch in range(3):
        logits = classifier.forward(token_ids, attention_mask)
        loss = cross_entropy(logits, labels)

        grads = classifier.backward(labels)
        optimizer.step(grads)

        print(f"Epoch {epoch}: Loss = {loss:.4f}")

    # 5. Predict
    predictions = classifier.predict(token_ids, attention_mask)
    print(f"Predictions: {predictions}")
```

## Summary

| Concept | Description |
|---------|-------------|
| Bidirectional | Attend to full context, both directions |
| Masked LM | Predict masked tokens from context |
| [CLS] token | Aggregate representation for classification |
| Fine-tuning | Add task head, train on downstream task |
| WordPiece | Subword tokenization for rare words |

**The essential insight:** BERT showed that pretraining on massive unlabeled text creates representations useful for almost any NLP task. The masked language modeling objective forces deep bidirectional understanding—each prediction requires integrating context from both sides. This pretrain-then-fine-tune paradigm became the standard approach for NLP.

**Historical context:** BERT (2018) achieved state-of-the-art on 11 NLP benchmarks simultaneously. It catalyzed the "pretrained models" era, leading to GPT-2/3, T5, and eventually ChatGPT. The encoder-only architecture excels at understanding; decoder-only (GPT) excels at generation; encoder-decoder (T5) does both.

**Next:** The repository continues with advanced topics in [modern LLM techniques](../llm-techniques/) covering RLHF, instruction tuning, and efficient inference.
