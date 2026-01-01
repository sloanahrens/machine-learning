# Fine-Tuning

```math
\boxed{\theta^* = \arg\min_\theta \sum_{(x,y) \in \mathcal{D}_{\text{task}}} L(f_\theta(x), y)}
```

**Fine-tuning** adapts pretrained models to specific tasks. A model trained on general text can become a sentiment classifier, a code generator, or a medical assistant. This transfer learning paradigm—pretrain once, fine-tune many—is why modern LLMs are practical: pretraining is done once at massive cost, fine-tuning is cheap.

Prerequisites: [pretraining](pretraining.md), [regularization](../neural-networks/regularization.md). Code: `numpy`.

---

## Why Fine-Tune?

### Pretrained Models Know a Lot

After pretraining, an LLM:
- Understands language syntax and semantics
- Has encyclopedic factual knowledge
- Can follow logical reasoning patterns
- Knows programming conventions

But it doesn't know:
- Your specific task format
- Your domain terminology
- Your quality standards
- Your organization's style

### The Transfer Learning Insight

Fine-tuning is efficient because:

```
Pretrained representations contain:
├── Language understanding (general)
├── World knowledge (general)
└── Task patterns (need specialization)

Fine-tuning updates:
├── All layers (full fine-tuning)
├── Last few layers (partial)
└── Adapters/LoRA (efficient)
```

**What this means:** You don't start from scratch. The pretrained model already "knows" language; fine-tuning teaches it your specific task.

## Full Fine-Tuning

### The Basic Recipe

1. Start with pretrained weights
2. Add task-specific head if needed
3. Train on task data with small learning rate

```python
import numpy as np

def full_fine_tuning(pretrained_model, task_data,
                      lr=1e-5, epochs=3, batch_size=32):
    """
    Fine-tune all model parameters on task.

    Args:
        pretrained_model: Model with pretrained weights
        task_data: List of (input, label) pairs
    """
    optimizer = AdamW(pretrained_model.params, lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        np.random.shuffle(task_data)

        for i in range(0, len(task_data), batch_size):
            batch = task_data[i:i+batch_size]
            inputs, labels = zip(*batch)

            # Forward
            outputs = pretrained_model.forward(inputs)
            loss = compute_loss(outputs, labels)

            # Backward and update
            grads = pretrained_model.backward(labels)
            optimizer.step(grads)

        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

    return pretrained_model
```

### Task-Specific Heads

For classification, add a head on top:

```python
class FineTunedClassifier:
    def __init__(self, pretrained_lm, num_classes, pool="cls"):
        self.lm = pretrained_lm
        self.pool = pool  # "cls", "mean", or "last"

        hidden_size = pretrained_lm.d_model
        self.classifier = np.random.randn(hidden_size, num_classes) * 0.02
        self.bias = np.zeros(num_classes)

    def forward(self, token_ids, attention_mask=None):
        # Get hidden states from LM
        hidden = self.lm.forward(token_ids, attention_mask)

        # Pool to single vector
        if self.pool == "cls":
            pooled = hidden[:, 0]  # First token
        elif self.pool == "mean":
            if attention_mask is not None:
                mask = attention_mask[:, :, np.newaxis]
                pooled = (hidden * mask).sum(1) / mask.sum(1)
            else:
                pooled = hidden.mean(axis=1)
        else:  # "last"
            pooled = hidden[:, -1]

        # Classify
        logits = pooled @ self.classifier + self.bias
        return logits
```

### Learning Rate Considerations

Pretrained weights need gentle updates:

| Context | Learning Rate |
|---------|---------------|
| Pretraining | 1e-4 to 6e-4 |
| Full fine-tuning | 1e-5 to 5e-5 |
| Partial fine-tuning | 1e-4 to 5e-4 (new layers) |

**Why so small?** Large learning rates destroy pretrained representations:

```
Before fine-tuning:
[Pretrained weights] → Good language understanding

With lr=1e-3:
[Catastrophically updated] → Lost language understanding, overfits task

With lr=2e-5:
[Gently adapted] → Kept understanding + learned task
```

## Catastrophic Forgetting

### The Problem

As the model learns new task, it forgets pretrained knowledge:

```python
def measure_forgetting(model, pretraining_eval_data, task_data):
    """Track performance on original task during fine-tuning."""
    pretrain_losses = []

    for epoch in range(10):
        # Fine-tune on task
        train_epoch(model, task_data)

        # Evaluate on pretraining distribution
        pretrain_loss = evaluate(model, pretraining_eval_data)
        pretrain_losses.append(pretrain_loss)

    print(f"Pretraining loss: {pretrain_losses[0]:.3f} → {pretrain_losses[-1]:.3f}")
    # Typically increases (worse) during fine-tuning
```

### Mitigation Strategies

**Lower learning rate:**
```python
# Gentle updates preserve more knowledge
lr = 1e-5  # Not 1e-3
```

**Regularization toward pretrained weights:**
```python
def l2_from_init_loss(model, pretrained_params, lambda_reg=0.01):
    """Penalize divergence from pretrained weights."""
    reg_loss = 0
    for p, p_init in zip(model.params, pretrained_params):
        reg_loss += np.sum((p - p_init) ** 2)
    return lambda_reg * reg_loss
```

**Replay/mixing:**
```python
def mixed_training_batch(task_data, pretrain_data, task_ratio=0.8):
    """Mix task data with pretraining data."""
    task_batch = sample(task_data, int(batch_size * task_ratio))
    pretrain_batch = sample(pretrain_data, int(batch_size * (1 - task_ratio)))
    return task_batch + pretrain_batch
```

**Freeze early layers:**
```python
def freeze_layers(model, num_freeze):
    """Freeze first N transformer layers."""
    for i, layer in enumerate(model.layers):
        if i < num_freeze:
            layer.requires_grad = False
```

**What this means:** There's a trade-off between task performance and general capability. Aggressive fine-tuning maximizes task accuracy but damages the model's broader abilities.

## Few-Shot vs Full Fine-Tuning

### Data Requirements

| Approach | Data Needed | When to Use |
|----------|-------------|-------------|
| Zero-shot | 0 examples | Prompt describes task |
| Few-shot | 2-32 examples | In-context learning |
| Light fine-tune | 100-1000 | Limited data |
| Full fine-tune | 10K+ | Abundant task data |

### In-Context Learning (Zero/Few-Shot)

No parameter updates—just prompting:

```python
def few_shot_inference(model, examples, query):
    """
    Few-shot learning via prompting.

    Args:
        examples: List of (input, output) demonstration pairs
        query: New input to classify/generate for
    """
    # Build prompt with examples
    prompt = ""
    for inp, out in examples:
        prompt += f"Input: {inp}\nOutput: {out}\n\n"
    prompt += f"Input: {query}\nOutput:"

    # Generate
    response = model.generate(prompt)
    return response
```

**Advantages:**
- No training required
- Instantly swap tasks
- Works with API-only models

**Disadvantages:**
- Uses context window for examples
- Limited by in-context capacity
- Can't exceed prompting ceiling

### When to Fine-Tune

Fine-tune when:
- You have task-specific data (thousands of examples)
- Need consistent, reliable outputs
- Task requires specialized knowledge
- Prompting doesn't achieve required accuracy
- Want smaller, faster model

Prompt when:
- Limited data
- Task changes frequently
- Using API-only model
- Prototyping

## Evaluation

### Task-Specific Metrics

| Task | Metrics |
|------|---------|
| Classification | Accuracy, F1, AUC |
| Generation | BLEU, ROUGE, perplexity |
| QA | Exact match, F1 |
| Summarization | ROUGE-L, human eval |
| Code | Pass@k, HumanEval |

### Held-Out Validation

```python
def evaluate_fine_tuned(model, test_data):
    """Standard evaluation on held-out test set."""
    predictions = []
    labels = []

    for inputs, target in test_data:
        pred = model.predict(inputs)
        predictions.append(pred)
        labels.append(target)

    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    return accuracy
```

### Overfitting Detection

Monitor training vs validation:

```python
def train_with_early_stopping(model, train_data, val_data, patience=3):
    """Stop when validation performance plateaus."""
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(100):
        # Train
        train_epoch(model, train_data)

        # Validate
        val_loss = evaluate_loss(model, val_data)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy_weights(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            restore_weights(model, best_weights)
            break

    return model
```

## Domain Adaptation

### Continued Pretraining

Fine-tune on domain text before task fine-tuning:

```
General pretrained → Domain pretrained → Task fine-tuned
     (GPT)            (MedicalGPT)        (DiagnosisGPT)
```

```python
def domain_adaptation(model, domain_corpus, task_data):
    """Two-stage adaptation: domain then task."""

    # Stage 1: Continue pretraining on domain
    for text in domain_corpus:
        tokens = tokenize(text)
        loss = next_token_loss(model, tokens)
        update(model, loss, lr=1e-5)

    # Stage 2: Fine-tune on task
    model = full_fine_tuning(model, task_data, lr=2e-5)

    return model
```

### When Domain Adaptation Helps

- Legal: Dense legal terminology
- Medical: Clinical notes, drug names
- Finance: Specific jargon, formats
- Code: Programming languages not in pretraining

## Multi-Task Fine-Tuning

### Learning Multiple Tasks

Train on multiple tasks simultaneously:

```python
def multi_task_fine_tuning(model, task_datasets, task_heads):
    """
    Train on multiple tasks with task-specific heads.

    Args:
        task_datasets: Dict of task_name -> data
        task_heads: Dict of task_name -> classifier head
    """
    # Combine all data with task labels
    all_data = []
    for task_name, data in task_datasets.items():
        for x, y in data:
            all_data.append((x, y, task_name))

    for epoch in range(3):
        np.random.shuffle(all_data)

        for x, y, task_name in all_data:
            # Forward through shared encoder
            hidden = model.encode(x)

            # Use task-specific head
            head = task_heads[task_name]
            logits = head.forward(hidden)

            loss = compute_loss(logits, y)

            # Update encoder and task head
            backward_and_update(model, head, loss)
```

**Benefits:**
- Shared representations improve data efficiency
- Regularizes through task diversity
- Single model for multiple tasks

**Challenges:**
- Balancing task difficulties
- Negative transfer between dissimilar tasks
- Larger combined dataset needed

## Summary

| Concept | Description |
|---------|-------------|
| Full fine-tuning | Update all parameters on task data |
| Catastrophic forgetting | Losing pretrained knowledge |
| Few-shot | Learn from examples in prompt |
| Domain adaptation | Pretrain further on domain |
| Multi-task | Train on multiple tasks jointly |

**The essential insight:** Fine-tuning works because pretrained models have learned general-purpose representations. Small amounts of task data can redirect these capabilities to specific tasks. The key is balancing task learning against forgetting—too aggressive and you lose generality; too conservative and you underfit the task.

**Next:** [RLHF](rlhf.md) covers learning from human preferences beyond supervised fine-tuning.
