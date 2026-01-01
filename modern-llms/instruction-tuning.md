# Instruction Tuning

$$
\boxed{L = -\sum_{t \in \text{response}} \log p(t | \text{instruction}, \text{input}, t_{<})}
$$

**Instruction tuning** teaches language models to follow directions. A pretrained model generates plausible text; an instruction-tuned model does what you ask. This is the "supervised fine-tuning" stage that precedes RLHF—it transforms a text predictor into an assistant.

Prerequisites: [fine-tuning](fine-tuning.md), [GPT](../transformers/gpt.md). Code: `numpy`.

---

## From Completion to Instruction-Following

### The Problem

Pretrained models complete text:

```
Input: "The capital of France is"
Output: "Paris, known for the Eiffel Tower and its rich history..."

Input: "What is the capital of France?"
Output: "What is the capital of Germany? What is the capital of Italy?..."
```

The model predicts likely next tokens, not necessarily answers.

### Instruction Tuning Solution

Train on instruction-response pairs:

```
Instruction: What is the capital of France?
Response: The capital of France is Paris.

Instruction: Write a haiku about autumn.
Response: Leaves fall gently down
Golden carpet on the ground
Nature's last hurrah

Instruction: Explain quantum computing to a 5-year-old.
Response: Imagine you have a magic coin...
```

**What this means:** Instruction tuning teaches the model a new "mode" of operation—instead of just continuing text, it learns to respond to requests.

## Data Format

### Basic Format

```
<instruction>
What is photosynthesis?
</instruction>

<response>
Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen...
</response>
```

### Chat Format

Most modern models use chat templates:

```python
def format_chat(messages):
    """
    Format conversation for training.

    messages: List of {"role": "user"|"assistant", "content": str}
    """
    formatted = ""
    for msg in messages:
        if msg["role"] == "user":
            formatted += f"<|user|>\n{msg['content']}<|end|>\n"
        elif msg["role"] == "assistant":
            formatted += f"<|assistant|>\n{msg['content']}<|end|>\n"
    return formatted

# Example
messages = [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "2 + 2 equals 4."}
]
print(format_chat(messages))
```

### System Prompts

Prepend behavior instructions:

```
<|system|>
You are a helpful, harmless, and honest AI assistant. Answer questions accurately and concisely.
<|end|>

<|user|>
What causes rain?
<|end|>

<|assistant|>
Rain is caused by water vapor in the atmosphere condensing into droplets...
<|end|>
```

```python
def format_with_system(system_prompt, conversation):
    """Add system prompt to conversation."""
    return f"<|system|>\n{system_prompt}<|end|>\n" + format_chat(conversation)
```

## Training

### Loss Computation

Only compute loss on the response tokens:

```python
import numpy as np

def instruction_tuning_loss(model, instruction, response, tokenizer):
    """
    Compute loss only on response tokens.

    Args:
        instruction: The instruction/prompt text
        response: The target response text
    """
    # Tokenize
    instruction_tokens = tokenizer.encode(instruction)
    response_tokens = tokenizer.encode(response)

    # Full sequence
    full_sequence = instruction_tokens + response_tokens

    # Forward pass
    logits = model.forward(full_sequence[:-1])  # Input
    targets = full_sequence[1:]  # Targets (shifted by 1)

    # Create mask: 0 for instruction, 1 for response
    instruction_len = len(instruction_tokens)
    mask = np.zeros(len(targets))
    mask[instruction_len - 1:] = 1  # Only count response tokens

    # Compute loss
    log_probs = log_softmax(logits)
    token_losses = -log_probs[np.arange(len(targets)), targets]
    masked_loss = (token_losses * mask).sum() / mask.sum()

    return masked_loss


def log_softmax(x):
    max_x = np.max(x, axis=-1, keepdims=True)
    return x - max_x - np.log(np.sum(np.exp(x - max_x), axis=-1, keepdims=True))
```

### Training Loop

```python
def instruction_tune(model, data, epochs=3, lr=2e-5, batch_size=8):
    """
    Instruction tuning training loop.

    data: List of (instruction, response) pairs
    """
    optimizer = AdamW(model.params, lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        np.random.shuffle(data)
        total_loss = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_loss = 0

            for instruction, response in batch:
                loss = instruction_tuning_loss(model, instruction, response)
                batch_loss += loss

            batch_loss /= len(batch)
            grads = backward(model, batch_loss)

            # Gradient clipping
            clip_gradients(grads, max_norm=1.0)

            optimizer.step(grads)
            total_loss += batch_loss

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(data) * batch_size:.4f}")

    return model
```

## Dataset Construction

### Types of Instructions

Diverse instructions improve generalization:

| Category | Example |
|----------|---------|
| Question answering | "What is the boiling point of water?" |
| Explanation | "Explain how vaccines work" |
| Generation | "Write a poem about mountains" |
| Rewriting | "Summarize this article in 3 sentences" |
| Coding | "Write a Python function to sort a list" |
| Reasoning | "If A > B and B > C, what can we conclude?" |
| Classification | "Is this review positive or negative?" |
| Extraction | "List the main characters in this story" |

### FLAN: Scaling Instruction Tuning

FLAN (Fine-tuned LAnguage Net) showed scaling task diversity helps:

```python
def flan_style_dataset(tasks):
    """
    Create FLAN-style dataset with task variety.

    tasks: Dict of task_name -> list of examples
    """
    data = []

    for task_name, examples in tasks.items():
        # Create multiple instruction templates per task
        templates = get_templates(task_name)

        for example in examples:
            # Use random template
            template = np.random.choice(templates)
            instruction = template.format(**example)
            response = example['output']

            data.append((instruction, response))

    return data

def get_templates(task_name):
    """Return instruction templates for a task."""
    if task_name == "sentiment":
        return [
            "Classify the sentiment: {text}",
            "Is this positive or negative? {text}",
            "What sentiment does this express? {text}",
            "Determine if positive/negative: {text}"
        ]
    elif task_name == "summarization":
        return [
            "Summarize: {text}",
            "Write a brief summary of: {text}",
            "TLDR: {text}",
            "What are the main points of: {text}"
        ]
    # ... more tasks
```

### Self-Instruct

Generate training data using the model itself:

```python
def self_instruct(model, seed_tasks, num_generate=1000):
    """
    Generate instruction-tuning data using the model.

    1. Start with seed tasks
    2. Generate new instructions
    3. Generate responses
    4. Filter low quality
    """
    all_tasks = list(seed_tasks)

    for _ in range(num_generate):
        # Sample seed examples
        examples = np.random.choice(all_tasks, 3)

        # Generate new instruction
        prompt = f"""
Given these example tasks:
1. {examples[0]['instruction']}
2. {examples[1]['instruction']}
3. {examples[2]['instruction']}

Generate a new, different task instruction:
"""
        new_instruction = model.generate(prompt)

        # Generate response for new instruction
        response = model.generate(f"Instruction: {new_instruction}\nResponse:")

        # Filter (length, diversity, quality checks)
        if is_valid(new_instruction, response, all_tasks):
            all_tasks.append({
                'instruction': new_instruction,
                'response': response
            })

    return all_tasks
```

## Key Instruction-Tuned Models

### Evolution

```
GPT-3 (2020): Pretrained only, few-shot via prompting
FLAN (2022): Instruction-tuned, better zero-shot
InstructGPT (2022): Instruction-tuned + RLHF
ChatGPT (2022): InstructGPT optimized for dialogue
```

### Model Comparison

| Model | Base | Method | Key Contribution |
|-------|------|--------|------------------|
| FLAN | T5 | Multi-task instruction tuning | Task diversity matters |
| InstructGPT | GPT-3 | SFT + RLHF | Human preference alignment |
| Alpaca | LLaMA | Self-Instruct | Low-cost instruction data |
| Vicuna | LLaMA | ShareGPT conversations | Using chat data |

## Multi-Turn Conversations

### Training on Dialogues

```python
def format_conversation_for_training(conversation):
    """
    Format multi-turn conversation.

    conversation: List of turns
    Returns: (context, response) pairs for each assistant turn
    """
    training_pairs = []

    for i, turn in enumerate(conversation):
        if turn['role'] == 'assistant':
            # Context is everything before
            context = format_chat(conversation[:i])
            response = turn['content']
            training_pairs.append((context, response))

    return training_pairs


def train_on_conversations(model, conversations):
    """Train on multi-turn conversations."""
    for convo in conversations:
        pairs = format_conversation_for_training(convo)
        for context, response in pairs:
            loss = instruction_tuning_loss(model, context, response)
            update(model, loss)
```

### Context Window Considerations

Long conversations may exceed context:

```python
def truncate_conversation(messages, max_tokens, tokenizer):
    """Keep most recent messages that fit."""
    # Always keep system prompt
    system_tokens = len(tokenizer.encode(messages[0]['content']))
    remaining = max_tokens - system_tokens

    kept_messages = [messages[0]]  # System

    # Add messages from most recent
    for msg in reversed(messages[1:]):
        msg_tokens = len(tokenizer.encode(format_message(msg)))
        if msg_tokens < remaining:
            kept_messages.insert(1, msg)
            remaining -= msg_tokens
        else:
            break

    return kept_messages
```

## Evaluation

### Benchmarks

| Benchmark | What It Measures |
|-----------|------------------|
| MMLU | Multitask knowledge |
| HellaSwag | Commonsense reasoning |
| TruthfulQA | Truthfulness |
| HumanEval | Code generation |
| MT-Bench | Multi-turn conversation quality |

### Human Evaluation

```python
def human_eval_setup(model, prompts):
    """Generate samples for human evaluation."""
    samples = []

    for prompt in prompts:
        response = model.generate(prompt)
        samples.append({
            'prompt': prompt,
            'response': response,
            'scores': {
                'helpfulness': None,  # 1-5
                'accuracy': None,     # 1-5
                'harmlessness': None, # 1-5
                'coherence': None     # 1-5
            }
        })

    return samples  # Send to annotators
```

## Practical Recipes

### Quality Over Quantity

Research shows high-quality data matters more than size:

```python
def quality_filter(examples):
    """Filter low-quality instruction-response pairs."""
    filtered = []

    for inst, resp in examples:
        # Length checks
        if len(inst.split()) < 3 or len(resp.split()) < 5:
            continue

        # Response actually addresses instruction
        if not response_addresses_instruction(inst, resp):
            continue

        # Not just copying instruction
        if resp.lower().startswith(inst.lower()[:20]):
            continue

        # No obvious errors
        if contains_obvious_errors(resp):
            continue

        filtered.append((inst, resp))

    return filtered
```

### Mixing Pretraining Data

Prevent catastrophic forgetting:

```python
def mixed_batch(instruction_data, pretrain_data, inst_ratio=0.8):
    """Mix instruction and pretraining data."""
    batch = []

    n_inst = int(batch_size * inst_ratio)
    n_pretrain = batch_size - n_inst

    batch.extend(sample(instruction_data, n_inst))
    batch.extend(sample(pretrain_data, n_pretrain))

    return batch
```

## Summary

| Concept | Description |
|---------|-------------|
| Instruction format | instruction + response pairs |
| Chat templates | Structured format with roles |
| Loss masking | Only compute loss on responses |
| Task diversity | Many instruction types improve generalization |
| Self-Instruct | Generate training data with model |

**The essential insight:** Instruction tuning is about changing the model's "mode" from text completion to task execution. By training on diverse (instruction, response) pairs, the model learns that when given a request, it should fulfill it—not just continue the text pattern. This is the foundation that makes chatbots useful.

**Historical context:** Instruction tuning was the key insight that made LLMs practical assistants. FLAN showed that task diversity matters; InstructGPT showed human demonstrations + RLHF makes models helpful. Modern assistants combine large-scale instruction tuning with RLHF.

**Next:** [Efficient Adaptation](efficient-adaptation.md) covers parameter-efficient fine-tuning methods like LoRA.
