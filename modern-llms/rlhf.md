# Reinforcement Learning from Human Feedback (RLHF)

```math
\boxed{\max_\pi \mathbb{E}_{x \sim D, y \sim \pi}[R(x, y)] - \beta \cdot D_{KL}(\pi \| \pi_{\text{ref}})}
```

**RLHF** trains language models to maximize human preferences. Supervised fine-tuning teaches models *what* to say; RLHF teaches them *how* humans want them to say it. This is how models learn to be helpful, harmless, and honest—qualities that can't be captured by next-token prediction alone.

Prerequisites: [fine-tuning](fine-tuning.md), [optimization](../math-foundations/optimization.md). Code: `numpy`.

---

## Why RLHF?

### The Limits of Supervised Learning

Supervised fine-tuning requires input-output pairs:
```
Input: "What's the capital of France?"
Output: "The capital of France is Paris."
```

But for many qualities we care about, there's no "correct" output:

```
Input: "Tell me a joke"
Output: ??? (which joke is "correct"?)

Input: "Explain quantum mechanics"
Output: ??? (how detailed? what tone?)

Input: "Write a poem about love"
Output: ??? (infinite valid poems)
```

### What We Can Specify

We can't always demonstrate the right answer, but we can **compare** answers:

```
Output A: "Paris is the capital of France, known for the Eiffel Tower."
Output B: "Paris."
Output C: "I cannot answer that question."

Human judgment: A > B > C
```

RLHF learns from these comparisons.

**What this means:** RLHF captures human preferences that are easy to judge but hard to demonstrate. We know a helpful response when we see it, even if we can't write down the rule.

## The RLHF Pipeline

### Three Stages

```
Stage 1: Supervised Fine-Tuning (SFT)
    Pretrained → [SFT on demonstrations] → SFT Model

Stage 2: Reward Model Training
    [Human comparisons] → Reward Model (RM)

Stage 3: RL Optimization
    SFT Model + RM → [PPO training] → RLHF Model
```

```python
def rlhf_pipeline(pretrained_model, sft_data, comparison_data, prompts):
    # Stage 1: SFT
    sft_model = supervised_fine_tune(pretrained_model, sft_data)

    # Stage 2: Train reward model
    reward_model = train_reward_model(comparison_data)

    # Stage 3: RL optimization
    rlhf_model = ppo_training(sft_model, reward_model, prompts)

    return rlhf_model
```

## Stage 1: Supervised Fine-Tuning

### Purpose

Create a starting policy that:
- Follows instructions
- Generates reasonable responses
- Is close to the desired behavior

```python
def supervised_fine_tune(model, demonstrations):
    """
    Fine-tune on human demonstrations.

    demonstrations: List of (prompt, response) pairs
    """
    for prompt, response in demonstrations:
        tokens = tokenize(prompt + response)
        loss = next_token_loss(model, tokens)
        update(model, loss)

    return model
```

### Data Quality Matters

SFT data should be:
- High quality (written by skilled annotators)
- Diverse (cover many types of requests)
- Representative of desired behavior

## Stage 2: Reward Model

### Collecting Comparisons

Annotators compare model outputs:

```python
def collect_comparisons(model, prompts):
    """Generate comparison data for reward model training."""
    comparisons = []

    for prompt in prompts:
        # Generate multiple responses
        responses = [model.generate(prompt) for _ in range(4)]

        # Human ranks them (or pairwise comparisons)
        ranking = get_human_ranking(prompt, responses)

        comparisons.append({
            'prompt': prompt,
            'responses': responses,
            'ranking': ranking  # e.g., [2, 0, 3, 1] (indices in order of preference)
        })

    return comparisons
```

### Bradley-Terry Model

Convert rankings to reward scores using the Bradley-Terry model:

```math
p(\text{response } i \succ \text{response } j) = \sigma(r_i - r_j)
```

```python
import numpy as np

class RewardModel:
    def __init__(self, base_model):
        """
        Reward model predicts scalar reward for (prompt, response).

        Architecture: Same as base model, but outputs single value.
        """
        self.encoder = base_model
        self.reward_head = np.random.randn(base_model.d_model, 1) * 0.01

    def forward(self, prompt_tokens, response_tokens):
        """
        Args:
            prompt_tokens: [batch, prompt_len]
            response_tokens: [batch, response_len]
        """
        # Concatenate and encode
        tokens = np.concatenate([prompt_tokens, response_tokens], axis=1)
        hidden = self.encoder.forward(tokens)

        # Use last token's hidden state
        last_hidden = hidden[:, -1, :]

        # Scalar reward
        reward = last_hidden @ self.reward_head
        return reward.squeeze(-1)


def train_reward_model(reward_model, comparisons, lr=1e-5):
    """Train reward model on pairwise comparisons."""
    optimizer = AdamW(reward_model.params, lr=lr)

    for comparison in comparisons:
        prompt = comparison['prompt']
        responses = comparison['responses']
        ranking = comparison['ranking']

        # Get rewards for all responses
        rewards = []
        for response in responses:
            r = reward_model.forward(prompt, response)
            rewards.append(r)
        rewards = np.array(rewards)

        # Pairwise loss: preferred response should have higher reward
        loss = 0
        n_pairs = 0
        for i in range(len(ranking)):
            for j in range(i + 1, len(ranking)):
                # ranking[i] is preferred over ranking[j]
                idx_better = ranking[i]
                idx_worse = ranking[j]

                # Bradley-Terry loss
                loss -= np.log(sigmoid(rewards[idx_better] - rewards[idx_worse]))
                n_pairs += 1

        loss /= n_pairs
        optimizer.step(backward(reward_model, loss))

    return reward_model


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
```

**What this means:** The reward model learns to predict which responses humans prefer. It's trained on comparisons, so it captures relative quality—not absolute scores.

## Stage 3: RL Optimization

### The Objective

Maximize reward while staying close to the SFT model:

```math
\max_\pi \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x)}[R(x, y) - \beta \cdot \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}]
```

**Why the KL penalty?**
- Prevents reward hacking (exploiting RM weaknesses)
- Keeps language fluent (don't deviate too far from pretrained)
- Stabilizes training

### PPO for Language Models

```python
class PPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model,
                 kl_coef=0.1, clip_range=0.2):
        self.policy = policy_model
        self.ref = ref_model  # Frozen SFT model
        self.reward_model = reward_model
        self.kl_coef = kl_coef
        self.clip_range = clip_range

    def compute_rewards(self, prompts, responses):
        """Compute reward with KL penalty."""
        rewards = []

        for prompt, response in zip(prompts, responses):
            # Raw reward from RM
            raw_reward = self.reward_model.forward(prompt, response)

            # KL penalty
            policy_logprob = self.policy.log_prob(prompt, response)
            ref_logprob = self.ref.log_prob(prompt, response)
            kl = policy_logprob - ref_logprob

            # Total reward (penalize KL)
            total_reward = raw_reward - self.kl_coef * kl
            rewards.append(total_reward)

        return np.array(rewards)

    def ppo_step(self, prompts):
        """One PPO update step."""
        # Generate responses with current policy
        responses = [self.policy.generate(p) for p in prompts]

        # Compute advantages
        rewards = self.compute_rewards(prompts, responses)
        values = self.policy.value(prompts, responses)
        advantages = rewards - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Store old log probs
        old_log_probs = np.array([
            self.policy.log_prob(p, r)
            for p, r in zip(prompts, responses)
        ])

        # PPO update
        for _ in range(4):  # Multiple epochs per batch
            new_log_probs = np.array([
                self.policy.log_prob(p, r)
                for p, r in zip(prompts, responses)
            ])

            # Ratio
            ratio = np.exp(new_log_probs - old_log_probs)

            # Clipped objective
            surr1 = ratio * advantages
            surr2 = np.clip(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -np.minimum(surr1, surr2).mean()

            # Update policy
            update(self.policy, policy_loss)

        return {'reward': rewards.mean(), 'kl': (old_log_probs - new_log_probs).mean()}
```

### Training Loop

```python
def ppo_training(sft_model, reward_model, prompts, epochs=1):
    """Full PPO training loop."""
    policy = copy_model(sft_model)
    ref_model = copy_model(sft_model)
    freeze(ref_model)  # Don't update reference

    trainer = PPOTrainer(policy, ref_model, reward_model)

    for epoch in range(epochs):
        for batch_prompts in batch(prompts, batch_size=64):
            stats = trainer.ppo_step(batch_prompts)
            print(f"Reward: {stats['reward']:.3f}, KL: {stats['kl']:.4f}")

    return policy
```

## Reward Hacking

### The Problem

Models can find ways to get high reward without being helpful:

```
Prompt: "Explain machine learning"

Reward-hacked response:
"Machine learning is AMAZING! It's the BEST thing ever!
I'm SO HAPPY to help you! This is WONDERFUL!"
(Excessive positivity might score well but isn't helpful)
```

### Detection and Prevention

**KL penalty:** Already included in objective
```python
# Keep policy close to reference
reward -= kl_coef * (log_policy - log_ref)
```

**Reward model ensembles:**
```python
def robust_reward(response, reward_models):
    """Use minimum of multiple reward models."""
    rewards = [rm.forward(response) for rm in reward_models]
    return min(rewards)  # Conservative estimate
```

**Iterative training:**
```python
def iterative_rlhf():
    """Retrain reward model periodically on policy outputs."""
    for iteration in range(5):
        # Train policy with current RM
        policy = ppo_training(policy, reward_model, prompts)

        # Collect new comparisons on policy outputs
        new_comparisons = collect_human_comparisons(policy, prompts)

        # Retrain reward model
        reward_model = train_reward_model(reward_model, new_comparisons)
```

**What this means:** Reward models are imperfect proxies for human preferences. Optimizing them too aggressively finds their flaws. The KL penalty and iterative refinement help keep the model on track.

## Alternatives to PPO

### Direct Preference Optimization (DPO)

Skip the reward model—train directly on preferences:

```math
L_{DPO} = -\log \sigma\left(\beta \log\frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
```

```python
def dpo_loss(policy, ref_model, prompt, chosen, rejected, beta=0.1):
    """
    Direct Preference Optimization loss.

    No reward model needed—train directly on preferences.
    """
    # Log probs under policy
    log_pi_chosen = policy.log_prob(prompt, chosen)
    log_pi_rejected = policy.log_prob(prompt, rejected)

    # Log probs under reference
    log_ref_chosen = ref_model.log_prob(prompt, chosen)
    log_ref_rejected = ref_model.log_prob(prompt, rejected)

    # DPO objective
    log_ratio_chosen = log_pi_chosen - log_ref_chosen
    log_ratio_rejected = log_pi_rejected - log_ref_rejected

    loss = -np.log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    return loss
```

**Advantages:**
- Simpler pipeline (no RM, no RL)
- More stable training
- Computationally cheaper

**Disadvantages:**
- Less flexible than explicit reward model
- Can't reuse RM for other purposes

### Constitutional AI (CAI)

Use AI to generate feedback:

```python
def constitutional_ai(model, prompt, principles):
    """
    AI-generated feedback based on principles.

    principles: List of rules like "Be helpful", "Be honest"
    """
    # Generate initial response
    response = model.generate(prompt)

    # Ask model to critique itself
    critique_prompt = f"""
    Response: {response}

    Does this response follow these principles?
    {principles}

    Critique:"""
    critique = model.generate(critique_prompt)

    # Revise based on critique
    revision_prompt = f"""
    Original: {response}
    Critique: {critique}
    Revised response:"""
    revised = model.generate(revision_prompt)

    return revised
```

## Practical Considerations

### Data Requirements

| Stage | Data Size | Cost |
|-------|-----------|------|
| SFT | 10K-100K demonstrations | High (expert writing) |
| RM | 50K-200K comparisons | Medium (rating is faster) |
| RL | 100K+ prompts | Low (no labels needed) |

### Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| KL coefficient | 0.01-0.2 | Higher = more conservative |
| PPO clip range | 0.1-0.3 | Standard PPO values |
| Learning rate | 1e-6 to 5e-6 | Very small |
| Batch size | 64-512 | Larger helps stability |
| Epochs per batch | 1-4 | More epochs = more updates |

### Monitoring

```python
def monitor_rlhf(trainer, eval_prompts):
    """Track key metrics during RLHF."""
    metrics = {}

    # Reward
    responses = [trainer.policy.generate(p) for p in eval_prompts]
    rewards = trainer.compute_rewards(eval_prompts, responses)
    metrics['mean_reward'] = rewards.mean()

    # KL from reference
    kl = compute_kl(trainer.policy, trainer.ref, eval_prompts)
    metrics['kl_divergence'] = kl

    # Response length
    lengths = [len(r.split()) for r in responses]
    metrics['mean_length'] = np.mean(lengths)

    # Perplexity on held-out text (checking for degradation)
    ppl = compute_perplexity(trainer.policy, held_out_text)
    metrics['perplexity'] = ppl

    return metrics
```

## Summary

| Stage | Goal | Method |
|-------|------|--------|
| SFT | Basic instruction following | Fine-tune on demonstrations |
| Reward Model | Learn human preferences | Train on comparisons |
| RL (PPO) | Optimize for preferences | Maximize reward - KL penalty |

**The essential insight:** RLHF solves the problem of optimizing for qualities that are hard to specify but easy to judge. By collecting human comparisons, training a reward model, and optimizing against it with RL, we can teach models to be helpful, harmless, and honest—without needing explicit rules for these properties.

**Historical context:** RLHF was key to ChatGPT's success. The base GPT model can generate text, but RLHF made it conversational, helpful, and (mostly) safe. This alignment process is what turns a powerful but unpredictable language model into a useful assistant.

**Next:** [Instruction Tuning](instruction-tuning.md) covers the SFT stage in more detail—training models to follow instructions.
