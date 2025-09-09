# Gate-Conditioned LoRA GPT

## 📌 Overview

**Gate-Conditioned LoRA GPT** is a new training paradigm that combines the efficiency of **LoRA adapters** with the adaptability of **Mixture-of-Experts (MoE)** routing.

Instead of using a single LoRA adapter, the model maintains **multiple LoRA experts**, and a **gating network** dynamically selects which experts to apply **per input sample**.

This creates a **conditional mixture-of-adapters model** that learns to specialize and route automatically, enabling efficient multi-task and domain adaptation.

---

## 🔬 Theory

### Architecture

* **Base GPT (frozen backbone)**: retains general-purpose knowledge.
* **LoRA Experts**: small trainable adapters attached to attention and projection layers.
* **Gating Network**: projects hidden states → probability distribution over experts.
* **Top-k Routing**: activates only a few experts per sample for efficiency.

### Training Objective

* **Language Modeling Loss**: standard cross-entropy on target tokens.
* **Entropy Regularization**: encourages balanced expert usage and avoids collapse to a single adapter.

$$
\mathcal{L} = \mathcal{L}_{LM} - \lambda \cdot H(p_\text{gate})
$$

Where $H(p_\text{gate})$ is the entropy of the gating distribution.

---

## ⚡ Why Gate-Conditioned LoRA?

✅ **Dynamic Specialization** — different LoRA experts specialize on different input patterns.
✅ **Parameter Efficiency** — LoRA experts are lightweight, so scaling is cheap.
✅ **Conditional Computation** — only top-k experts are activated per input.
✅ **Robustness** — gating + entropy regularization prevents overfitting.
✅ **Extensible** — can be combined with evolutionary training (HybridGARL) or RLHF.

---

## 📜 Example Usage

```python
from gate_conditioned_lora_gpt import GateConditionedLoRA
from transformers import AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GateConditionedLoRA(base_model_name="gpt2", r=16, num_experts=4, topk=2).to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer("Translate 'bonjour' to English:", return_tensors="pt").to(device)
labels = tokenizer("hello", return_tensors="pt").input_ids.to(device)

out = model(**inputs, labels=labels)
print("Loss:", out["loss"])
print("Gate probabilities:", out["gate_probs"])
```

---

## 🧪 Training Loop (Pseudo-code)

```python
for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs["loss"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 🔑 Key Applications

* **Multi-task training**: each LoRA learns a different task/domain.
* **Dynamic domain adaptation**: model routes inputs to specialized adapters.
* **Ensemble-style robustness**: behaves like a conditional ensemble but cheaper.

---

## 📖 Citation (concept draft)

```
@article{gateconditionedlora2025,
  title={Gate-Conditioned LoRA: Conditional Mixture-of-Adapters for Efficient Multi-Task GPT},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

---

✨ Gate-Conditioned LoRA GPT = **LoRA × Mixture-of-Experts** — efficient, dynamic, and adaptive.


### PSEUDOCODE

```python
#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# -----------------------
# Gate-Conditioned LoRA GPT
# -----------------------
class GateConditionedLoRA(nn.Module):
    def __init__(self, base_model_name="gpt2", r=16, num_experts=4, topk=2, entropy_weight=0.01):
        super().__init__()

        # Base GPT backbone (frozen)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.hidden_size = self.base_model.config.hidden_size
        self.num_experts = num_experts
        self.topk = topk
        self.entropy_weight = entropy_weight

        # Create multiple LoRA experts
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            lora_cfg = LoraConfig(
                r=r,
                lora_alpha=2*r,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.05,
                task_type="CAUSAL_LM"
            )
            expert = get_peft_model(AutoModelForCausalLM.from_pretrained(base_model_name), lora_cfg)
            self.experts.append(expert)

        # Gating network: hidden state -> expert logits
        self.gate_proj = nn.Linear(self.hidden_size, num_experts)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Hidden states from frozen base GPT
        with torch.no_grad():
            base_out = self.base_model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = base_out.hidden_states[-1][:, 0, :]  # CLS-like

        # Gating distribution
        gate_scores = self.gate_proj(hidden)               # (batch, num_experts)
        gate_probs = torch.softmax(gate_scores, dim=-1)    # (batch, num_experts)

        # Select top-k experts
        gate_mask = torch.zeros_like(gate_probs)
        _, topk_idx = torch.topk(gate_probs, self.topk, dim=-1)
        gate_mask.scatter_(1, topk_idx, 1.0)

        # Run experts
        all_logits = []
        for expert in self.experts:
            out = expert(input_ids=input_ids, attention_mask=attention_mask, labels=None, return_dict=True)
            all_logits.append(out.logits.unsqueeze(0))  # (1, batch, seq, vocab)

        all_logits = torch.cat(all_logits, dim=0)  # (num_experts, batch, seq, vocab)

        # Mix expert outputs
        weighted_logits = []
        for i in range(input_ids.size(0)):
            weights = (gate_mask[i] * gate_probs[i]).unsqueeze(-1).unsqueeze(-1)
            mixed = (weights * all_logits[:, i]).sum(dim=0)
            weighted_logits.append(mixed.unsqueeze(0))
        logits = torch.cat(weighted_logits, dim=0)  # (batch, seq, vocab)

        # Loss (optional)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Entropy regularization
            entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=-1).mean()
            loss = lm_loss - self.entropy_weight * entropy

        return {"loss": loss, "logits": logits, "gate_probs": gate_probs, "gate_mask": gate_mask}


# -----------------------
# Example Run
# -----------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GateConditionedLoRA(base_model_name="gpt2", r=16, num_experts=4, topk=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 🔥 Example prompt
    prompt = "Translate 'bonjour' to English:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Greedy decode using gated logits
    next_token = torch.argmax(outputs["logits"][:, -1, :], dim=-1)
    decoded = tokenizer.decode(next_token)

    print("Prompt:", prompt)
    print("Next token prediction:", decoded)
    print("Gate probabilities:", outputs["gate_probs"].detach().cpu())
```
