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
