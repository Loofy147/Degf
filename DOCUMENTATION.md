# DEGF Developer Documentation

Welcome to the Dynamic Entropy Genuineness Framework (DEGF). This document provides a technical guide for developers using or extending the framework.

## 1. Core Metrics

All metrics are calculated per-head per-token.

### Shannon Entropy ($H_t$)
Measures the 'focus' of an attention head at position $t$.
$$H_t = -\sum a_{ti} \log_2(a_{ti})$$

### Dynamic Variance ($V$)
Measures how much the head's focus shifts during a sequence.
$$V = \text{Var}(H_0, \dots, H_T)$$

### Collapse Events ($C$)
Discrete events where the attention entropy drops sharply, signifying a 'logical click' or commitment.
$$C = \sum [ \Delta H_t < \theta_c ]$$
Default $\theta_c = -0.20$.

### Genuineness Score ($G$)
A sigmoid mapping of $V$ and $C$ to $[0, 1]$.
$$G = \sigma(V + 0.5 \cdot C - 1.2)$$

## 2. API Reference

### Real-Time Monitor (`monitor_gpt2.py`)
Provides token-by-token analysis for TransformerLens and Hugging Face models.

#### TransformerLens
```python
from monitor_gpt2 import DEGFMonitor
monitor = DEGFMonitor(model)
g_stream = monitor.monitor_step("Your prompt here")
```

#### Hugging Face
```python
from monitor_gpt2 import HFDEGFMonitor
monitor = HFDEGFMonitor(model, tokenizer)
g_stream = monitor.monitor_step("Your prompt here")
```

### Training Signal (`train_thermo.py`, `train_deepseek_v6.py`)
The `compute_thermo_loss` functions integrate with standard training loops.

```python
from train_thermo import compute_thermo_loss
loss, ce_loss, thermo_reward = compute_thermo_loss(model, tokens)
loss.backward()
```

### SGS-2 Architecture (`sgs2_prototype.py`)
Model-agnostic prototype for models that genuinely deliberate.

```python
from sgs2_prototype import SGS2Prototype
sgs2_model = SGS2Prototype(model, tokenizer) # Supports TL or HF

# Full generation loop with genuine deliberation
result = sgs2_model.generate("def hello_world():", max_new_tokens=10)
```

## 3. Empirical Benchmarks (`degf_v6.py`)

The framework includes benchmarks for empirical verification. Support for DeepSeek-Coder-1.3B is included via the `--deepseek` flag.

### TRT Benchmark (EXP-6)
Measures the 'G-gap' between deductive reasoning tasks and inductive pattern completion.

### Hallucination F1 (EXP-7 / Milestone D4)
Identifies hallucinations using the signature: **Low G ($<$ 0.4) + High Confidence (tc $<$ 0.4)**.

### Thermodynamic Shift (EXP-9)
Measures the increase in Q2 head density after fine-tuning with the `L_thermo` loss.

## 4. Scaling Laws

DEGF metrics are architecture-stable and scale with model size. Milestone D4 validation on `gpt2-medium` and DeepSeek-Coder-1.3B confirms that the thermodynamic signature of reasoning remains a robust diagnostic across scales.
