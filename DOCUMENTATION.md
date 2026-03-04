# DEGF Developer Documentation

Welcome to the Dynamic Entropy Genuineness Framework (DEGF). This document provides a technical guide for developers using or extending the framework.

## 1. Core Metrics

All metrics are calculated per-head per-token.

### Shannon Entropy ($)
Measures the 'focus' of an attention head at position $.
1308H_t = -\sum a_{ti} \log_2(a_{ti})1308

### Dynamic Variance ($)
Measures how much the head's focus shifts during a sequence.
1308V = \text{Var}(H_0, \dots, H_T)1308

### Collapse Events ($)
Discrete events where the attention entropy drops sharply, signifying a 'logical click' or commitment.
1308C = \sum [ \Delta H_t < \theta_c ]1308
Default $\theta_c = -0.20$.

### Genuineness Score ($)
A sigmoid mapping of $ and $ to 1$.
1308G = \sigma(V + 0.5 \cdot C - 1.2)1308

## 2. API Reference

### Real-Time Monitor (`monitor_gpt2.py`)
The `DEGFMonitor` class provides token-by-token analysis.

```python
from monitor_gpt2 import DEGFMonitor
monitor = DEGFMonitor(model)
g_stream = monitor.monitor_step("Your prompt here")

for entry in g_stream:
    print(f"Token: {entry['token']} | G: {entry['G']:.4f} | Quality: {entry['Q']:.4f}")
```

### Training Signal (`train_thermo.py`)
The `compute_thermo_loss` function integrates with any standard training loop.

```python
loss, ce_loss, thermo_reward = compute_thermo_loss(model, tokens)
loss.backward()
```

### SGS-2 Recurrence (`sgs2_prototype.py`)
A prototype for models that genuinely deliberate.

```python
from sgs2_prototype import SGS2Prototype
sgs2_model = SGS2Prototype(base_model)
logits = sgs2_model("Reason about this...") # Triggers Phase Gate recurrence
```

## 3. Configuration

Key constants can be found in `degf_core.py`:
- `K_DEG`: Rate of genuineness degradation (-bash.8129$).
- `K_REC`: Rate of genuineness recovery (.2371$).
- `THETA_C`: Collapse threshold (himBHs0.20$).

## 4. Calibration

The himBHsscore is calibrated against reasoning quality using the following formula (IMP-4):
1308\hat{Q} = 0.802 \cdot G - 0.113 \cdot tc + 0.1451308
where $ is the normalized token cost (surprisal).
