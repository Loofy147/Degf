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
from train_thermo import compute_thermo_loss
loss, ce_loss, thermo_reward = compute_thermo_loss(model, tokens)
loss.backward()
```

### SGS-2 Architecture (`sgs2_prototype.py`)
The `SGS2Prototype` class implements the Phase Gate and Elaboration Guillotine.

```python
from sgs2_prototype import SGS2Prototype
sgs2_model = SGS2Prototype(base_model)

# Full generation loop with genuine deliberation
result = sgs2_model.generate(
    "If all dogs are mammals...",
    max_new_tokens=20,
    max_loops=3,
    window=5,
    threshold=-0.15
)
```

## 3. Empirical Benchmarks (`degf_v6.py`)

The framework includes several benchmarks for empirical verification. Use the `--real` flag to run them against a live model.

### TRT Benchmark (EXP-6)
Measures the 'G-gap' between deductive reasoning tasks and inductive pattern completion.

### Hallucination F1 (EXP-7)
Identifies hallucinations using the signature: **Low G ($<$ 0.4) + High Confidence (tc $<$ 0.4)**.

### Thermodynamic Shift (EXP-9)
Measures the increase in Q2 head density after fine-tuning with the `L_thermo` loss.

## 4. Configuration

Key constants can be found in `degf_core.py`:
- `K_DEG`: Rate of genuineness degradation (0.8129).
- `K_REC`: Rate of genuineness recovery (1.2371).
- `THETA_C`: Collapse threshold (-0.20).

## 5. Calibration

The G-score is calibrated against reasoning quality using the following formula (IMP-4):
$$\hat{Q} = 0.802 \cdot G - 0.113 \cdot tc + 0.145$$
where $tc$ is the normalized token cost (surprisal).
