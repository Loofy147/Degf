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

## 5. APEX Integrated System (v2.0)

APEX is a unified orchestrator that integrates DEGF metrics with 16 layers of signal processing, synthesis, and mathematical discovery.

### Architecture Overview (`apex_v2.py`)
- **Signal Detection**: Winsorized-SNR, Hurst exponent, Spectral Entropy, and Isolation Forest anomalies.
- **Synthesis Engine**: 6 methods including `UltraSynthesisV3` with diversity injection.
- **Extensive Tuning Engine**: Regime-aware hyperparameter optimization grounded in DEGF G-scores (`extensive_tuning_v3.py`).
- **Meta-Monitoring**: Level-3 self-application of DEGF to the engine's own computation stream.

### Key Components

#### ExtensiveTuningEngine (`extensive_tuning_v3.py`)
Robust optimization that balances performance (Q-score) and structural genuineness (G-score).
- **Anti-Gaming Guard**: Penalizes parameter sets that produce trivially perfect but non-genuine results.
- **Regime Fingerprinting**: Automatically selects tuning grids based on signal characteristics (Trending, Periodic, Stochastic).

#### SelfHealLoop
Automatically triggers the `SelfOptimizer` when `meta_G` drops below 0.50, ensuring the engine maintains high-quality synthesis.

### Usage
```python
from apex_v2 import APEX, ResearchDataset
apex = APEX()
report = apex.run(ResearchDataset("MyData", series))
```

## 6. Mathematical Foundations (`advanced_modules.py`)

DEGF grounding allows the discovery engine to handle advanced mathematical domains with high reliability.

- **MELNIKOV**: Hamiltonian chaos detection via homoclinic/heteroclinic analysis.
- **SLOWFAST**: Fenichel theory and Canard explosion analysis for singularly perturbed systems.
- **DDE**: Hopf bifurcation and multi-stability analysis for delay differential equations.
- **PDE_RD**: Turing instability and energy functional analysis for reaction-diffusion systems.
