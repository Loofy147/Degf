# DEGF: Dynamic Entropy Genuineness Framework

This repository contains the reference implementation of the Dynamic Entropy Genuineness Framework (DEGF), as specified in the research papers. DEGF provides a thermodynamic theory for distinguishing 'genuine' reasoning from 'mechanical' pattern completion in LLMs.

## Core Features

### Track A — The Real-Time Reasoning Monitor
- **Live G-Stream**: Monitor token-by-token genuine computation (`monitor_gpt2.py`).
- **Hallucination Early-Warning**: Detected by low $ + low surprisal (high confidence).
- **Elaboration Guillotine**: Automatic truncation of vacuous responses when $\Delta G$ drops below threshold.

### Track B — Thermodynamic Training Signal
- **{thermo}$ Loss**: Encourages development of Q2 logic engine heads (`train_thermo.py`).
- **Emergent Reasoning**: Reward internal thermodynamic signatures of reasoning rather than just output formatting.

### Track C — SGS-2 Architecture Prototype
- **Decoupled Stream**: Separates reasoning (Latent Reasoner) from formatting (Syntax Decoder) (`sgs2_prototype.py`).
- **Phase Gate Recurrence**: G-stream controlled recurrence for deep deliberation before emission.

## Milestone A3 Validation

The framework has been empirically validated on GPT-2-small:
- **IOI Accuracy Drop**: 100% when ablating Q2 reasoning heads.
- **Induction Accuracy Drop**: 0% (mechanical circuits spared).
- **Double-Dissociation**: Confirmed.

## Usage

### Run Tests
```bash
python3 test_degf_v2.py
```

### Run A3 Ablation Test
```bash
python3 ablation_a3.py
```

### Start Monitor
```bash
python3 monitor_gpt2.py
```

### SGS-2 Inference Simulation
```bash
python3 sgs2_prototype.py
```

## Track D — APEX Unified Orchestrator
- **16-Layer Synthesis**: Comprehensive processing from raw signal detection to MCDA ranking.
- **DEGF Grounding**: Hyperparameter tuning grounded in internal genuineness metrics to prevent "gaming".
- **Self-Healing**: Autonomous engine optimization based on meta-G self-assessment.
