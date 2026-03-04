# DEGF: Dynamic Entropy Genuineness Framework

This repository contains the reference implementation of the Dynamic Entropy Genuineness Framework (DEGF), as specified in the research papers.

## Key Components

- **`degf_core.py`**: The core mathematical framework implementing Shannon Entropy ($), Dynamic Variance ($), Collapse Events ($), and the Genuineness Score ($).
- **`degf_v2.py`**: Advanced improvements including adaptive thresholds, plateau simulators, and cascade detectors.
- **`monitor_gpt2.py`**: A live reasoning monitor that runs alongside GPT-2 to output a real-time himBHsstream.
- **`ablation_a3.py`**: Implementation of Milestone A3: The double-dissociation ablation test on GPT-2-small.

## Milestone A3 Validation

The DEGF framework has been empirically validated on GPT-2-small:
- **Ablation Targets**: 35 heads in late layers (L6-L11) with high himBHsscores.
- **IOI Accuracy Drop**: 100% (Baseline 100% -> Ablated 0%).
- **Induction Accuracy Drop**: 0% (Baseline 100% -> Ablated 100%).

This confirms that the DEGF metrics correctly identify the causal circuits responsible for genuine logical reasoning while sparing mechanical pattern-completion circuits.

## Running Tests

To run the full suite of 107 tests (V1 + V2):
```bash
python3 test_degf_v2.py
```

## Live Monitoring

To monitor a model's himBHsstream:
```bash
python3 monitor_gpt2.py
```
