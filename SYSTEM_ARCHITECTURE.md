# APEX v2.0 System Architecture

This document outlines the integration of core logics extracted from the repository into the unified APEX v2.0 orchestrator.

## 1. Core Logic Extraction & Integration Mapping

| Source File | Extracted Logic | Integration Point in APEX v2.0 |
| :--- | :--- | :--- |
| `integrated_synthesis_engine` | DEGF G-score (Sigmoid), UltraV3 | `Layer C` (Analyzer), `Layer B` (Synthesis) |
| `apex_v2.py` (v1) | Signal Decomp, Winsorized SNR | `Layer E` (Signal Detection) |
| `omega_v2` | Meta-Monitor, Self-Heal Loop | `Layer M` (Meta-DEGF), `Layer N` (Healing) |
| `discovery_engine_v5` | Routh-Hurwitz, Spectral Fingerprint | `Layer O` (Discovery Bridge), `Layer E` |
| `advanced_modules` | DDE/PDE Stability, Melnikov Chaos | `Layer O` (Advanced Mathematical Parsers) |
| `extensive_tuning_v3` | **New** Multi-Objective Optimizer | `Layer G` (Predictive), `Layer L` (WFCV) |

## 2. The Extensive Tuning Engine (v3.0)

The `ExtensiveTuningEngine` represents the pinnacle of "Thermodynamic Grounding." It replaces traditional grid search with a genuineness-weighted search:

$$Score = Q_{mean} + 0.15 \cdot G_{degf} - Penalty_{gaming}$$

- **Robustness**: Uses Recursive Grid Refinement to avoid local minima.
- **Effectiveness**: Regime-aware priors (Hurst/Spectral Entropy) narrow the search space to relevant domains.
- **Genuineness**: Anti-Gaming guards ensure that high performance is backed by genuine structural alignment with the data.

## 3. Structural Isomorphism
The system maintains the "Spine of DEGF" across three levels:
1. **Level 1**: Token Attention (Transformers)
2. **Level 2**: Dimension Attention (Vector Synthesis)
3. **Level 3**: Run Attention (Meta-Monitoring)

All levels use the same thermodynamic constants ($K_{deg}, \theta_c$) to maintain a consistent definition of "genuine reasoning."
