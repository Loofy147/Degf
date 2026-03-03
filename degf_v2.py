"""
DEGF v2 — Full Improvement Suite
==================================
Implements all 6 paper-identified improvements + 4 novel additions:

IMPROVEMENTS (from paper Section 7):
  IMP-1  Adaptive θ_c(t)             — position-scaled collapse threshold
  IMP-2  Q4 Entropy Plateau Simulator — stable-plateau context head model
  IMP-3  Empirical k_deg/k_rec Fitter — ODE least-squares fit to H_t data
  IMP-4  G Calibration Framework      — correlate G with synthetic quality proxy
  IMP-5  Windowed Collapse C_w        — sliding window C with exponential decay
  IMP-6  Cross-Layer V Correlation    — coordinated-circuit detection matrix

NEW ADDITIONS:
  NEW-1  G Stability Score            — per-head G variance across prompts
  NEW-2  MLP Interaction Proxy        — V boost from nonlinear MLP coupling
  NEW-3  Reasoning Quality Predictor  — calibrated G → quality linear model
  NEW-4  Circuit Cascade Detector     — collapse-event propagation across layers
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass, field
from typing import Optional
import warnings

# Import v1 core (we extend, not replace)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from degf_core import (
    compute_H_series, compute_V, compute_G, classify_quadrant,
    filter_genuine_diffuse, simulate_G_trajectory,
    HeadProfile, ModelScan, DEGFSimulator,
    K_DEG, K_REC, G_MAX, THETA_C, LAMBDA, GAMMA
)


# ═══════════════════════════════════════════════════════════════════════════════
# IMP-1  ADAPTIVE θ_c(t)
# Problem: fixed threshold over-counts collapses at t<10, under-counts at t>100
# Fix:     θ_c(t) = -0.20 / log₂(t + 2)  (scales down as sequence grows)
# ═══════════════════════════════════════════════════════════════════════════════

def adaptive_theta_c(t: int, base: float = -0.20) -> float:
    """Position-dependent collapse threshold."""
    return base / np.log2(t + 2)


def count_collapses_adaptive(H: np.ndarray, base_theta: float = -0.20) -> int:
    """
    Count Collapse Events with adaptive threshold per position.
    Each ΔH_t is compared to θ_c(t) = base / log₂(t+2).
    """
    if len(H) < 2:
        return 0
    count = 0
    for t in range(1, len(H)):
        theta = adaptive_theta_c(t, base_theta)
        if (H[t] - H[t-1]) < theta:
            count += 1
    return count


def compare_collapse_methods(H: np.ndarray) -> dict:
    """
    Compare fixed vs adaptive collapse counting.
    Returns dict with counts and per-position threshold comparison.
    """
    from degf_core import count_collapses
    C_fixed    = count_collapses(H)
    C_adaptive = count_collapses_adaptive(H)
    thresholds = np.array([adaptive_theta_c(t) for t in range(1, len(H))])
    diffs      = np.diff(H)
    return {
        "C_fixed"    : C_fixed,
        "C_adaptive" : C_adaptive,
        "delta_C"    : C_adaptive - C_fixed,
        "thresholds" : thresholds,     # (T-1,) adaptive values
        "diffs"      : diffs,          # (T-1,) ΔH values
        "improvement": abs(C_adaptive - C_fixed) / max(C_fixed, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# IMP-2  Q4 ENTROPY PLATEAU SIMULATOR
# Problem: context heads had monotone H → high V → misclassified as Q2/Q1
# Fix:     after burn-in, hold H near max-entropy plateau (V≈0, C=0, G≈0)
# ═══════════════════════════════════════════════════════════════════════════════

def context_head_attn_plateau(
    T: int,
    rng: np.random.Generator,
    burn_in: int = 10,
    plateau_noise_std: float = 0.005,
) -> np.ndarray:
    """
    Corrected context-broadcast head attention.
    Phase 1 (t < burn_in): attention grows normally (H increases monotonically).
    Phase 2 (t >= burn_in): attention is near-uniform, H plateaus at H_max(t)
                            with only tiny Gaussian noise.
    This produces V≈0, C=0, G≈0 → correctly maps to Q4 (MECHANICAL_DIFFUSE).
    """
    A = np.zeros((T, T))
    H_plateau = None

    for t in range(T):
        if t < burn_in:
            # Normal grow phase
            base = np.ones(t + 1)
            noise = rng.uniform(0.0, 0.02, size=t + 1)
            row = base + noise
            A[t, :t+1] = row / row.sum()
        else:
            # PLATEAU: generate uniform-ish attention and anchor to fixed H level
            # Dirichlet with high concentration → near-uniform, tiny variance
            alpha = np.ones(t + 1) * 50.0   # very high α → very uniform
            row = rng.dirichlet(alpha)
            A[t, :t+1] = row

    return A


def compute_V_detrended(H: np.ndarray, burn_in: int = 10) -> float:
    """
    Detrended Entropy Variance.
    For context-broadcast heads, H grows as log₂(t+1) simply because
    the uniform distribution over more keys has higher entropy.
    This growth is structural, not diagnostic.
    We subtract the expected growth curve and measure residual variance.

    V_detrended = var(H_t - log₂(t+1))  for t >= burn_in

    Returns ~0 for plateau heads, >0 for logic/pattern heads.
    """
    T = len(H)
    if T < burn_in + 2:
        return compute_V(H)
    t_idx   = np.arange(T)
    expected = np.log2(t_idx + 1.0)     # expected entropy for uniform attn
    detrended = H - expected
    return float(np.var(detrended[burn_in:]))


def plateau_head_G_score(T: int = 64, seed: int = 42) -> dict:
    """Compute G for a plateau head and verify Q4 classification."""
    rng = np.random.default_rng(seed)
    A   = context_head_attn_plateau(T, rng)
    H   = compute_H_series(A)
    V   = compute_V_detrended(H)           # ← detrended V for correct Q4
    C   = count_collapses_adaptive(H)
    G   = compute_G(V, C)
    tc  = 0.3   # context heads: low surprisal
    q, _ = classify_quadrant(tc, G)
    return {"V": V, "C": C, "G": G, "H_std": float(np.std(H[10:])), "quadrant": q}


# ═══════════════════════════════════════════════════════════════════════════════
# IMP-3  EMPIRICAL k_deg / k_rec FITTER
# Problem: k constants were set analytically, never fitted to data
# Fix:     scipy curve_fit on observed G trajectories from H_t series
# ═══════════════════════════════════════════════════════════════════════════════

def _degrade_model(t_arr: np.ndarray, k: float, G0: float) -> np.ndarray:
    """Analytical solution: G(t) = G0 * exp(-k * t)"""
    return G0 * np.exp(-k * t_arr)


def _recover_model(t_arr: np.ndarray, k: float, G0: float, Gmax: float = 1.0) -> np.ndarray:
    """Analytical solution: G(t) = Gmax - (Gmax-G0)*exp(-k*t)"""
    return Gmax - (Gmax - G0) * np.exp(-k * t_arr)


def fit_k_from_G_trajectory(G_observed: np.ndarray, mode: str = "degrade",
                            dt: float = 0.01) -> dict:
    """
    Fit k_deg or k_rec from an observed G trajectory using least-squares.

    Parameters
    ----------
    G_observed : observed G values over time
    mode       : "degrade" or "recover"
    dt         : time step size (default 0.01 to match ODE simulator)

    Returns
    -------
    dict with fitted k, G0, residual, and 95% confidence interval
    """
    t_arr = np.arange(len(G_observed), dtype=float) * dt   # ← use dt scale
    G0_guess = float(G_observed[0])

    try:
        if mode == "degrade":
            popt, pcov = curve_fit(
                lambda t, k: _degrade_model(t, k, G0_guess),
                t_arr, G_observed,
                p0=[0.8], bounds=([0.0], [10.0]),
                maxfev=5000,
            )
            k_fit = float(popt[0])
            k_std = float(np.sqrt(pcov[0, 0]))
            G_pred = _degrade_model(t_arr, k_fit, G0_guess)
        else:
            popt, pcov = curve_fit(
                lambda t, k: _recover_model(t, k, G0_guess),
                t_arr, G_observed,
                p0=[1.2], bounds=([0.0], [10.0]),
                maxfev=5000,
            )
            k_fit = float(popt[0])
            k_std = float(np.sqrt(pcov[0, 0]))
            G_pred = _recover_model(t_arr, k_fit, G0_guess)

        residuals = G_observed - G_pred
        rmse      = float(np.sqrt(np.mean(residuals**2)))

        return {
            "k_fitted"   : k_fit,
            "k_std"      : k_std,
            "k_ci_95"    : (k_fit - 1.96*k_std, k_fit + 1.96*k_std),
            "G0"         : G0_guess,
            "rmse"       : rmse,
            "r2"         : float(1.0 - np.var(residuals) / max(np.var(G_observed), 1e-12)),
            "mode"       : mode,
            "canonical"  : K_DEG if mode == "degrade" else K_REC,
            "vs_canonical": abs(k_fit - (K_DEG if mode == "degrade" else K_REC)),
        }

    except Exception as e:
        return {"error": str(e), "mode": mode}


def fit_constants_from_scan(scan: ModelScan) -> dict:
    """
    Fit k_deg and k_rec from all degrading/recovering head G series in a scan.
    Degrade heads: H_t has overall downward trend in G.
    Recover heads: H_t has overall upward trend in G.
    """
    degrade_Gs, recover_Gs = [], []

    for p in scan.profiles:
        H = p.entropy_series
        if len(H) < 10:
            continue
        # Compute rolling G using 8-token windows
        window = 8
        G_series = []
        for t in range(window, len(H), 1):
            H_window = H[t-window:t]
            V_w = float(np.var(H_window))
            C_w = count_collapses_adaptive(H_window)
            G_series.append(compute_G(V_w, C_w))

        if len(G_series) < 4:
            continue

        G_arr  = np.array(G_series)
        trend  = np.polyfit(np.arange(len(G_arr)), G_arr, 1)[0]

        if trend < -0.001:
            degrade_Gs.append(G_arr)
        elif trend > 0.001:
            recover_Gs.append(G_arr)

    results = {}

    if degrade_Gs:
        # Average over all degrading heads, fit once
        max_len = min(len(g) for g in degrade_Gs)
        G_mean_deg = np.mean([g[:max_len] for g in degrade_Gs], axis=0)
        results["degrade"] = fit_k_from_G_trajectory(G_mean_deg, "degrade")
        results["n_degrade_heads"] = len(degrade_Gs)
    else:
        results["degrade"] = {"error": "no degrading heads found"}

    if recover_Gs:
        max_len = min(len(g) for g in recover_Gs)
        G_mean_rec = np.mean([g[:max_len] for g in recover_Gs], axis=0)
        results["recover"] = fit_k_from_G_trajectory(G_mean_rec, "recover")
        results["n_recover_heads"] = len(recover_Gs)
    else:
        results["recover"] = {"error": "no recovering heads found"}

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# IMP-4  G CALIBRATION FRAMEWORK
# Problem: G has no external standard — 0.7 means nothing absolute
# Fix:     calibrate G against synthetic reasoning quality proxy;
#          derive linear transform G → Q_hat (quality estimate)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_calibration_dataset(n: int = 1000, seed: int = 7) -> dict:
    """
    Generate synthetic (G, quality) pairs for calibration.
    Quality proxy rules (based on mechanistic priors):
      - High G (>0.7) + Low token cost   → high quality (logic-heavy reasoning)
      - High G (>0.7) + High token cost  → medium quality (committed but rigid)
      - Low  G (<0.3) + any cost         → low quality (pattern completion)
      - Mid  G (0.3-0.7)                 → quality ≈ 0.5 + noise
    """
    rng = np.random.default_rng(seed)
    G   = rng.uniform(0.0, 1.0, n)
    tc  = rng.uniform(0.0, 1.0, n)

    # Deterministic quality with Gaussian noise
    quality = np.zeros(n)
    for i in range(n):
        if G[i] > 0.7 and tc[i] < 0.5:
            quality[i] = 0.75 + 0.20 * G[i] + rng.normal(0, 0.08)
        elif G[i] > 0.7 and tc[i] >= 0.5:
            quality[i] = 0.50 + 0.15 * G[i] + rng.normal(0, 0.10)
        elif G[i] < 0.3:
            quality[i] = 0.10 + 0.40 * G[i] + rng.normal(0, 0.06)
        else:
            quality[i] = 0.40 + 0.20 * G[i] + 0.05 * tc[i] + rng.normal(0, 0.12)
        quality[i] = float(np.clip(quality[i], 0.0, 1.0))

    return {"G": G, "token_cost": tc, "quality": quality}


def calibrate_G(cal_data: dict) -> dict:
    """
    Fit a linear calibration model: Q_hat = a*G + b*tc + c.
    Returns coefficients and fit statistics.
    """
    G  = cal_data["G"]
    tc = cal_data["token_cost"]
    Q  = cal_data["quality"]
    n  = len(G)

    # Build design matrix [G, tc, 1]
    X = np.column_stack([G, tc, np.ones(n)])
    # OLS: β = (X'X)^-1 X'y
    try:
        beta, residuals_sq, rank, sv = np.linalg.lstsq(X, Q, rcond=None)
    except Exception as e:
        return {"error": str(e)}

    Q_hat = X @ beta
    residuals = Q - Q_hat
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((Q - Q.mean())**2))
    r2  = 1.0 - ss_res / max(ss_tot, 1e-12)
    r_pearson, p_val = pearsonr(G, Q)
    r_spearman, _    = spearmanr(G, Q)

    return {
        "a_G"          : float(beta[0]),
        "b_tc"         : float(beta[1]),
        "c_intercept"  : float(beta[2]),
        "r2"           : float(r2),
        "pearson_r"    : float(r_pearson),
        "spearman_r"   : float(r_spearman),
        "p_value"      : float(p_val),
        "rmse"         : float(np.sqrt(ss_res / n)),
        "n_samples"    : n,
        "interpretation": _interpret_calibration(float(r2), float(r_pearson)),
    }


def _interpret_calibration(r2: float, r: float) -> str:
    if r2 > 0.85: return "Strong calibration — G reliably predicts quality"
    if r2 > 0.60: return "Moderate calibration — G useful with caveats"
    if r2 > 0.40: return "Weak calibration — G is partially informative"
    return "Poor calibration — G needs additional features"


def predict_quality(G: float, tc: float, cal_result: dict) -> float:
    """Apply calibration coefficients to predict quality from (G, token_cost)."""
    q = cal_result["a_G"] * G + cal_result["b_tc"] * tc + cal_result["c_intercept"]
    return float(np.clip(q, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# IMP-5  WINDOWED COLLAPSE C_w  (exponential decay weighting)
# Problem: C counts all collapses equally; recent ones are more decision-relevant
# Fix:     C_w = Σ exp(-λ*(T-t)) * I[ΔH_t < θ_c(t)]
# ═══════════════════════════════════════════════════════════════════════════════

def count_collapses_weighted(
    H: np.ndarray,
    decay_lambda: float = 0.1,
    use_adaptive_theta: bool = True,
) -> float:
    """
    Exponentially decay-weighted collapse count.
    Recent collapses are weighted exp(0) = 1.0; early collapses decay toward 0.
    Returns a float (not int) since weights are continuous.
    """
    if len(H) < 2:
        return 0.0

    T    = len(H)
    C_w  = 0.0
    for t in range(1, T):
        theta = adaptive_theta_c(t) if use_adaptive_theta else THETA_C
        if (H[t] - H[t-1]) < theta:
            weight = np.exp(-decay_lambda * (T - t))
            C_w   += weight
    return float(C_w)


def compute_G_v2(
    H: np.ndarray,
    use_adaptive_theta: bool = True,
    use_weighted_C: bool = True,
) -> dict:
    """
    Enhanced G computation using all v2 improvements.
    Returns full diagnostic dict instead of scalar.
    """
    V      = compute_V(H)
    C_raw  = count_collapses_adaptive(H) if use_adaptive_theta else \
             __import__('degf_core').count_collapses(H)
    C_w    = count_collapses_weighted(H, use_adaptive_theta=use_adaptive_theta)
    G_old  = compute_G(V, C_raw)                  # v1 baseline
    G_new  = compute_G(V, C_w)                    # v2 with weighted C

    return {
        "V"      : V,
        "C_raw"  : C_raw,
        "C_w"    : C_w,
        "G_v1"   : G_old,
        "G_v2"   : G_new,
        "delta_G": G_new - G_old,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# IMP-6  CROSS-LAYER V CORRELATION MATRIX
# Problem: heads analysed in isolation — coordinated circuits invisible
# Fix:     compute Pearson(V_l1, V_l2) for all layer pairs → detect circuits
# ═══════════════════════════════════════════════════════════════════════════════

def cross_layer_V_matrix(scan: ModelScan) -> np.ndarray:
    """
    Build an (n_layers × n_layers) correlation matrix of per-layer mean V.
    Entry [i,j] = Pearson correlation of the V-vectors of layers i and j,
    where each layer's V-vector has one entry per head.

    High positive correlation between layers i and j suggests coordinated
    reasoning circuits spanning those layers.
    """
    n_layers = scan.n_layers
    # Build layer → V_vector mapping
    layer_V: dict[int, list] = {l: [] for l in range(n_layers)}
    for p in scan.profiles:
        layer_V[p.layer].append(p.V)

    # Layers that have data
    layers_with_data = sorted(k for k, v in layer_V.items() if len(v) > 0)
    n = len(layers_with_data)
    corr_matrix = np.eye(n)

    for i, l1 in enumerate(layers_with_data):
        for j, l2 in enumerate(layers_with_data):
            if i == j:
                continue
            v1 = np.array(layer_V[l1])
            v2 = np.array(layer_V[l2])
            min_len = min(len(v1), len(v2))
            if min_len < 3:
                continue
            try:
                r, _ = pearsonr(v1[:min_len], v2[:min_len])
                corr_matrix[i, j] = float(r) if not np.isnan(r) else 0.0
            except Exception:
                pass

    return corr_matrix, layers_with_data


def find_coordinated_circuits(
    corr_matrix: np.ndarray,
    layers: list,
    threshold: float = 0.70,
) -> list[tuple]:
    """
    Return list of (layer_i, layer_j, correlation) for pairs above threshold.
    These are candidate coordinated reasoning circuits.
    """
    n = corr_matrix.shape[0]
    circuits = []
    for i in range(n):
        for j in range(i+1, n):
            r = corr_matrix[i, j]
            if abs(r) >= threshold:
                circuits.append((layers[i], layers[j], float(r)))
    return sorted(circuits, key=lambda x: -abs(x[2]))


# ═══════════════════════════════════════════════════════════════════════════════
# NEW-1  G STABILITY SCORE  (multi-prompt variance)
# Idea:   a head's quadrant should be stable across diverse inputs
#         if it is genuinely specialised; instability → noise/overlap
# ═══════════════════════════════════════════════════════════════════════════════

def compute_G_stability(
    G_across_prompts: np.ndarray,
) -> dict:
    """
    Given G scores for the same head across P different prompts,
    compute stability metrics.

    Parameters
    ----------
    G_across_prompts : (P,) array of G scores

    Returns
    -------
    dict: mean_G, std_G, stability_score, classification
    """
    mean_G = float(np.mean(G_across_prompts))
    std_G  = float(np.std(G_across_prompts))
    cv     = std_G / max(mean_G, 1e-6)     # coefficient of variation

    # Stability score: 1 = perfectly stable, 0 = maximally variable
    stability = float(np.clip(1.0 - cv, 0.0, 1.0))

    if stability > 0.85:
        label = "STABLE_SPECIALIST"
    elif stability > 0.60:
        label = "SEMI_STABLE"
    else:
        label = "UNSTABLE_GENERALIST"

    return {
        "mean_G"   : mean_G,
        "std_G"    : std_G,
        "cv"       : cv,
        "stability": stability,
        "label"    : label,
        "n_prompts": len(G_across_prompts),
    }


def scan_G_stability(
    simulator: DEGFSimulator,
    layer: int,
    head: int,
    n_prompts: int = 20,
) -> dict:
    """
    Simulate G stability for a single (layer, head) across n_prompts.
    Each 'prompt' uses a fresh RNG seed → different attention realisations.
    """
    G_list = []
    archetype = "name_mover" if layer >= int(0.65 * simulator.n_layers) \
                and head % 3 != 0 else "induction"

    for seed_offset in range(n_prompts):
        sim2 = DEGFSimulator(
            simulator.n_layers, simulator.n_heads, simulator.seq_len
        )
        sim2.rng = np.random.default_rng(seed_offset * 13 + layer * 7 + head)

        if archetype == "name_mover":
            A = sim2._name_mover_attn()
        else:
            A = sim2._induction_head_attn()

        H  = compute_H_series(A)
        Cv = count_collapses_adaptive(H)
        V  = compute_V(H)
        G_list.append(compute_G(V, Cv))

    return compute_G_stability(np.array(G_list))


# ═══════════════════════════════════════════════════════════════════════════════
# NEW-2  MLP INTERACTION PROXY
# Idea:   Attention heads don't operate alone — MLPs amplify or suppress
#         the residual stream post-attention. We model this as a V-multiplier
#         dependent on layer depth (deep MLPs have stronger nonlinear effects).
# ═══════════════════════════════════════════════════════════════════════════════

def mlp_interaction_factor(
    layer: int,
    n_layers: int,
    head_G: float,
    mlp_scale: float = 1.8,
) -> float:
    """
    Compute MLP interaction factor for a head at given layer.
    Deep layers have stronger MLP amplification (sigmoid curve).
    High-G heads get amplified more (MLP follows attention routing).

    Returns V_effective multiplier ∈ [1.0, mlp_scale].
    """
    depth_ratio  = layer / max(n_layers - 1, 1)
    depth_factor = 1.0 / (1.0 + np.exp(-8.0 * (depth_ratio - 0.6)))
    G_factor     = head_G                       # high-G → more MLP coupling
    multiplier   = 1.0 + (mlp_scale - 1.0) * depth_factor * G_factor
    return float(multiplier)


def compute_G_with_mlp(
    H: np.ndarray,
    layer: int,
    n_layers: int,
) -> dict:
    """Compute G_v2 then apply MLP interaction multiplier to effective V."""
    V  = compute_V(H)
    C  = count_collapses_adaptive(H)
    G  = compute_G(V, C)

    mlp_mult = mlp_interaction_factor(layer, n_layers, G)
    V_eff    = V * mlp_mult
    G_mlp    = compute_G(V_eff, C)

    return {
        "V_base"    : V,
        "V_eff"     : V_eff,
        "mlp_factor": mlp_mult,
        "G_no_mlp"  : G,
        "G_with_mlp": G_mlp,
        "G_boost"   : G_mlp - G,
        "layer"     : layer,
        "n_layers"  : n_layers,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NEW-3  REASONING QUALITY PREDICTOR (calibrated, multi-feature)
# Idea:   combine G, token_cost, C_w, stability, MLP factor into a
#         single calibrated quality estimate using linear regression
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityModel:
    """Fitted reasoning quality prediction model."""
    weights: np.ndarray     # [w_G, w_tc, w_Cw, w_stab, w_mlp, bias]
    feature_names: list
    r2: float
    rmse: float
    fitted: bool = False

    def predict(self, G: float, tc: float, C_w: float,
                stability: float, mlp_factor: float) -> float:
        if not self.fitted:
            return G  # fallback
        x = np.array([G, tc, C_w, stability, mlp_factor, 1.0])
        return float(np.clip(self.weights @ x, 0.0, 1.0))


def train_quality_model(n_samples: int = 2000, seed: int = 42) -> QualityModel:
    """
    Generate synthetic multi-feature training data and fit a quality predictor.
    Features: G, token_cost, C_w, G_stability, MLP_factor
    Target: quality (synthetic but consistent with DEGF theory)
    """
    rng  = np.random.default_rng(seed)
    n    = n_samples

    G         = rng.uniform(0.0, 1.0, n)
    tc        = rng.uniform(0.0, 1.0, n)
    C_w       = rng.uniform(0.0, 5.0, n)
    stability = rng.uniform(0.0, 1.0, n)
    mlp_f     = rng.uniform(1.0, 1.8, n)

    # Quality: theory-consistent ground truth
    quality = (
        0.40 * G +
        (-0.15) * tc +          # high surprisal → lower quality
        0.10 * np.log1p(C_w) +  # more collapses → more commits → better
        0.15 * stability +       # stable specialist heads → better
        0.10 * (mlp_f - 1.0) +  # MLP amplification → better
        0.05 +                   # intercept
        rng.normal(0, 0.05, n)
    )
    quality = np.clip(quality, 0.0, 1.0)

    # OLS fit
    X    = np.column_stack([G, tc, C_w, stability, mlp_f, np.ones(n)])
    beta, _, _, _ = np.linalg.lstsq(X, quality, rcond=None)

    Q_hat     = X @ beta
    residuals = quality - Q_hat
    ss_res    = float(np.sum(residuals**2))
    ss_tot    = float(np.sum((quality - quality.mean())**2))
    r2        = 1.0 - ss_res / max(ss_tot, 1e-12)
    rmse      = float(np.sqrt(ss_res / n))

    return QualityModel(
        weights=beta,
        feature_names=["G", "token_cost", "C_w", "stability", "mlp_factor", "bias"],
        r2=float(r2),
        rmse=rmse,
        fitted=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# NEW-4  CIRCUIT CASCADE DETECTOR
# Idea:   a reasoning circuit isn't just high-G heads — it's heads where
#         collapse events happen in sequence across layers (causal chain)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CascadeChain:
    """A sequence of (layer, head) pairs with temporally linked collapse events."""
    links: list[tuple]      # [(layer, head, collapse_time), ...]
    length: int
    avg_G: float
    total_delay: int        # total token-lag across the chain
    strength: float         # mean G of chain members

    def __str__(self):
        chain_str = " → ".join(f"L{l}H{h}@t{t}" for l,h,t in self.links)
        return f"Chain[{self.length}] {chain_str} (strength={self.strength:.2f})"


def detect_collapse_times(H: np.ndarray, use_adaptive: bool = True) -> list[int]:
    """Return token positions where collapse events occur."""
    times = []
    for t in range(1, len(H)):
        theta = adaptive_theta_c(t) if use_adaptive else THETA_C
        if (H[t] - H[t-1]) < theta:
            times.append(t)
    return times


def detect_cascade_chains(
    scan: ModelScan,
    max_lag: int = 3,
    min_chain_length: int = 2,
    min_G: float = 0.50,
) -> list[CascadeChain]:
    """
    Detect cascades: sequences of collapse events propagating through layers.
    A cascade exists when layer L+k (k ≤ max_lag) has a collapse within
    max_lag tokens after layer L — suggesting information flow.

    Returns sorted list of CascadeChain objects.
    """
    # Build per-head collapse time list
    head_collapses: dict[tuple, list] = {}
    head_G: dict[tuple, float] = {}

    for p in scan.profiles:
        if p.G < min_G:
            continue
        times = detect_collapse_times(p.entropy_series)
        if times:
            key = (p.layer, p.head)
            head_collapses[key] = times
            head_G[key] = p.G

    # Sort by layer
    sorted_heads = sorted(head_collapses.keys(), key=lambda x: x[0])

    chains = []
    visited = set()

    for i, (l1, h1) in enumerate(sorted_heads):
        if (l1, h1) in visited:
            continue

        chain_links = []
        for t1 in head_collapses[(l1, h1)]:
            current_chain = [(l1, h1, t1)]

            # Try to extend chain forward through layers
            for j, (l2, h2) in enumerate(sorted_heads):
                if l2 <= l1:
                    continue
                if l2 - l1 > 4:   # max layer gap
                    break
                for t2 in head_collapses[(l2, h2)]:
                    if 0 <= t2 - t1 <= max_lag:
                        current_chain.append((l2, h2, t2))
                        break

            if len(current_chain) >= min_chain_length:
                chain_links.append(current_chain)

        if chain_links:
            # Pick longest chain
            best = max(chain_links, key=len)
            lhts = best
            avg_G  = float(np.mean([head_G.get((l,h), 0.0) for l,h,_ in lhts]))
            t_total = lhts[-1][2] - lhts[0][2] if len(lhts) > 1 else 0
            chains.append(CascadeChain(
                links    = lhts,
                length   = len(lhts),
                avg_G    = avg_G,
                total_delay = t_total,
                strength = avg_G,
            ))
            for l,h,_ in lhts:
                visited.add((l,h))

    return sorted(chains, key=lambda c: (-c.length, -c.strength))


# ═══════════════════════════════════════════════════════════════════════════════
# V2 SIMULATOR — extends DEGFSimulator with plateau heads
# ═══════════════════════════════════════════════════════════════════════════════

class DEGFSimulatorV2(DEGFSimulator):
    """
    Extended simulator with correct Q4 plateau heads and MLP-augmented G scores.
    """

    def _context_head_attn(self) -> np.ndarray:
        """Override: use plateau model for correct Q4 mapping."""
        return context_head_attn_plateau(self.seq_len, self.rng)

    def scan_v2(self, layer_range: tuple = None) -> ModelScan:
        """Full scan using all v2 improvements."""
        if layer_range is None:
            layer_range = (0, self.n_layers)

        scan = ModelScan(n_layers=self.n_layers, n_heads=self.n_heads)

        for l in range(*layer_range):
            for h in range(self.n_heads):
                A  = self.generate_attention(l, h)
                H  = compute_H_series(A)
                tc = self.simulate_token_cost(l, h)

                # V2: use detrended V for early (context) layers
                is_context = l < int(0.65 * self.n_layers)
                V = compute_V_detrended(H) if is_context else compute_V(H)

                C_a  = count_collapses_adaptive(H)
                C_w  = count_collapses_weighted(H)
                G_v2 = compute_G(V, C_a)

                profile = HeadProfile(
                    layer=l, head=h,
                    entropy_series=H,
                    token_cost=tc,
                )
                # Override with v2 values
                object.__setattr__(profile, 'V', V)
                object.__setattr__(profile, 'G', G_v2)
                q, interv = classify_quadrant(tc, G_v2)
                object.__setattr__(profile, 'quadrant', q)
                object.__setattr__(profile, 'intervention', interv)

                scan.profiles.append(profile)

        scan.targets_genuine_diffuse    = filter_genuine_diffuse(scan.profiles)
        scan.targets_mechanical_committed = [
            (p.layer, p.head) for p in scan.profiles if "Q3" in p.quadrant
        ]
        return scan
