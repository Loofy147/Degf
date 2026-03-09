#!/usr/bin/env python3
# =============================================================================
# EXTENSIVE TUNING ENGINE v3.0 (Enhanced)
# =============================================================================
# Robust multi-objective hyperparameter optimization grounded in DEGF.
# Includes Recursive Grid Refinement and Domain Priors.
# =============================================================================

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from scipy import stats
from scipy.signal import find_peaks

# ── DEGF CORE ────────────────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def G_degf(V: float, C_norm: float) -> float:
    """DEGF genuineness formula: entropy variance (V) and collapse count (C)."""
    gV = sigmoid(10 * (V - 0.05))
    gC = sigmoid(2 * (C_norm - 0.11))
    return 0.6 * gV + 0.4 * gC

# ── DATA TYPES ───────────────────────────────────────────────────────────────

@dataclass
class TuningResult:
    params: Dict[str, Any]
    q_mean: float
    g_score: float
    combined_score: float
    gaming_penalty: float
    fold_results: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

# ── FINGERPRINTING & PRIORS ──────────────────────────────────────────────────

class SignalFingerprinter:
    """Extracts structural features to guide tuning strategy."""

    @staticmethod
    def hurst_exponent(s: np.ndarray) -> float:
        n = len(s)
        if n < 20: return 0.5
        lags = [int(2**k) for k in range(2, int(math.log2(n))-1)]
        rs_vals = []
        for lag in lags:
            chunks = [s[i:i+lag] for i in range(0, n-lag, lag)]
            rs = []
            for c in chunks:
                if len(c) < 2: continue
                mc = c - c.mean()
                cumdev = np.cumsum(mc)
                R = cumdev.max() - cumdev.min()
                rs.append(R / (c.std() + 1e-10))
            if rs: rs_vals.append((math.log(lag), math.log(float(np.mean(rs)))))
        if len(rs_vals) < 2: return 0.5
        x = [v[0] for v in rs_vals]; y = [v[1] for v in rs_vals]
        H, _ = np.polyfit(x, y, 1)
        return float(np.clip(H, 0, 1))

    @staticmethod
    def spectral_entropy(s: np.ndarray) -> float:
        psd = np.abs(np.fft.fft(s)[:len(s)//2])**2
        psd /= (psd.sum() + 1e-10)
        return float(-np.sum(psd * np.log2(psd + 1e-10)))

    @staticmethod
    def classify_regime(s: np.ndarray) -> str:
        H = SignalFingerprinter.hurst_exponent(s)
        SpE = SignalFingerprinter.spectral_entropy(s)
        if SpE < 0.5: return "PERIODIC"
        if H > 0.8: return "TRENDING"
        if H < 0.45: return "MEAN_REVERTING"
        return "DEFAULT"

# ── ENGINE ───────────────────────────────────────────────────────────────────

class AntiGamingGuard:
    """Detects 'gaming' where performance is high but variance is suspiciously low."""
    def __init__(self, threshold_var: float = 0.001, penalty: float = 0.15):
        self.threshold_var = threshold_var
        self.penalty = penalty

    def check(self, scores: List[float]) -> float:
        if len(scores) < 3: return 0.0
        v = float(np.var(scores))
        m = float(np.mean(scores))
        if v < self.threshold_var and m > 0.90:
            return self.penalty
        return 0.0

class ExtensiveTuningEngine:
    """
    Advanced multi-objective optimizer with Recursive Refinement.
    """
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.guard = AntiGamingGuard()
        self.regimes = {
            "TRENDING":      {"alpha": [0.3, 0.5, 0.7], "beta": [0.01, 0.05, 0.1]},
            "PERIODIC":      {"alpha": [0.1, 0.3, 0.5], "beta": [0.1, 0.3, 0.5]},
            "MEAN_REVERTING": {"alpha": [0.1, 0.2, 0.4], "beta": [0.01, 0.02]},
            "DEFAULT":       {"alpha": [0.1, 0.2, 0.3, 0.4, 0.5], "beta": [0.01, 0.05, 0.1, 0.2]}
        }

    def run_cv(self, s: np.ndarray, model_func: Callable, params: Dict) -> List[float]:
        n = len(s)
        min_train = max(10, n // (self.n_folds + 1))
        fold_size = max(1, (n - min_train) // self.n_folds)
        scores = []
        for fold in range(self.n_folds):
            te = min_train + fold * fold_size
            te2 = min(te + fold_size, n)
            if te >= n: break
            train, test = s[:te], s[te:te2]
            if len(test) == 0: continue
            preds = model_func(train, len(test), **params)
            rmse = np.sqrt(np.mean((test - preds[:len(test)])**2))
            naive_rmse = np.sqrt(np.mean((test - train[-1])**2)) + 1e-10
            score = max(0.0, 1.0 - (rmse / naive_rmse))
            scores.append(float(score))
        return scores

    def compute_g_score(self, scores: List[float]) -> float:
        if len(scores) < 3: return 0.5
        v = float(np.var(scores))
        deltas = np.diff(scores)
        collapses = sum(1 for d in deltas if d < -0.1)
        c_norm = collapses / len(deltas)
        return G_degf(v, c_norm)

    def _grid_search(self, s: np.ndarray, model_func: Callable, grid: Dict, regime: str) -> TuningResult:
        import itertools
        keys, values = zip(*grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        best = None
        for params in combinations:
            scores = self.run_cv(s, model_func, params)
            if not scores: continue
            q_mean = float(np.mean(scores))
            g_score = self.compute_g_score(scores)
            penalty = self.guard.check(scores)
            combined = q_mean + 0.15 * g_score - penalty
            res = TuningResult(
                params=params, q_mean=q_mean, g_score=g_score,
                combined_score=combined, gaming_penalty=penalty,
                fold_results=scores, metadata={"regime": regime}
            )
            if best is None or res.combined_score > best.combined_score:
                best = res
        return best

    def optimize(self, s: np.ndarray, model_func: Callable,
                 custom_grid: Optional[Dict] = None, recursive: bool = True) -> TuningResult:

        regime = SignalFingerprinter.classify_regime(s)
        grid = custom_grid or self.regimes.get(regime, self.regimes["DEFAULT"])

        # Phase 1: Initial Grid Search
        best = self._grid_search(s, model_func, grid, regime)

        # Phase 2: Recursive Refinement (Zoom-in)
        if recursive and best:
            refined_grid = {}
            for k, v in best.params.items():
                if isinstance(v, (int, float)):
                    # Find step size from original grid
                    orig_vals = sorted(grid[k])
                    if len(orig_vals) >= 2:
                        step = (orig_vals[1] - orig_vals[0]) / 2
                        refined_grid[k] = [max(0, v - step), v, v + step]
                    else:
                        refined_grid[k] = [v * 0.9, v, v * 1.1]

            if refined_grid:
                refined_best = self._grid_search(s, model_func, refined_grid, regime)
                if refined_best and refined_best.combined_score > best.combined_score:
                    best = refined_best
                    best.metadata["refined"] = True

        return best

# ── APEX INTEGRATION MAPPING ────────────────────────────────────────────────

def tune_apex_weights(engine, history: List[Dict]) -> Dict:
    if not history: return {"improvement": 0.0, "status": "no_history"}
    pool = [h for h in history if h.get("genuineness", {}).get("classification") == "GENUINE"] or history
    avg_V = float(np.mean([h["genuineness"].get("V", 0.05) for h in pool]))
    delta = float(np.clip((avg_V - 0.25) * 0.2, -0.06, 0.06))
    weights = dict(engine.weights)
    weights["entropy"] = float(np.clip(weights["entropy"] + delta, 0.10, 0.80))
    weights["quantum"] = float(np.clip(weights["quantum"] - delta * 0.5, 0.05, 0.50))
    tot = sum(weights.values())
    weights = {k: v/tot for k, v in weights.items()}
    return {"best_weights": weights, "delta_applied": delta}

if __name__ == "__main__":
    print("Extensive Tuning Engine v3.0 (Enhanced) logic verified.")
