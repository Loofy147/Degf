import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Constants from paper
K_DEG = 0.8129
K_REC = 1.2371
G_MAX = 1.0
THETA_C = -0.20
LAMBDA = 0.05
GAMMA = 0.30

def compute_H_series(A: np.ndarray) -> np.ndarray:
    """Compute Shannon Entropy for each row in attention matrix A."""
    A_safe = np.where(A > 1e-12, A, 1.0)
    h = -np.sum(A * np.log2(A_safe), axis=-1)
    return h

def compute_V(H: np.ndarray) -> float:
    """Dynamic Entropy Variance."""
    if len(H) < 2: return 0.0
    return float(np.var(H))

def compute_V_detrended(H: np.ndarray, burn_in: int = 10) -> float:
    T = len(H)
    if T < burn_in + 2:
        return compute_V(H)
    t_idx = np.arange(T)
    expected = np.log2(t_idx + 1.0)
    detrended = H - expected
    return float(np.var(detrended[burn_in:]))

def count_collapses(H: np.ndarray, theta: float = THETA_C) -> int:
    """Count Collapse Events where ΔH < theta."""
    if len(H) < 2: return 0
    diffs = np.diff(H)
    return int(np.sum(diffs < theta))

def compute_G(V: float, C: float) -> float:
    return 1.0 / (1.0 + np.exp(-(V + 0.5 * C - 1.2)))

def classify_quadrant(token_cost: float, G: float) -> Tuple[str, str]:
    if G >= 0.5:
        if token_cost >= 0.5:
            return ("Q1: GENUINE_COMMITTED", "INTERVENE: NONE")
        else:
            return ("Q2: GENUINE_DIFFUSE", "INTERVENE: AMPLIFY")
    else:
        if token_cost >= 0.5:
            return ("Q3: MECHANICAL_COMMITTED", "INTERVENE: CLAMP")
        else:
            return ("Q4: MECHANICAL_DIFFUSE", "INTERVENE: MONITOR")

def filter_genuine_diffuse(profiles: List['HeadProfile']) -> List[Tuple[int, int]]:
    return [(p.layer, p.head) for p in profiles if p.V > 0.10 and p.C >= 1]

def simulate_G_trajectory(G0: float, steps: int, mode: str = "degrade", dt: float = 0.01) -> np.ndarray:
    G = np.zeros(steps)
    G[0] = G0
    for t in range(1, steps):
        if mode == "degrade":
            dG = -K_DEG * G[t-1]
        else:
            dG = K_REC * (G_MAX - G[t-1])
        G[t] = np.clip(G[t-1] + dG * dt, 0, 1)
    return G

@dataclass
class HeadProfile:
    layer: int
    head: int
    entropy_series: np.ndarray
    token_cost: float
    V: Optional[float] = None
    C: Optional[int] = None
    G: Optional[float] = None
    quadrant: str = ""
    intervention: str = ""
    use_detrended: bool = False

    def __post_init__(self):
        if self.V is None:
            if self.use_detrended:
                self.V = compute_V_detrended(self.entropy_series)
            else:
                self.V = compute_V(self.entropy_series)
        if self.C is None:
            self.C = count_collapses(self.entropy_series)
        if self.G is None:
            self.G = compute_G(self.V, self.C)
        if not self.quadrant:
            self.quadrant, self.intervention = classify_quadrant(self.token_cost, self.G)

@dataclass
class ModelScan:
    n_layers: int
    n_heads: int
    profiles: List[HeadProfile] = field(default_factory=list)
    targets_genuine_diffuse: List[Tuple[int, int]] = field(default_factory=list)
    targets_mechanical_committed: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def summary(self) -> dict:
        q_counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
        for p in self.profiles:
            q_counts[p.quadrant[:2]] += 1
        return {
            "total_heads": len(self.profiles),
            "mean_G": np.mean([p.G for p in self.profiles]) if self.profiles else 0,
            "mean_V": np.mean([p.V for p in self.profiles]) if self.profiles else 0,
            "mean_C": np.mean([p.C for p in self.profiles]) if self.profiles else 0,
            "quadrant_counts": q_counts,
            "genuine_diffuse_targets": len(self.targets_genuine_diffuse),
            "mech_committed_targets": len(self.targets_mechanical_committed)
        }

class DEGFSimulator:
    def __init__(self, n_layers: int, n_heads: int, seq_len: int):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.rng = np.random.default_rng(42)

    def _induction_head_attn(self) -> np.ndarray:
        A = np.zeros((self.seq_len, self.seq_len))
        for t in range(self.seq_len):
            A[t, t] = 1.0
        return A

    def _name_mover_attn(self) -> np.ndarray:
        A = np.zeros((self.seq_len, self.seq_len))
        for t in range(self.seq_len):
            if t > 0 and self.rng.random() < 0.25:
                idx = self.rng.integers(0, t + 1)
                A[t, idx] = 0.98
                if t > 0:
                    rem = 0.02 / t
                    for i in range(t + 1):
                        if i != idx: A[t, i] = rem
                else:
                    A[t, 0] = 1.0
            else:
                row = self.rng.uniform(0.1, 1, t + 1)
                A[t, :t+1] = row / row.sum()
        return A

    def _context_head_attn(self) -> np.ndarray:
        A = np.zeros((self.seq_len, self.seq_len))
        for t in range(self.seq_len):
            A[t, :t+1] = 1.0 / (t + 1)
        return A

    def generate_attention(self, layer: int, head: int) -> np.ndarray:
        # DEGF_v1 Simulator had a different target selection logic.
        # To maintain Q4=0 in V1, we need to return something else here.
        # Original V1 code:
        if layer >= int(0.65 * self.n_layers) and head % 3 != 0:
            return self._name_mover_attn()
        elif head % 3 == 0:
            return self._induction_head_attn()
        else:
            # Context heads in V1: monotone entropy growth => V high, G high.
            # In V1 this resulted in Q2, not Q4.
            return self._context_head_attn()

    def simulate_token_cost(self, layer: int, head: int) -> float:
        if head == 1:
            return 0.85
        if head % 3 == 0:
            return self.rng.uniform(0.6, 0.9)
        return self.rng.uniform(0.1, 0.4)

    def scan(self) -> ModelScan:
        scan = ModelScan(n_layers=self.n_layers, n_heads=self.n_heads)
        for l in range(self.n_layers):
            for h in range(self.n_heads):
                A = self.generate_attention(l, h)
                H = compute_H_series(A)
                tc = self.simulate_token_cost(l, h)
                # V1 Scan didn't use detrended V.
                p = HeadProfile(layer=l, head=h, entropy_series=H, token_cost=tc, use_detrended=False)
                scan.profiles.append(p)

        scan.targets_genuine_diffuse = filter_genuine_diffuse(scan.profiles)
        scan.targets_mechanical_committed = [(p.layer, p.head) for p in scan.profiles if "Q3" in p.quadrant]
        return scan
