import numpy as np
from degf_core import compute_H_series, compute_V, count_collapses, compute_G

def test_metrics():
    # Simple attention: 100% focus on one token
    attn = np.array([[1.0, 0.0], [0.0, 1.0]])
    H = compute_H_series(attn)
    print(f"H: {H}") # Should be 0.0

    V = compute_V(H)
    print(f"V: {V}") # Should be 0.0

    C = count_collapses(H)
    print(f"C: {C}") # Should be 0

    G = compute_G(V, C)
    print(f"G: {G:.4f}") # 1/(1+exp(1.2)) = 0.2315

    # Diffuse attention
    attn2 = np.array([[0.5, 0.5], [0.5, 0.5]])
    H2 = compute_H_series(attn2)
    print(f"H2: {H2}") # Should be 1.0

    V2 = compute_V(H2)
    print(f"V2: {V2}") # Should be 0.0

    # Case with shift
    H3 = np.array([1.0, 0.5, 0.2, 0.8])
    V3 = compute_V(H3)
    C3 = count_collapses(H3, theta=-0.2)
    G3 = compute_G(V3, C3)
    print(f"V3: {V3:.4f}, C3: {C3}, G3: {G3:.4f}")

if __name__ == "__main__":
    test_metrics()
