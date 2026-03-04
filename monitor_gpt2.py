import torch
import numpy as np
from transformer_lens import HookedTransformer
from degf_core import compute_H_series, compute_V, compute_G, count_collapses, HeadProfile

def scan_model(model, prompts):
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    head_history = {(l, h): [] for l in range(n_layers) for h in range(n_heads)}

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        for l in range(n_layers):
            pattern = cache["pattern", l]
            for h in range(n_heads):
                attn = pattern[h].cpu().numpy()
                H = compute_H_series(attn)
                p = HeadProfile(layer=l, head=h, entropy_series=H, token_cost=0.5)
                head_history[(l, h)].append(p)

    averaged_profiles = []
    for (l, h), profiles in head_history.items():
        avg_V = np.mean([p.V for p in profiles])
        avg_C = np.mean([p.C for p in profiles])
        avg_G = np.mean([p.G for p in profiles])
        rep = HeadProfile(layer=l, head=h, entropy_series=profiles[0].entropy_series, token_cost=0.5)
        object.__setattr__(rep, 'V', avg_V)
        object.__setattr__(rep, 'C', avg_C)
        object.__setattr__(rep, 'G', avg_G)
        averaged_profiles.append(rep)

    return averaged_profiles

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    prompts = [
        "When John and Mary went to the store, John gave a drink to Mary",
        "After Alice and Bob finished lunch, Alice gave a book to Bob",
        "The quick brown fox jumps over the lazy dog",
        "1 2 3 1 2 3",
        "A B C A B C"
    ]
    profiles = scan_model(model, prompts)

    # Q2 targets: High G, Late layers (>= 6)
    q2_targets = [p for p in profiles if p.G > 0.5 and p.layer >= 6]

    # If too many, take top G
    if len(q2_targets) > 36:
        q2_targets = sorted(q2_targets, key=lambda x: -x.G)[:36]

    print(f"Identified {len(q2_targets)} Q2 targets in late layers.")
    for p in q2_targets[:10]:
        print(f"L{p.layer}H{p.head} -> G={p.G:.4f}")

    with open("q2_targets.txt", "w") as f:
        for p in q2_targets:
            f.write(f"{p.layer},{p.head}\n")
