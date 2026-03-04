import torch
import numpy as np
from transformer_lens import HookedTransformer
from monitor_gpt2 import scan_model
from degf_core import classify_quadrant

def run_prompt_stability():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    # 5 semantically equivalent prompts (IOI-like)
    prompts = [
        "When John and Mary went to the store, John gave a drink to Mary.",
        "John gave Mary a drink after they both went to the store.",
        "After going to the shop, John handed Mary a refreshing beverage.",
        "Mary received a drink from John once they arrived at the store.",
        "Upon arriving at the market, John provided Mary with a drink."
    ]

    all_head_G = {}

    print(f"Scanning {len(prompts)} prompt variations...")
    for i, p in enumerate(prompts):
        profiles = scan_model(model, [p])
        for prof in profiles:
            key = (prof.layer, prof.head)
            all_head_G.setdefault(key, []).append(prof.G)

    results = []
    for (l, h), G_list in all_head_G.items():
        G_arr = np.array(G_list)
        mean_G = float(np.mean(G_arr))
        std_G = float(np.std(G_arr))
        cv = std_G / max(mean_G, 1e-6)

        # Determine quadrant based on mean
        # use 0.5 as absolute G threshold for Q2 (Genuine)
        if mean_G >= 0.5:
            quadrant = "Q2"
        else:
            quadrant = "Q3"

        results.append({
            "head": (l, h),
            "mean_G": mean_G,
            "std_G": std_G,
            "cv": cv,
            "quadrant": quadrant
        })

    q2_heads = [r for r in results if r['quadrant'] == 'Q2']
    q2_heads.sort(key=lambda x: x['cv'])

    print("\n--- Top 10 Most Stable Q2 (Reasoning) Heads ---")
    print(f"{'Head':<10} | {'Mean G':<8} | {'CV':<8}")
    print("-" * 30)
    for r in q2_heads[:10]:
        print(f"L{r['head'][0]}H{r['head'][1]:<7} | {r['mean_G']:.4f} | {r['cv']:.4f}")

    q3_heads = [r for r in results if r['quadrant'] == 'Q3']
    q3_heads.sort(key=lambda x: x['cv'])

    print("\n--- Top 10 Most Stable Q3 (Mechanical) Heads ---")
    print(f"{'Head':<10} | {'Mean G':<8} | {'CV':<8}")
    print("-" * 30)
    for r in q3_heads[:10]:
        print(f"L{r['head'][0]}H{r['head'][1]:<7} | {r['mean_G']:.4f} | {r['cv']:.4f}")

    mean_cv_q2 = np.mean([r['cv'] for r in q2_heads]) if q2_heads else 0
    mean_cv_q3 = np.mean([r['cv'] for r in q3_heads]) if q3_heads else 0
    print(f"\nMean CV Q2: {mean_cv_q2:.4f}")
    print(f"Mean CV Q3: {mean_cv_q3:.4f}")
    print(f"Stability Ratio (Q2/Q3): {mean_cv_q2 / max(mean_cv_q3, 1e-6):.4f}")

if __name__ == "__main__":
    run_prompt_stability()
