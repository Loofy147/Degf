import torch
import numpy as np
from transformer_lens import HookedTransformer
from monitor_gpt2 import scan_model
from ablation_a3 import evaluate_accuracy, get_ioi_benchmark, get_induction_benchmark, run_ablation

def run_medium_a3():
    device = "cpu" # We'll use CPU due to no GPU and 8GB RAM
    print("Loading GPT-2-medium...")
    model = HookedTransformer.from_pretrained("gpt2-medium", device=device)

    # Identify Q2 targets in layers 16-23 (top 35% of 24 layers)
    # The scan will help us find the best targets.
    print("Scanning GPT-2-medium for Q2 targets...")
    prompts = ["When John and Mary went to the store, John gave a drink to Mary"]
    profiles = scan_model(model, prompts)

    # Q2 Target Criteria: G >= 0.5, V > 0.10, C >= 1, layer in late-mid/late
    targets = [(p.layer, p.head) for p in profiles if p.G >= 0.5 and p.V > 0.10 and p.C >= 1 and p.layer >= 16]

    print(f"Identified {len(targets)} Q2 targets in layers 16-23.")

    # Benchmarks
    ioi_data = get_ioi_benchmark()
    ind_data = get_induction_benchmark()

    # 1. Baseline
    print("\n[Baseline Accuracy]")
    ioi_base = evaluate_accuracy(model, ioi_data)
    ind_base = evaluate_accuracy(model, ind_data)
    print(f"  IOI: {ioi_base:.2%} | Induction: {ind_base:.2%}")

    # 2. Ablation
    print(f"\n[Ablated (Q2) Accuracy - {len(targets)} heads]")
    # TransformerLens handles mean ablation by setting activation to 0 or mean
    # run_ablation in ablation_a3.py does this
    ioi_abl = run_ablation(model, targets, ioi_data)
    ind_abl = run_ablation(model, targets, ind_data)
    print(f"  IOI: {ioi_abl:.2%} | Induction: {ind_abl:.2%}")

    # 3. Double-Dissociation Delta
    print(f"\n[Delta Results]")
    print(f"  IOI Drop: {ioi_abl - ioi_base:.2%}")
    print(f"  Induction Drop: {ind_abl - ind_base:.2%}")

if __name__ == "__main__":
    run_medium_a3()
