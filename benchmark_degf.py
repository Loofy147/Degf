import torch
import numpy as np
from transformer_lens import HookedTransformer
from ablation_a3 import evaluate_accuracy, get_ioi_benchmark, get_induction_benchmark
from degf_v2 import detect_cascade_chains
from monitor_gpt2 import scan_model

def run_benchmarks(model, name):
    print(f"\nBenchmarking {name}...")
    ioi_data = get_ioi_benchmark()
    ind_data = get_induction_benchmark()

    ioi_acc = evaluate_accuracy(model, ioi_data)
    ind_acc = evaluate_accuracy(model, ind_data)

    # Run scan for metrics
    prompts = ["When John and Mary went to the store, John gave a drink to Mary"]
    profiles = scan_model(model, prompts)

    # Mock ModelScan for cascade detector
    from degf_core import ModelScan
    scan = ModelScan(n_layers=model.cfg.n_layers, n_heads=model.cfg.n_heads, profiles=profiles)
    chains = detect_cascade_chains(scan)

    q2_count = len([p for p in profiles if p.G > 0.5])
    max_chain = chains[0].length if chains else 0

    print(f"  IOI Accuracy: {ioi_acc:.2%}")
    print(f"  Induction Accuracy: {ind_acc:.2%}")
    print(f"  Q2 Head Count: {q2_count}")
    print(f"  Max Cascade Chain: {max_chain}")

    return {"ioi": ioi_acc, "q2": q2_count, "chain": max_chain}

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. Baseline
    base_model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    base_res = run_benchmarks(base_model, "Baseline GPT-2")

    # 2. Thermo-trained (Simulate by loading previously fine-tuned state if possible,
    # or just run the benchmarks on the 'model' from the training script if it were kept)
    # Since we didn't save, we'll run a quick fine-tuning here to show the delta.
    from train_thermo import compute_thermo_loss
    import torch.optim as optim

    print("\n--- Applying L_thermo Fine-Tuning ---")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    texts = ["John gave the book to Mary because John liked Mary."]
    tokens = model.to_tokens(texts).to(device)

    for _ in range(5):
        optimizer.zero_grad()
        loss, _, _ = compute_thermo_loss(model, tokens)
        loss.backward()
        optimizer.step()

    thermo_res = run_benchmarks(model, "Thermo-Trained GPT-2")

    print("\n--- Final Comparison ---")
    print(f"  Q2 Density Delta: {thermo_res['q2'] - base_res['q2']:+d} heads")
    print(f"  Chain Length Delta: {thermo_res['chain'] - base_res['chain']:+d} links")
