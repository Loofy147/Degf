import torch
from transformer_lens import HookedTransformer
from monitor_gpt2 import DEGFMonitor
import numpy as np

def run_real_trt():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    monitor = DEGFMonitor(model)

    tasks = [
        {"name": "Syllogism (Deductive)", "prompt": "All men are mortal. Socrates is a man. Therefore, Socrates is mortal.", "type": "Deductive"},
        {"name": "Math (Deductive)", "prompt": "The square root of 64 is 8. The square root of 81 is 9.", "type": "Deductive"},
        {"name": "Pattern (Inductive)", "prompt": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10", "type": "Inductive"},
        {"name": "Factual (Inductive)", "prompt": "The capital of France is Paris. The capital of Germany is Berlin.", "type": "Inductive"}
    ]

    print(f"{'Task':<25} | {'Mean G':<10} | {'Type':<10}")
    print("-" * 50)

    results = []
    for task in tasks:
        g_stream = monitor.monitor_step(task["prompt"])
        # We exclude the first few tokens (burn-in)
        gs = [e["G"] for e in g_stream[5:]]
        mean_g = np.mean(gs) if gs else 0.0
        print(f"{task['name']:<25} | {mean_g:.4f}     | {task['type']}")
        results.append({"name": task["name"], "type": task["type"], "G": mean_g})

    deductive_gs = [r["G"] for r in results if r["type"] == "Deductive"]
    inductive_gs = [r["G"] for r in results if r["type"] == "Inductive"]

    print("\n--- Summary ---")
    print(f"Mean Deductive G: {np.mean(deductive_gs):.4f}")
    print(f"Mean Inductive G: {np.mean(inductive_gs):.4f}")
    print(f"TRT Gap: {np.mean(deductive_gs) - np.mean(inductive_gs):.4f}")

if __name__ == "__main__":
    run_real_trt()
