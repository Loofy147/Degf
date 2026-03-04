import torch
import numpy as np
from transformer_lens import HookedTransformer
from monitor_gpt2 import DEGFMonitor

def run_cot_lift_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    monitor = DEGFMonitor(model)

    # Task: Simple reasoning
    prompt_direct = "Q: If all A are B, and all B are C, then all A are? A:"
    prompt_cot = "Q: If all A are B, and all B are C, then all A are? Let's think step by step. All A are B. All B are C. Therefore, all A are"

    print("--- Direct Answer ---")
    g_direct = monitor.monitor_step(prompt_direct)
    mean_g_direct = np.mean([entry['G'] for entry in g_direct])
    print(f"Mean G (Direct): {mean_g_direct:.4f}")

    print("\n--- Chain-of-Thought ---")
    g_cot = monitor.monitor_step(prompt_cot)
    # Focus on the 'reasoning' part of CoT
    mean_g_cot = np.mean([entry['G'] for entry in g_cot])
    print(f"Mean G (CoT): {mean_g_cot:.4f}")

    print(f"\nCoT G-lift: {mean_g_cot - mean_g_direct:+.4f}")

if __name__ == "__main__":
    run_cot_lift_experiment()
